import os
import time
from datetime import datetime
from typing import Optional, Tuple, Union

from dataset import StateTransitionDataset

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader, DistributedSampler

from config_muon import TrainingArgs, TrainConfig
from muon import MuonWithAuxAdam
from model import GPT

from utils import (
    setup_ddp,
    cleanup_ddp,
    get_lr,
    get_lr_muon,
    calculate_eval_loss,
    calculate_autoregressive_eval_metrics,
    show_in_training_samples
)

def train():
    """Main training function."""
    config = TrainConfig()
    
    setup_ddp(config)
    
    is_master_process = (config.distributed_args.rank == 0)
    if is_master_process:
        os.makedirs(config.logging_args.out_dir, exist_ok=True)

    torch.manual_seed(config.training_args.seed + config.distributed_args.rank)
    torch_dtype = {
        'float32': torch.float32, 
        'bfloat16': torch.bfloat16, 
        'float16': torch.float16
    }[config.device_args.dtype]
    device_type = 'cuda' if 'cuda' in config.device_args.device else 'cpu'
    ctx = torch.amp.autocast(
        device_type=device_type, 
        dtype=torch_dtype, 
        enabled=(torch_dtype != torch.float32)
    )

    if is_master_process:
        print("Loading dataset...")
    
    train_dataset = StateTransitionDataset(data_dir=config.data_args.data_dir)
    val_dataset = StateTransitionDataset(data_dir=config.data_args.val_data_dir)

    config.model_args.vocab_size = train_dataset.vocab_size
    config.model_args.single_state_token_length = train_dataset.single_state_token_length
    config.model_args.record_token_length = train_dataset.record_token_length

    if config.distributed_args.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=config.distributed_args.world_size, 
            rank=config.distributed_args.rank, 
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=config.distributed_args.world_size, 
            rank=config.distributed_args.rank, 
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training_args.micro_batch_size, 
        sampler=train_sampler, 
        num_workers=config.data_args.num_workers_dataloader, 
        pin_memory=True, 
        shuffle=(train_sampler is None)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training_args.micro_batch_size, 
        sampler=val_sampler, 
        num_workers=config.data_args.num_workers_dataloader
    )
    
    if is_master_process:
        print(f"Train dataset size: {len(train_dataset)} records")

    model_args_dict = vars(config.model_args)
    expected_gpt_args = GPT.__init__.__code__.co_varnames[1:]
    gpt_init_args = {k: v for k, v in model_args_dict.items() if k in expected_gpt_args}
    model = GPT(**gpt_init_args)
    model.to(config.distributed_args.local_rank if config.distributed_args.use_ddp else config.device_args.device)
    
    if config.device_args.compile_model and hasattr(torch, 'compile'):
        if is_master_process:
            print("Compiling model...")
        model = torch.compile(model)

    if config.distributed_args.use_ddp:
        model = DDP(model, device_ids=[config.distributed_args.local_rank])
    
    raw_model = model.module if config.distributed_args.use_ddp else model

    if hasattr(raw_model, 'configure_optimizers'):
        optimizer = raw_model.configure_optimizers(config.training_args)
    else:
        params_muon = []
        params_adamw = []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            is_lm_head_or_embedding = name.startswith('lm_head.') or name.startswith('embedding.')
            if not is_lm_head_or_embedding and param.dim() >= 2:
                params_muon.append(param)
            else:
                params_adamw.append(param)

        param_groups = [
            dict(params=params_muon, use_muon=True,
                 lr=config.training_args.muon_learning_rate, weight_decay=config.training_args.weight_decay),
            dict(params=params_adamw, use_muon=False,
                 lr=config.training_args.learning_rate, betas=(config.training_args.beta1, config.training_args.beta2), weight_decay=config.training_args.weight_decay, eps=config.training_args.adam_eps),
           ]

        optimizer = MuonWithAuxAdam(param_groups)

        
    profile_skip_iters = 200
    profile_wait_steps = 5
    profile_warmup_steps = 5
    profile_active_steps = 10
    profile_repeat_cycles = 1

    profiler_output_dir = os.path.join(
        config.logging_args.out_dir, 
        f"profiler_traces_rank{config.distributed_args.rank}"
    )
    if config.distributed_args.rank == 0:
        os.makedirs(os.path.dirname(profiler_output_dir), exist_ok=True)
    os.makedirs(profiler_output_dir, exist_ok=True)

    if is_master_process: 
        print(f"Profiler will start after {profile_skip_iters} iterations.")
        print(f"Profiling window: wait={profile_wait_steps}, warmup={profile_warmup_steps}, "
              f"active={profile_active_steps}, repeat={profile_repeat_cycles}")

    iter_num = 0
    best_val_loss = float('inf')
    
    if config.init_args.init_from == 'resume':
        resume_path = config.init_args.resume_from_checkpoint_path
        if resume_path and os.path.exists(resume_path):
            if is_master_process:
                print(f"Resuming training from checkpoint: {resume_path}")
            
            map_location = (
                {'cuda:%d' % 0: 'cuda:%d' % config.distributed_args.local_rank} 
                if config.distributed_args.use_ddp 
                else config.device_args.device
            )
            checkpoint = torch.load(resume_path, map_location=map_location)
    
            state_dict = checkpoint['model']
            unwanted_prefix = 'module.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            raw_model.load_state_dict(state_dict)
    
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
    
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            
            if is_master_process:
                print(f"Resumed from iteration {iter_num-1}, best_val_loss: {best_val_loss}")
        else:
            if is_master_process:
                print(f"Resume path {resume_path} not found. Starting from scratch.")
            config.init_args.init_from = "scratch"
    elif config.init_args.init_from != "scratch":
        if is_master_process:
            print(f"Initializing from type '{config.init_args.init_from}' not fully implemented. "
                  f"Starting from scratch.")

    if is_master_process:
        print(f"Starting training for {config.training_args.max_iters} iterations...")
    
    current_data_iter = iter(train_loader)
    
    if config.model_args.single_state_token_length is None and is_master_process:
        print("Warning: model_args.single_state_token_length is not set. Autoregressive eval might fail.")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=profile_skip_iters, 
            wait=profile_wait_steps,
            warmup=profile_warmup_steps,
            active=profile_active_steps,
            repeat=profile_repeat_cycles 
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        for iter_num in range(iter_num, config.training_args.max_iters):
            t0 = time.time()

            lr = get_lr(iter_num, config.training_args)
            lr_muon = get_lr_muon(iter_num, config.training_args)
            for param_group in optimizer.param_groups:
                if param_group['use_muon']:
                    param_group['lr'] = lr_muon
                else:
                    param_group['lr'] = lr

            if ((iter_num == config.training_args.max_iters - 1) or 
                (iter_num > 0 and iter_num % config.logging_args.eval_interval == 0)):
                
                if is_master_process:
                    print(f"\n--- Evaluation at Iter {iter_num} ---")
                
                if config.data_args.val_data_dir:
                    eval_device = (config.distributed_args.local_rank 
                                 if config.distributed_args.use_ddp 
                                 else config.device_args.device)
                    
                    assert val_loader is not None
                    eval_loss = calculate_eval_loss(raw_model, val_loader, ctx, config, eval_device)
                    autoregressive_exact_match, autoregressive_frac_acc = float('nan'), float('nan')
                    
                    if (config.model_args.single_state_token_length is not None and 
                        config.model_args.single_state_token_length > 0):
                        autoregressive_exact_match, autoregressive_frac_acc = calculate_autoregressive_eval_metrics(
                            raw_model, val_loader, ctx, config, eval_device
                        )
                    elif is_master_process:
                        print(f"  Skipping autoregressive eval: single_state_token_length not properly set "
                              f"(value: {config.model_args.single_state_token_length}).")
                    
                    if is_master_process:
                        print(f"  Iter {iter_num:6d} | Eval Loss (XEnt): {eval_loss:.4f}")
                        print(f"  Iter {iter_num:6d} | AR Exact Match:   {autoregressive_exact_match:.4f}")
                        print(f"  Iter {iter_num:6d} | AR Frac. Acc:     {autoregressive_frac_acc:.4f}")
                    
                    if eval_loss < best_val_loss or config.init_args.always_save_checkpoint:
                        best_val_loss = eval_loss
                        if is_master_process:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': config.model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }
                            ckpt_path = os.path.join(config.logging_args.out_dir, f'ckpt_best_val.pt')
                            print(f"Saving best validation checkpoint to {ckpt_path}")
                            torch.save(checkpoint, ckpt_path)
                else:
                    if is_master_process:
                        print(f"  Iter {iter_num:6d} | Skipping evaluation: val_data_dir not set or val_loader is None.")
                
                if is_master_process:
                    print(f"--- End Evaluation ---\n")

            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            for micro_step in range(config.training_args.gradient_accumulation_steps):
                if config.distributed_args.use_ddp and train_sampler is not None:
                    epoch = (iter_num * config.training_args.batch_size) // len(train_dataset)
                    train_sampler.set_epoch(epoch)
                
                try:
                    batch = next(current_data_iter)
                except StopIteration:
                    current_data_iter = iter(train_loader)
                    batch = next(current_data_iter)

                x = batch.to(
                    config.distributed_args.local_rank 
                    if config.distributed_args.use_ddp 
                    else config.device_args.device
                )
                input_ids = x[:, :-1]
                targets = x[:, 1:]
                
                if (micro_step == 0 and is_master_process and 
                    iter_num % config.logging_args.log_interval == 0 and iter_num > 0):
                    show_in_training_samples(
                        raw_model=raw_model,
                        current_training_batch=x,
                        train_dataset=train_dataset,
                        device=torch.device(device_type),
                        config=config,
                        iter_num=iter_num,
                        num_samples_to_show=5
                    )

                with ctx:
                    logits = model(input_ids)
                    
                    k_loss_calc = config.model_args.single_state_token_length
                    if k_loss_calc is None:
                        if is_master_process:
                            print("Error: single_state_token_length (k) not set for loss calculation.")
                    
                    start_index_for_loss = k_loss_calc - 1
                    
                    if start_index_for_loss < 0 or start_index_for_loss >= logits.size(1):
                        print(f"Warning: Calculated start_index_for_loss ({start_index_for_loss}) is out of bounds "
                              f"for sequence length {logits.size(1)}. Using full sequence for loss. "
                              f"(k_loss_calc={k_loss_calc}, record_token_length={config.model_args.record_token_length})")
                        logits_for_loss = logits
                        targets_for_loss = targets
                    else:
                        logits_for_loss = logits[:, start_index_for_loss:]
                        targets_for_loss = targets[:, start_index_for_loss:]
 
                    loss = F.cross_entropy(
                        logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                        targets_for_loss.reshape(-1), 
                        ignore_index=-1
                    )
                    loss = loss / config.training_args.gradient_accumulation_steps
       
                if config.distributed_args.use_ddp:
                    model.require_backward_grad_sync = (micro_step == config.training_args.gradient_accumulation_steps - 1)
                
                loss.backward()

            if config.training_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_args.grad_clip)
            
            optimizer.step()

            if device_type == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            
            if iter_num % config.logging_args.log_interval == 0 and is_master_process:
                lossf = loss.item() * config.training_args.gradient_accumulation_steps
                print(f"Iter {iter_num:6d} | Loss: {lossf:.8f} | Muon LR: {lr_muon:.2e} | AdamW LR: {lr:.2e} | dt: {dt*1000:.2f}ms | time = {datetime.now()}")

            if iter_num > 0 and iter_num % config.logging_args.save_interval == 0 and is_master_process:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config.model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                ckpt_path = os.path.join(config.logging_args.out_dir, f'ckpt_iter_{iter_num}.pt')
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint, ckpt_path)

            prof.step()
            
    if is_master_process:
        print("Training finished.")
    
    train_dataset.close()
    val_dataset.close()
    cleanup_ddp(config)


if __name__ == '__main__':
    # run with `torchrun --standalone --nproc_per_node=2`
    train()
