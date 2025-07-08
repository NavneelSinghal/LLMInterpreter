import os
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import TrainConfig, TrainingArgs, ModelArgs
from dataset import StateTransitionDataset

class eval_mode_for_model:
    
    def __init__(self, model_to_eval: nn.Module):
        self.model = model_to_eval
        self.original_mode = None

    def __enter__(self):
        self.original_mode = self.model.training
        self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_mode is not None:
            self.model.train(self.original_mode)


@torch.no_grad()
def calculate_eval_loss(model, val_loader, ctx, config: TrainConfig, device):
    with eval_mode_for_model(model):
        losses = []
        k = config.model_args.single_state_token_length
        if k is None:
            print("Warning: single_state_token_length (k) not set in config for eval. Using full sequence.")
        
        for batch in val_loader:
            x = batch.to(device)
            input_ids = x[:, :-1]
            targets = x[:, 1:]
            
            with ctx:
                logits = model(input_ids)

                if k is not None:
                    start_index_for_loss = k - 1
                    if start_index_for_loss < 0 or start_index_for_loss >= logits.size(1):
                        print(f'Error: issue with logits and targets indexing')
                        return float('nan')
                    else:
                        logits_for_loss = logits[:, start_index_for_loss:]
                        targets_for_loss = targets[:, start_index_for_loss:]
                else:
                    print(f'Error: issue with logits and targets indexing')
                    return float('nan')
                    
                loss = F.cross_entropy(
                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)), 
                    targets_for_loss.reshape(-1), 
                    ignore_index=-1
                )
            losses.append(loss.item())
        
        if not losses:
            return float('nan')

        avg_loss = torch.tensor(losses, device=device).mean()
        if config.distributed_args.use_ddp and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.item()


@torch.no_grad()
def calculate_autoregressive_eval_metrics(model, val_loader, ctx, config: TrainConfig, device):
    total_exact_matches = 0
    sum_fractional_accuracy = 0.0
    total_samples_processed = 0
    
    k = config.model_args.single_state_token_length
    if k is None or k <= 0:
        if config.distributed_args.rank == 0:
            print("Error (autoregressive_eval): model_args.single_state_token_length (k) must be a positive integer.")
        return float('nan'), float('nan')
    
    num_tokens_to_predict = k
    prompt_length = k

    expected_record_length = prompt_length + num_tokens_to_predict
    if config.model_args.record_token_length != expected_record_length:
        if config.distributed_args.rank == 0:
            print(f"Warning (autoregressive_eval): record_token_length ({config.model_args.record_token_length}) "
                  f"does not match expected prompt_length ({prompt_length}) + num_tokens_to_predict ({num_tokens_to_predict}). "
                  f"This might lead to incorrect evaluation slicing if ground truth is shorter than num_tokens_to_predict.")

    with eval_mode_for_model(model):
        for batch_idx, batch_data in enumerate(val_loader):
            full_sequences = batch_data.to(device)
            
            current_batch_actual_record_length = full_sequences.size(1)
            if current_batch_actual_record_length < prompt_length:
                print(f"Warning (autoregressive_eval): Batch {batch_idx} record length {current_batch_actual_record_length} "
                      f"is less than prompt_length {prompt_length}. Skipping batch.")
                continue

            prompt_tokens = full_sequences[:, :prompt_length]
            actual_gt_tokens_available_for_prediction = max(0, current_batch_actual_record_length - prompt_length)
            ground_truth_to_compare = full_sequences[:, prompt_length:prompt_length + actual_gt_tokens_available_for_prediction]
            
            if actual_gt_tokens_available_for_prediction == 0 and num_tokens_to_predict > 0:
                print(f"Warning (autoregressive_eval): Batch {batch_idx} has no ground truth tokens after prompt. "
                      f"Skipping sample for metrics calculation.")

            generated_output_full = model.generate(prompt_tokens, num_tokens_to_predict, temperature=0.0) 
            predicted_tokens = generated_output_full[:, prompt_length:]
            current_micro_batch_size = predicted_tokens.size(0)
            
            for i in range(current_micro_batch_size):
                gt_single = ground_truth_to_compare[i] 
                pred_single = predicted_tokens[i]      
                len_gt = gt_single.size(0)
                len_pred = pred_single.size(0)
                
                if len_gt == 0:
                    if num_tokens_to_predict > 0:
                        sum_fractional_accuracy += 0.0 
                    total_samples_processed += 1
                    continue

                compare_len_frac = min(len_pred, len_gt)
                matches = (pred_single[:compare_len_frac] == gt_single[:compare_len_frac]).float().sum()
                fractional_acc_sample = matches / len_gt
                sum_fractional_accuracy += fractional_acc_sample.item()
                
                is_exact_match = (len_pred >= len_gt) and (matches == len_gt)
                if is_exact_match:
                    total_exact_matches += 1
            
            total_samples_processed += current_micro_batch_size
            if total_samples_processed >= 1000:
                break

        stats_tensor = torch.tensor([
            total_exact_matches, 
            sum_fractional_accuracy, 
            total_samples_processed
        ], dtype=torch.float64, device=device)

        if config.distributed_args.use_ddp and dist.is_initialized():
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        global_exact_matches = stats_tensor[0].item()
        global_sum_fractional_accuracy = stats_tensor[1].item()
        global_total_samples = stats_tensor[2].item()

        avg_exact_match = global_exact_matches / global_total_samples if global_total_samples > 0 else 0.0
        avg_fractional_accuracy = global_sum_fractional_accuracy / global_total_samples if global_total_samples > 0 else 0.0
    
    return avg_exact_match, avg_fractional_accuracy


@torch.no_grad()
def show_in_training_samples(raw_model,
                           current_training_batch: torch.Tensor, 
                           train_dataset: StateTransitionDataset, 
                           device: torch.device,
                           config: TrainConfig,
                           iter_num: int,
                           num_samples_to_show: int = 2,
                           top_k_probs_to_show: int = 5):
    if not (config.model_args.single_state_token_length and config.model_args.single_state_token_length > 0):
        if config.distributed_args.rank == 0:
            print(f"Iter {iter_num}: Skipping in-training sample generation, "
                  f"single_state_token_length not set properly.")
        return

    k = config.model_args.single_state_token_length
    prompt_length = k
    num_tokens_to_generate = k

    with eval_mode_for_model(raw_model):
        _, to_string_fn = train_dataset.get_tokenizer()

        num_to_actually_show = min(num_samples_to_show, current_training_batch.size(0))
        if num_to_actually_show == 0:
            return
        
        sample_indices_in_batch = random.sample(range(current_training_batch.size(0)), num_to_actually_show)
        
        if config.distributed_args.rank == 0:
            print(f"\n===== In-Training Samples at Iter {iter_num} (from current batch) =====")

        for i, batch_idx in enumerate(sample_indices_in_batch):
            if config.distributed_args.rank == 0:
                print(f" --- Sample {i+1}/{num_to_actually_show} (from batch) ---")
            
            full_record_ids = current_training_batch[batch_idx] 

            if full_record_ids.size(0) < prompt_length:
                if config.distributed_args.rank == 0:
                    print(f"  Skipping sample: record too short for prompt "
                          f"({full_record_ids.size(0)} < {prompt_length})")
                continue

            prompt_ids_single = full_record_ids[:prompt_length]

            gt_continuation_ids = torch.empty(0, dtype=torch.long, device=full_record_ids.device)
            len_gt_continuation = 0
            
            if full_record_ids.size(0) >= prompt_length + num_tokens_to_generate:
                gt_continuation_ids = full_record_ids[prompt_length:prompt_length + num_tokens_to_generate]
            elif full_record_ids.size(0) > prompt_length:
                gt_continuation_ids = full_record_ids[prompt_length:]
            len_gt_continuation = gt_continuation_ids.size(0)
            
            prompt_ids_batched_for_ar = prompt_ids_single.unsqueeze(0)
            generated_full_ids_ar = raw_model.generate(
                prompt_tokens=prompt_ids_batched_for_ar, 
                num_tokens_to_generate=num_tokens_to_generate,
                temperature=0.0 
            )

            generated_continuation_ids_ar = generated_full_ids_ar[0, prompt_length:]

            predicted_continuation_ids_tf = torch.empty(0, dtype=torch.long, device=full_record_ids.device)
            all_tf_logits_for_state2_segment = None 

            if full_record_ids.size(0) > 1:
                input_ids_for_tf_model_fwd = full_record_ids[:-1].unsqueeze(0)
                _all_logits_from_tf_fwd = raw_model(input_ids_for_tf_model_fwd, pos_offset=0)
                
                tf_logit_start_idx = prompt_length - 1
                num_tf_preds_to_get = min(num_tokens_to_generate, len_gt_continuation)

                if (tf_logit_start_idx >= 0 and 
                    (tf_logit_start_idx + num_tf_preds_to_get) <= _all_logits_from_tf_fwd.size(1) and 
                    num_tf_preds_to_get > 0):
                    
                    all_tf_logits_for_state2_segment = _all_logits_from_tf_fwd[
                        0, tf_logit_start_idx:tf_logit_start_idx + num_tf_preds_to_get, :
                    ]
                    predicted_continuation_ids_tf = torch.argmax(all_tf_logits_for_state2_segment, dim=-1)
                elif config.distributed_args.rank == 0 and num_tf_preds_to_get > 0:
                    print(f"  Warning: Could not extract relevant teacher-forced logits for TF preds in sample {i+1}.")
            
            if config.distributed_args.rank == 0:
                print(f"  Prompt:          '{to_string_fn(prompt_ids_single.tolist())}'")
                if len_gt_continuation > 0:
                    print(f"  GT Next:         '{to_string_fn(gt_continuation_ids.tolist())}' (len: {len_gt_continuation})")
                else:
                    print(f"  GT Next:         (Not enough tokens in record for GT continuation)")
                print(f"  AR Gen:          '{to_string_fn(generated_continuation_ids_ar.tolist())}' "
                      f"(len: {len(generated_continuation_ids_ar)})")
                if len(predicted_continuation_ids_tf) > 0:
                    print(f"  TF NextPreds:    '{to_string_fn(predicted_continuation_ids_tf.tolist())}' "
                          f"(len: {len(predicted_continuation_ids_tf)})")
                else:
                    print(f"  TF NextPreds:    (Could not be generated or N/A for sample {i+1})")
        
        if config.distributed_args.rank == 0:
            print(f"===== End In-Training Samples at Iter {iter_num} =====")


def setup_ddp(config: TrainConfig):
    if config.distributed_args.use_ddp:
        dist.init_process_group(backend=config.distributed_args.backend)
        config.distributed_args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        config.distributed_args.rank = int(os.environ.get('RANK', 0))
        config.distributed_args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(config.distributed_args.local_rank)
        print(f"DDP setup: World Size {config.distributed_args.world_size}, "
              f"Rank {config.distributed_args.rank}, Local Rank {config.distributed_args.local_rank}")
    else:
        config.distributed_args.world_size = 1
        config.distributed_args.rank = 0
        config.distributed_args.local_rank = 0
        if config.device_args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
        print(f"Non-DDP setup: Device {config.device_args.device}")
    
    config.__post_init__()


def cleanup_ddp(config: TrainConfig):
    if config.distributed_args.use_ddp:
        dist.destroy_process_group()


def get_lr(it, config: TrainingArgs):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def get_lr_muon(it, config: TrainingArgs):
    if it < config.warmup_iters:
        return config.muon_learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr_muon
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr_muon + coeff * (config.muon_learning_rate - config.min_lr_muon)
