from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArgs:
    vocab_size: int = 33
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_len: int = 512
    rope_base: float = 1000.0
    dropout_rate: float = 0.0
    single_state_token_length: Optional[int] = None
    record_token_length: Optional[int] = None


@dataclass
class TrainingArgs:
    batch_size: int = 128 * 4
    micro_batch_size: int = 64 * 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4 * 4
    muon_learning_rate: float = 2e-2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    max_iters: int = 10000  # 700M tokens roughly
    lr_decay_iters: Optional[int] = None
    min_lr: float = 3e-5 * 4
    min_lr_muon: float = 2e-3
    warmup_iters: int = 1000
    optimizer_name: str = 'muon_with_adamw'
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class DataArgs:
    data_dir: str = "better_data"
    val_data_dir: Optional[str] = "better_val_data"
    num_workers_dataloader: int = 2


@dataclass
class LoggingArgs:
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 500
    out_dir: str = "out_muon"
    run_name: Optional[str] = "interpreter_train"


@dataclass
class DeviceArgs:
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True


@dataclass
class DistributedArgs:
    use_ddp: bool = True
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0  # Will be updated by DDP setup
    local_rank: int = 0  # Will be updated by DDP setup


@dataclass
class InitArgs:
    init_from: str = "scratch"
    resume_from_checkpoint_path: Optional[str] = None
    always_save_checkpoint: bool = False

@dataclass
class TrainConfig:
    model_args: ModelArgs = field(default_factory=ModelArgs)
    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    logging_args: LoggingArgs = field(default_factory=LoggingArgs)
    device_args: DeviceArgs = field(default_factory=DeviceArgs)
    distributed_args: DistributedArgs = field(default_factory=DistributedArgs)
    init_args: InitArgs = field(default_factory=InitArgs)

    def __post_init__(self):
        actual_world_size = self.distributed_args.world_size
    
        if self.training_args.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
    
        if actual_world_size > 0 and self.training_args.batch_size > 0:
            if self.training_args.batch_size % (self.training_args.micro_batch_size * actual_world_size) != 0:
                raise ValueError(
                    f"Global batch_size ({self.training_args.batch_size}) must be divisible by "
                    f"micro_batch_size ({self.training_args.micro_batch_size}) * world_size ({actual_world_size})"
                )
            self.training_args.gradient_accumulation_steps = (
                self.training_args.batch_size // (self.training_args.micro_batch_size * actual_world_size)
            )
        elif self.training_args.gradient_accumulation_steps <= 0:
            self.training_args.gradient_accumulation_steps = 1
    
        if self.training_args.lr_decay_iters is None:
            self.training_args.lr_decay_iters = self.training_args.max_iters
    
        if actual_world_size > 0:
            self.effective_batch_size = (
                self.training_args.micro_batch_size * 
                self.training_args.gradient_accumulation_steps * 
                actual_world_size
            )
            print(f"Initialized/Updated TrainConfig. Grad Accum: {self.training_args.gradient_accumulation_steps}, "
                  f"Effective global batch size: {self.effective_batch_size}, WorldSize: {actual_world_size}")
            if self.effective_batch_size != self.training_args.batch_size:
                print(f"Warning: Effective batch size ({self.effective_batch_size}) does not match "
                      f"target global batch_size ({self.training_args.batch_size}).")
