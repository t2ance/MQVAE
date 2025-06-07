import warnings
from dataclasses import dataclass, field
from typing import Optional, Any, Literal, List

import torch


@dataclass
class TrainingArguments:
    title: str = field(
        default=''
    )

    # General Training Parameters
    train_batch_size: int = field(
        default=64,
        metadata={
            "help": "Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)"
        }
    )
    codebook_train_batch_size: int = field(
        default=None
    )
    generator_train_batch_size: int = field(
        default=None
    )
    discriminator_train_batch_size: int = field(
        default=None
    )
    eval_batch_size: int = field(
        default=None
    )

    hyper_gradient_type: Optional[str] = field(
        default='darts'
    )

    generator_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay of generator"},
    )
    generator_codebook_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay of generator-codebook"},
    )
    codebook_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay of codebook"},
    )
    discriminator_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay of discriminator"},
    )

    generator_codebook_lr: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Learning rate (generator-codebook lr)"},
    )
    generator_lr: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Learning rate (generator lr)"},
    )
    optimizer_codebook_type: Optional[float] = field(
        default='sgd'
    )
    codebook_lr: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Learning rate (codebook lr)"},
    )
    codebook_momentum: Optional[float] = field(
        default=0.9,
        metadata={"help": "Momentum (codebook uses SGD to optimize)"},
    )
    discriminator_lr: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Learning rate (discriminator lr)"},
    )
    generator_codebook_min_lr: float = field(
        default=0
    )
    generator_min_lr: float = field(
        default=0
    )
    codebook_min_lr: float = field(
        default=0
    )
    discriminator_min_lr: float = field(
        default=0
    )
    generator_codebook_scheduler_type: str = field(
        default='half-cosine'
    )
    generator_scheduler_type: str = field(
        default='half-cosine'
    )
    codebook_scheduler_type: str = field(
        default='half-cosine'
    )
    discriminator_scheduler_type: str = field(
        default='half-cosine'
    )
    beta1: float = field(
        default=0.5
    )
    beta2: float = field(
        default=0.9
    )
    codebook_beta1: float = field(
        default=None
    )
    codebook_beta2: float = field(
        default=None
    )
    eps: float = field(
        default=1e-7
    )
    cache_dir: str = field(
        default=None
    )
    generator_codebook_warmup_steps: int = field(
        default=None,
        metadata={"help": "Epochs to warmup LR for generator-codebook"},
    )
    generator_warmup_steps: int = field(
        default=None,
        metadata={"help": "Epochs to warmup LR for generator"},
    )
    discriminator_warmup_steps: int = field(
        default=None,
        metadata={"help": "Epochs to warmup LR for discriminator"},
    )
    codebook_warmup_steps: int = field(
        default=None,
        metadata={"help": "Epochs to warmup LR for codebook"},
    )
    log_dir: str = field(
        default=None,
        metadata={"help": "Path where to tensorboard log"},
    )
    vqgan_dir: str = field(
        default=None,
        metadata={"help": 'Load checkpoint of official checkpoint of vqgan'}
    )
    resume_dir: str = field(
        default=None,
        metadata={"help": "Resume from checkpoint"}
    )
    load_state: bool = field(
        default=True,
        metadata={"help": "Load the state of optimizer, scheduler and step, or not"}
    )
    load_problems: List[str] = field(
        default=None
    )

    device: str = field(
        default="cuda",
        metadata={"help": "Device to use for training / testing"},
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed"}
    )
    num_workers: int = field(
        default=None,
        metadata={"help": "Number of data loader workers"}
    )
    pin_mem: bool = field(
        default=True,
        metadata={
            "help": "Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU."
        },
    )

    # Distributed Training Parameters'
    world_size: int = field(
        default=1,
        metadata={"help": "Number of distributed processes"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training"}
    )
    timedelta: int = field(
        default=1800
    )

    vq_config_path: str = field(
        default="vqgan_configs/vq-f16.yaml",
        metadata={"help": "Path to VQ-GAN config file"}
    )
    image_size: int = field(
        default=None,
        metadata={"help": "Image size"}
    )

    disc_start: int = field(
        default=None,
        metadata={"help": "GAN loss start epoch"}
    )
    rate_q: float = field(
        default=0.1,
        metadata={"help": "Quantization loss rate"}
    )
    alpha: float = field(
        default=1.
    )
    beta: float = field(
        default=1.
    )
    gamma: float = field(
        default=0
    )
    codebook_rate_q: float = field(
        default=None,
        metadata={"help": "Quantization loss rate"}
    )
    generator_rate_q: float = field(
        default=None,
        metadata={"help": "Quantization loss rate"}
    )
    rate_p: float = field(
        default=None,
        metadata={"help": "VGG loss rate"}
    )
    rate_d: float = field(
        default=0.1,
        metadata={"help": "GAN loss rate"}
    )
    dataset: str = field(
        default="imagenet",
        metadata={"help": "Dataset name"}
    )
    gradient_accumulation: int = field(
        default=1
    )
    gradient_clipping: float = field(
        default=0.
    )
    train_steps: int = field(
        default=20000
    )
    valid_steps: int = field(
        default=None
    )
    log_steps: int = field(
        default=50,
        metadata={
            "help": 'Steps for logging training summary'
        }
    )
    save_steps: int = field(
        default=None,
        metadata={
            "help": 'Steps for saving model'
        }
    )
    cycle_steps: int = field(
        default=None
    )
    strategy: str = field(
        default='default'
    )
    precision: str = field(
        default='fp32'
    )

    drop_last: bool = field(
        default=False
    )
    warmup_ratio: Optional[float] = field(
        default=0.1
    )
    run: Any = field(
        default=None
    )
    hf_token: str = field(
        default=''
    )
    wandb_token: str = field(
        default='TOKEN'
    )

    n_vision_words: int = field(
        default=16384,
        metadata={"help": "Number of vision words"}
    )

    quantizer_type: str = field(
        default="hyper-net",
        metadata={"help": "Type of quantizer (EMA/ORG/HYPER-NET)"}
    )

    embed_dim: int = field(
        default=8,
        metadata={"help": "Feature dimension"}
    )
    hyper_net_hidden_dim: int = field(
        default=8,
        metadata={"help": "Hidden Feature dimension"}
    )
    mlp_hidden_dim: int = field(
        default=32,
        metadata={"help": "Hidden Feature dimension"}
    )
    z_channels: int = field(
        default=64,
        metadata={"help": "z channels for vqvae"}
    )
    resolution_factor: int = field(
        default=None,
        metadata={"help": "resolution factor for vqvae"}
    )
    optimization_type: str = field(
        default='meta'
    )
    hyper_net_type: str = field(
        default='linear'
    )
    perceptual_loss: torch.nn.Module = field(
        default=None
    )

    l2_proj: bool = field(
        default=False
    )

    generator_type: Optional[Literal['vqgan', 'vqvae']] = field(
        default='vqgan'
    )

    image_height: Optional[int] = field(
        default=None
    )
    image_width: Optional[int] = field(
        default=None
    )

    max_train_samples: int = field(
        default=-1
    )
    max_val_samples: int = field(
        default=None
    )

    streaming: Optional[bool] = field(
        default=True
    )
    trailing_norm: Optional[bool] = field(
        default=True
    )

    def is_distributed(self):
        return self.world_size > 1

    def is_main_process(self):
        return self.local_rank == 0 or self.local_rank == -1

    def log(self, message, all_: bool = False):
        if all_ or self.is_main_process():
            print(message)

    def set_num_workers(self, default=1):
        if self.num_workers is None:
            try:
                import psutil
                cpu_count = len(psutil.Process().cpu_affinity())
                if cpu_count is None:
                    raise ValueError
                workers = max(1, cpu_count // (2 * self.world_size))
            except:
                workers = default
                warnings.warn(f"Can not get number of cpu. Using num_workers={workers}")

            self.num_workers = workers

        self.log(f'Working with {self.num_workers} cpus.')

    def __post_init__(self):

        if self.codebook_rate_q is None:
            self.codebook_rate_q = self.rate_q
        if self.generator_rate_q is None:
            self.generator_rate_q = self.rate_q
        if self.eval_batch_size is None:
            self.eval_batch_size = self.train_batch_size
            self.log(f'eval_batch_size set {self.eval_batch_size}')
        if self.disc_start is None:
            if self.dataset == 'imagenet':
                self.disc_start = int(self.train_steps * 0.2)
            else:
                self.disc_start = int(self.train_steps * 0.1)
        if self.codebook_train_batch_size is None:
            self.codebook_train_batch_size = self.train_batch_size
        if self.generator_train_batch_size is None:
            self.generator_train_batch_size = self.train_batch_size
        if self.discriminator_train_batch_size is None:
            self.discriminator_train_batch_size = self.train_batch_size
        if self.valid_steps is None:
            self.valid_steps = int(self.train_steps * 0.02)
            self.log(f'valid_steps set to {self.valid_steps}')
        if self.save_steps is None:
            self.save_steps = int(self.train_steps * 0.04)
            self.log(f'save_steps set to {self.save_steps}')
        if self.cycle_steps is None:
            self.cycle_steps = int(self.train_steps * 0.1)
            self.log(f'cycle_steps set to {self.cycle_steps}')

        if self.codebook_beta1 is None:
            self.codebook_beta1 = self.beta1
        if self.codebook_beta2 is None:
            self.codebook_beta2 = self.beta2
        if self.resolution_factor is None:
            if self.dataset == 'cifar10':
                self.resolution_factor = 4
            elif self.dataset == 'celeba':
                self.resolution_factor = 8

        if self.image_size is None:
            if self.dataset in ['imagenet', 'ffhq']:
                self.image_size = 256
                self.image_width = 16
                self.image_height = 16
            elif self.dataset == 'cifar10':
                self.image_size = 32
                self.image_width = 8
                self.image_height = 8
            elif self.dataset == 'celeba':
                self.image_size = 128
                self.image_width = 16
                self.image_height = 16
            elif self.dataset == 'imagenet100':
                self.image_size = 128
                self.image_width = 16
                self.image_height = 16
            else:
                raise NotImplementedError(f'Unknown dataset {self.dataset}')

        warmup_steps = int(self.warmup_ratio * self.train_steps)
        if self.codebook_warmup_steps is None:
            self.codebook_warmup_steps = warmup_steps
        if self.generator_codebook_warmup_steps is None:
            self.generator_codebook_warmup_steps = warmup_steps
        if self.generator_warmup_steps is None:
            self.generator_warmup_steps = warmup_steps
        if self.discriminator_warmup_steps is None:
            self.discriminator_warmup_steps = warmup_steps

        if self.generator_type == 'vqvae' and self.rate_p is None:
            self.log('Perpetual loss will not be used')
            self.rate_p = 0
        else:
            self.rate_p = 1

        if self.cache_dir is None:
            self.cache_dir = f'./data/mqgan/datasets/{self.dataset}'
        if self.log_dir is None:
            self.log_dir = f"./data/mqgan/checkpoints/{self.dataset}"

        self.log(f'Image size: {self.image_size}x{self.image_size} '
                 f'| Latent size: {self.image_height}x{self.image_width}')


if __name__ == '__main__':
    ...
