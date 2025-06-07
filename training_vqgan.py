import datetime
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing
import torch.nn.functional as F
import wandb
from betty.configs import Config, EngineConfig
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.utils import log_from_loss_dict, convert_tensor
from huggingface_hub import login

from arguments import TrainingArguments
from dataset import load_dataset_for_training
from utils import get_time_str, wrap_with_loader, load_models, generator_loss, get_lr_scheduler

torch.multiprocessing.set_sharing_strategy('file_system')


class DistributedProblem(ImplicitProblem, ABC):
    def __init__(self, args: TrainingArguments, engine=None, dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.engine = engine
        self.dataset = dataset

    def configure_train_data_loader(self):
        return wrap_with_loader(self.dataset, self.args, self.args.train_batch_size)

    def patch_module(self):
        """
        Patch module given the systems configuration (e.g., DDP, FSDP)
        """
        self.module.to(self.device)
        self.args.log(f'Distributed problem moved module to device {self.device}', all_=True)
        if self._strategy in ["distributed", "zero"]:
            self.args.log(f'Setting distributed module ({self._strategy}) for {self.name}')
            self.synchronize_params(self.parameters())
        elif self._strategy == "fsdp":
            if self.is_rank_zero():
                self.logger.warning("FSDP requires PyTorch version >= 1.12")
        elif self._strategy == "accelerate":
            self.module = self.accelerator.prepare(self.module)

        self.args.log('End of patching module')
        torch.cuda.empty_cache()

    def patch_scheduler(self):
        """
        Patch scheduler given the systems configuration (e.g., DDP, FSDP)
        """
        try:
            import inspect
            kwargs = {}
            sig = inspect.signature(self.scheduler.__class__.__init__)
            for param in sig.parameters:
                key = param
                if key == "self":
                    continue
                elif key == "optimizer":
                    kwargs[key] = self.optimizer
                elif key == "last_epoch":
                    kwargs[key] = getattr(self.scheduler, key) - 1
                elif key == "lr_lambda":
                    kwargs[key] = getattr(self.scheduler, "lr_lambdas")
                else:
                    value = getattr(self.scheduler, key)
                    kwargs[key] = value
            new_scheduler = self.scheduler.__class__(**kwargs)
            self.scheduler = new_scheduler
        except Exception as e:
            self.args.log(f'Failed to path scheduler: {e}')
        if self._strategy == "accelerate":
            self.scheduler = self.accelerator.prepare(self.scheduler)

    def patch_data_loader(self, loader):
        """
        Patch data loader given the systems configuration (e.g., DDP, FSDP)
        """
        if self._strategy in ["distributed", "zero", "fsdp"]:
            patched_loader = loader
        elif self._strategy == "accelerate":
            patched_loader = self.accelerator.prepare(loader)
        else:
            patched_loader = loader

        return patched_loader

    def get_batch_single_loader(self, idx):
        """
        Load training batch from one of the user-provided data loader(s)

        :return: New training batch
        :rtype: Any
        """
        data_iterator = self.train_data_iterator[idx]
        try:
            batch = next(data_iterator)
        except StopIteration:
            if idx == 0:
                self.epoch_callback_exec()
            self.epoch_counter[idx] += 1
            train_data_loader = self.train_data_loader[idx]
            self.train_data_iterator[idx] = iter(train_data_loader)
            batch = next(self.train_data_iterator[idx])
        if not isinstance(batch, dict):
            batch = tuple(convert_tensor(value, self.device) for value in batch)
        else:
            for key, value in batch.items():
                batch[key] = convert_tensor(value, self.device)

        return batch


class GeneratorProblem(DistributedProblem):

    def training_step(self, batch):
        # self.args.log(f'Generator training step @ {self.engine.global_step}!')
        x, y = batch['image'], batch['label']
        x, y = x.to(self.args.device), y.to(self.args.device)
        to_return = self.forward(x, self.codebook.module)

        loss = generator_loss(
            to_return, x, self.engine.global_step,
            self.discriminator.module if hasattr(self, 'discriminator') else None,
            self.module, self.args, rate_q=self.args.generator_rate_q
        )
        if self.args.is_main_process() and self.engine.global_step % self.args.log_steps == 0:
            log_dict = {
                'generator/lr': self.optimizer.param_groups[0]['lr'],
                **{f'generator/{k}': v.item() for k, v in loss.items()},
            }
            wandb.log(log_dict, step=self.engine.global_step)

        return loss['loss']


class DiscriminatorProblem(DistributedProblem):

    def training_step(self, batch):
        # print(f'Discriminator training step @ {self.engine.global_step}!')
        x, y = batch['image'], batch['label']
        x, y = x.to(self.args.device), y.to(self.args.device)

        if hasattr(self, 'generator'):
            to_return = self.generator(x, self.codebook.module)
        else:
            to_return = self.generator_codebook(x)
        x_rec, qloss = to_return['x_rec'], to_return['qloss']

        logits_real = self.forward(x.contiguous().detach().clone())
        logits_fake = self.forward(x_rec.detach().clone())

        loss = 0.5 * (torch.mean(F.relu(1. - logits_real)) + torch.mean(F.relu(1. + logits_fake)))

        if self.args.is_main_process() and self.engine.global_step % self.args.log_steps == 0:
            log_dict = {
                'discriminator/lr': self.optimizer.param_groups[0]['lr'],
                'discriminator/loss': loss.item()
            }
            wandb.log(log_dict, step=self.engine.global_step)

        return loss


class CodebookProblem(DistributedProblem):
    def training_step(self, batch):
        x, y = batch['image'], batch['label']
        x, y = x.to(self.args.device), y.to(self.args.device)
        to_return = self.generator(x, self.module)
        loss = generator_loss(
            to_return, x, self.engine.global_step,
            self.discriminator.module if hasattr(self, 'discriminator') else None,
            self.generator.module, self.args, rate_q=self.args.codebook_rate_q
        )

        if self.args.is_main_process() and self.engine.global_step % self.args.log_steps == 0:
            wandb.log({
                'codebook/lr': self.optimizer.param_groups[0]['lr'],
                'codebook/perplexity': to_return['perplexity'],
                'codebook/perplexity_loss': to_return['perplexity_loss'],
                **{f'codebook/{k}': v.item() for k, v in loss.items()},
            }, step=self.engine.global_step)

        return loss['loss']


class MQEngine(Engine):
    def train_step(self):
        """
        Running one-step gradient descent for all leaf problems.
        """
        for leaf in self.leaves:
            if isinstance(leaf, DiscriminatorProblem) and self.global_step < self.args.disc_start:
                continue

            leaf.step(global_step=self.global_step)

    def __init__(self, args: TrainingArguments, **kwargs):
        self.args = args
        super().__init__(**kwargs)

    def run(self):
        """
        Execute multilevel optimization by running gradient descent for leaf problems.
        """
        self.train()
        torch.cuda.empty_cache()
        for it in range(1 + self.global_step, 1 + self.train_iters):
            self.global_step += 1
            if self.distributed():
                dist.barrier()
            self.train_step()

            if it % self.valid_step == 0:
                if self.distributed():
                    dist.barrier()
                torch.cuda.empty_cache()
                self.eval()
                validation_stats = self.validation() or {}
                log_loss = log_from_loss_dict(validation_stats)
                self.logger.info(
                    f"[Validation (rank {self._local_rank})] [Global Step {self.global_step}] " f"{log_loss}"
                )
                self.logger.log(
                    validation_stats, tag="validation", step=self.global_step
                )
                self.train()

                if self.early_stopping is not None:
                    stop = self.early_stopping(validation_stats)
                    if stop:
                        self.logger.info("Early stopping is executed!")
                        break
                if self.distributed():
                    dist.barrier()

        self.cleanup()

    def distributed(self):
        return self._world_size > 1

    def configure_systems(self):
        """
        Configure basic systems set-up like distributed training and device placement.
        """
        # configure distributed training
        if self._strategy in ["distributed", "zero", "fsdp"]:
            self.args.log(f'Configuring distributed system')
            self._world_size = dist.get_world_size()
            assert self._world_size > 1
            self._rank = dist.get_rank()

            device_count = torch.cuda.device_count()
            self._local_rank = self._rank % device_count

        self.args.log(f'Configuring system, with rank = {self._rank} and local rank = {self._local_rank}')

        dist_dict = {}
        dist_dict["strategy"] = self._strategy
        dist_dict["backend"] = self._backend
        dist_dict["world_size"] = self._world_size
        dist_dict["rank"] = self._rank
        dist_dict["local_rank"] = self._local_rank

        # args.rank = dist_dict["rank"]

        # configure device for the current rank
        if self._strategy in ["distributed", "zero", "fsdp"]:
            self.device = torch.device("cuda", self._local_rank)
        elif self._strategy == "accelerate":
            self.device = self.accelerator.device
        elif self._strategy == "cpu":
            self.device = "cpu"
        elif self._strategy == "gpu":
            self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dist_dict = dist_dict

        return dist_dict

    @abstractmethod
    def validation(self):
        raise NotImplementedError()


def setup():
    from transformers import HfArgumentParser
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--json_file", type=str, default=None, help="Path to JSON file for arguments")
    commandline_args, remaining_args = parser.parse_known_args()

    if commandline_args.json_file is not None:
        hf_parser = HfArgumentParser(TrainingArguments)
        args: TrainingArguments = hf_parser.parse_json_file(json_file=commandline_args.json_file)[0]
    else:
        hf_parser = HfArgumentParser(TrainingArguments)
        args: TrainingArguments = hf_parser.parse_args_into_dataclasses(args=remaining_args)[0]

    cudnn.benchmark = True

    args.world_size = torch.cuda.device_count()
    if args.world_size > 1:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=args.timedelta),
                                world_size=args.world_size)
        args.local_rank = dist.get_rank()
        args.strategy = 'distributed'
        args.log(f'Using multi-gpu ({args.world_size} gpus in total)')
        args.log(f'This is gpu {args.local_rank}', all_=True)
    else:
        args.strategy = 'default'
        args.local_rank = 0
        args.log(f'Using single-gpu')

    args.set_num_workers()
    args.seed += args.local_rank
    args.cache_dir_back = str(args.cache_dir)
    args.cache_dir = args.cache_dir + f'/{args.local_rank}'
    login(token=args.hf_token)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        args.device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
    else:
        args.device = torch.device("cpu")

    if args.is_main_process():
        wandb.login(key=args.wandb_token)
        name = f"{args.dataset} {args.title} {get_time_str()}"
        args.run = wandb.init(project="vqgan", name=name, config=args.__dict__)
        wandb.run.name = f'{name} {args.run.id}'
        args.log(f'Run id = {args.run.id}')
        wandb.run.save()
        if args.log_dir is not None:
            args.log_dir = os.path.join(args.log_dir, str(args.run.id))
            args.log(f'Logging to path {args.log_dir}')
        else:
            args.log(f"Don't log in this run")
    args.log(args)
    return args


def main():
    args = setup()

    loaded_datasets = load_dataset_for_training(args)
    generator_module, codebook_module, discriminator_module = load_models(args)
    inner_config = Config(
        type=args.hyper_gradient_type, precision=args.precision, gradient_accumulation=args.gradient_accumulation,
        gradient_clipping=args.gradient_clipping
    )
    outer_config = Config(
        type=args.hyper_gradient_type, precision=args.precision, retain_graph=True,
        gradient_accumulation=args.gradient_accumulation, gradient_clipping=args.gradient_clipping
    )
    engine_config = EngineConfig(train_iters=args.train_steps, valid_step=args.valid_steps, strategy=args.strategy)
    optimizer_generator = torch.optim.AdamW(
        generator_module.parameters(), weight_decay=args.generator_weight_decay, lr=args.generator_lr,
        betas=(args.beta1, args.beta2), eps=args.eps
    )
    scheduler_generator = get_lr_scheduler(
        args=args, optimizer=optimizer_generator, lr=args.generator_lr, scheduler_type=args.generator_scheduler_type,
        warmup_steps=args.generator_warmup_steps, min_lr=args.generator_min_lr,
        train_steps=args.train_steps
    )
    generator_problem = GeneratorProblem(
        name="generator", module=generator_module, optimizer=optimizer_generator, config=inner_config,
        dataset=(loaded_datasets['dataset_train_inner']), scheduler=scheduler_generator, args=args,
    )
    # optimizer_codebook = torch.optim.AdamW(
    #     codebook_module.parameters(), weight_decay=args.codebook_weight_decay, lr=args.codebook_lr,
    #     betas=(args.beta1, args.beta2), eps=args.eps
    # )
    optimizer_codebook = torch.optim.SGD(
        codebook_module.parameters(), weight_decay=args.codebook_weight_decay, lr=args.codebook_lr,
        momentum=args.codebook_momentum
    )
    scheduler_codebook = get_lr_scheduler(
        args=args, optimizer=optimizer_codebook, lr=args.codebook_lr, scheduler_type=args.codebook_scheduler_type,
        scale=args.gradient_accumulation if args.optimization_type == 'meta' else 1,
        warmup_steps=args.codebook_warmup_steps, min_lr=args.codebook_min_lr
    )
    codebook_problem = CodebookProblem(
        name="codebook", module=codebook_module, optimizer=optimizer_codebook, config=outer_config,
        dataset=(loaded_datasets['dataset_train_outer']), scheduler=scheduler_codebook, args=args
    )
    problems = [generator_problem, codebook_problem]
    if args.generator_type == 'vqgan':
        optimizer_discriminator = torch.optim.AdamW(
            discriminator_module.parameters(), weight_decay=args.discriminator_weight_decay, lr=args.discriminator_lr,
            betas=(args.beta1, args.beta2), eps=args.eps
        )
        scheduler_discriminator = get_lr_scheduler(
            args=args, optimizer=optimizer_discriminator, lr=args.discriminator_lr,
            scheduler_type=args.discriminator_scheduler_type, train_steps=args.train_steps - args.disc_start,
            warmup_steps=args.discriminator_warmup_steps, min_lr=args.discriminator_min_lr
        )
        discriminator_problem = DiscriminatorProblem(
            name="discriminator", module=discriminator_module, optimizer=optimizer_discriminator, config=inner_config,
            dataset=loaded_datasets['dataset_train_inner'], scheduler=scheduler_discriminator, args=args
        )
        problems.append(discriminator_problem)
    if args.optimization_type == 'alt':
        l2u = {}
        u2l = {}
    elif args.optimization_type == 'meta':
        l2u = {generator_problem: [codebook_problem]}
        u2l = {codebook_problem: [generator_problem]}
    else:
        raise NotImplementedError('Unknown optimization type', args.optimization_type)
    dependencies = {"l2u": l2u, "u2l": u2l}
    engine = MQEngine(args=args, config=engine_config, problems=problems, dependencies=dependencies)
    generator_problem.engine = engine
    if args.generator_type == 'vqgan':
        discriminator_problem.engine = engine
    codebook_problem.engine = engine
    engine.run()


if __name__ == "__main__":
    main()
