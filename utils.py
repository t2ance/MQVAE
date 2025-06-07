import datetime
import math
import time

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler

from arguments import TrainingArguments
from models.discriminator import NLayerDiscriminator, weights_init
from models.lpips import LPIPS
from models.models_vq import VQModel, Quantizer


def wrap_with_loader(dataset, args: TrainingArguments, batch_size):
    print(f'wrapping dataset with dataloader | batch_size={batch_size}')
    if args.streaming or not args.is_distributed():
        sampler = None
    else:
        sampler = DistributedSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=args.drop_last,
        sampler=sampler
    )


def load_models(args: TrainingArguments):
    def wrap(model):
        if args.strategy == 'default':
            return model.to(args.device)
        elif args.strategy == 'distributed':
            from torch.nn.parallel import DistributedDataParallel
            return DistributedDataParallel(
                module=model.to(args.device),
                gradient_as_bucket_view=True,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

    args.perceptual_loss = LPIPS().to(args.device).eval()

    discriminator_module = NLayerDiscriminator().apply(weights_init)
    args.log(f'Architecture discriminator_module {discriminator_module}')
    args.log(f'#Params discriminator_module {count_params(discriminator_module)}')

    generator_module = VQModel(args=args).apply(weights_init)
    codebook_module = Quantizer(args=args).apply(weights_init)
    args.log(f'Architecture generator_module {generator_module}')
    args.log(f'#Params generator_module {count_params(generator_module)}')
    args.log(f'Architecture codebook_module {codebook_module}')
    args.log(f'#Params codebook_module {count_params(codebook_module)}')
    return wrap(generator_module), wrap(codebook_module), wrap(discriminator_module)


def count_params(model=None, parameters=None):
    if parameters is None:
        parameters = model.parameters()
    pp = 0
    for p in list(parameters):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_time_str():
    return time.strftime(
        "%Y-%m-%d %H:%M:%S", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).timetuple()
    )


def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


def generator_loss(to_return, x, global_step, discriminator, generator, args: TrainingArguments, rate_q=None):
    rate_p, rate_d, disc_start = args.rate_p, args.rate_d, args.disc_start
    if rate_q is None:
        rate_q = args.rate_q
    x_rec, qloss = to_return['x_rec'], to_return['qloss']
    rec_loss = torch.mean(torch.abs(x.contiguous() - x_rec.contiguous()))
    if args.rate_p == 0:
        ploss = 0
    else:
        ploss = torch.mean(args.perceptual_loss(x.contiguous(), x_rec.contiguous()))
    to_return_losses = {
        'qloss': qloss,
        'rec_loss': rec_loss,
        'ploss': ploss,
    }

    if global_step > disc_start and discriminator is not None:
        if args.is_distributed():
            last_layer = generator.module.get_last_layer()
        else:
            last_layer = generator.get_last_layer()
        gloss = -torch.mean(discriminator(x_rec))
        d_weight = calculate_adaptive_weight(rec_loss + rate_p * ploss, gloss, last_layer)
        loss = rec_loss + rate_p * ploss + d_weight * rate_d * gloss + rate_q * qloss

        to_return_losses.update({
            'loss': loss,
            'gloss': gloss,
            'd_weight': d_weight
        })
    else:
        loss = rec_loss + rate_q * qloss + rate_p * ploss
        to_return_losses.update({
            'loss': loss
        })
    return to_return_losses


def get_lr_scheduler(
        args: TrainingArguments, optimizer, lr, scheduler_type, warmup_steps, scale=1, train_steps=None, min_lr=None,
        start_step=None
):
    if train_steps is None:
        train_steps = args.train_steps

    def delayed_lambda(steps, lr_lambda_):
        def func(step):
            if step < steps:
                return 0.0
            else:
                return lr_lambda_(step - steps)

        return func

    def half_cosine(step):
        step = step * scale
        if step < warmup_steps:
            return step / warmup_steps
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (train_steps - warmup_steps)))
            return (1 - min_lr / lr) * cosine_decay + min_lr / lr

    def cosine(step):
        step = step * scale

        if step < warmup_steps:
            return step / warmup_steps
        else:
            step_after_warmup = step - warmup_steps
            current_cycle_step = step_after_warmup % args.cycle_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_cycle_step / args.cycle_steps))

            return (1 - min_lr / lr) * cosine_decay + min_lr / lr

    def constant(step):
        return 1

    if scheduler_type == 'half-cosine':
        lr_lambda = half_cosine
    elif scheduler_type == 'cosine':
        lr_lambda = cosine
    elif scheduler_type == 'constant':
        lr_lambda = constant
    else:
        raise NotImplementedError('Unknown scheduler type', scheduler_type)

    if start_step is not None:
        args.log(f'Lr scheduler is delayed by {start_step} steps')
        lr_lambda = delayed_lambda(start_step, lr_lambda)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


if __name__ == '__main__':
    ...
