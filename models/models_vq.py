import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from torch import einsum
from torch.nn import Embedding

from arguments import TrainingArguments
from models.vqgan_encoder_decoder import VQGANEncoder, VQGANDecoder
from models.vqvae_encoder_decoder import VQVAEEncoder, VQVAEDecoder


class Quantizer(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, args: TrainingArguments):
        super().__init__()
        assert args.quantizer_type == 'hyper-net'
        self.alpha, self.beta, self.gamma = args.alpha, args.beta, args.gamma

        args.log('Learn the embedding during training')
        if args.hyper_net_type == 'basis':
            self.embedding = Embedding(args.embed_dim, args.embed_dim)
        else:
            self.embedding = Embedding(args.n_vision_words, args.hyper_net_hidden_dim)

        self.args = args
        self.quantize_type = args.quantizer_type
        self.embed_dim = args.embed_dim

        if args.hyper_net_type == 'null':
            assert args.embed_dim == args.hyper_net_hidden_dim
            module_list = [
                torch.nn.Identity()
            ]
        elif args.hyper_net_type == 'linear':
            module_list = [
                torch.nn.Linear(args.hyper_net_hidden_dim, args.embed_dim)
            ]
        elif args.hyper_net_type == 'mlp':
            module_list = [
                torch.nn.Linear(args.hyper_net_hidden_dim, args.mlp_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(args.mlp_hidden_dim, args.embed_dim)
            ]
        elif args.hyper_net_type == 'mlp-tanh':
            module_list = [
                torch.nn.Linear(args.hyper_net_hidden_dim, args.mlp_hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(args.mlp_hidden_dim, args.embed_dim)
            ]
        elif args.hyper_net_type == 'basis':
            module_list = [
                torch.nn.Linear(args.embed_dim, args.n_vision_words)
            ]
        else:
            raise NotImplementedError(f'Unknown hyper-net type {self.args.hyper_net_type}')

        self.hyper_net = torch.nn.Sequential(
            *module_list
        )
        self.initialized = False

    def get_codebook(self):
        weight_tensor = self.embedding.weight
        weight_tensor = weight_tensor.to(dtype=next(self.hyper_net.parameters()).dtype)
        if self.args.hyper_net_type in 'basis':
            codebook = self.hyper_net(weight_tensor).transpose(0, 1)
        elif self.args.hyper_net_type == 'attention':
            codebook = self.hyper_net(weight_tensor.unsqueeze(0)).squeeze(0)
        else:
            codebook = self.hyper_net(weight_tensor)

        return codebook

    def distance(self, z, codebook):
        to_return = {}
        to_return['z'] = z
        to_return['codebook'] = codebook
        if self.args.l2_proj:
            distance = -einsum('n d, c d -> n c', z, codebook)
        else:
            distance = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(codebook ** 2, dim=1) \
                       - 2 * torch.einsum('bd,dn->bn', z, rearrange(codebook, 'n d -> d n'))
        to_return['distance'] = distance
        return to_return

    def forward(self, z):
        codebook = self.get_codebook()

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        if self.args.l2_proj:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            codebook = F.normalize(codebook, p=2, dim=-1)

        d = self.distance(z_flattened, codebook)['distance']

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, codebook).view(z.shape)

        loss = self.alpha * torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        e_mean_non_diff = F.one_hot(
            min_encoding_indices, num_classes=self.args.n_vision_words
        ).view(-1, self.args.n_vision_words).float().mean(0)

        perplexity = torch.exp(-torch.sum(e_mean_non_diff * torch.log(e_mean_non_diff + 1e-10)))

        if self.gamma != 0:
            e_mean_diff = F.softmax(-d / 0.1, dim=1).mean(dim=0)
            perplexity_loss = torch.exp(torch.sum(e_mean_diff * torch.log(e_mean_diff + 1e-10)))
            loss += self.gamma * perplexity_loss
        else:
            perplexity_loss = 0.
        return z_q, loss, (d, perplexity, min_encoding_indices, perplexity_loss)


class VQModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.args = args
        if args.generator_type == 'vqgan':
            config = OmegaConf.load(args.vq_config_path)
            args.log(yaml.dump(OmegaConf.to_container(config)))
            ddconfig = config.model.params['ddconfig']

            self.encoder = VQGANEncoder(**ddconfig)
            self.decoder = VQGANDecoder(**ddconfig)
            self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], args.embed_dim, 1)

            self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, ddconfig["z_channels"], 1)

        elif args.generator_type == 'vqvae':
            self.encoder = VQVAEEncoder(dim_z=args.z_channels, factor=args.resolution_factor)
            self.decoder = VQVAEDecoder(dim_z=args.z_channels, factor=args.resolution_factor)
            self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)

            self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)

        else:
            raise NotImplementedError()

        if args.trailing_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=args.embed_dim)

        self.quantizer = None

    def forward(self, input, quantizer: Quantizer = None):
        quantizer = self.quantizer if self.quantizer is not None else quantizer
        quant, qloss, [_, perplexity, tk_labels, perplexity_loss] = self.encode(input, quantizer)
        return {
            'x_rec': self.decode(quant),
            'quant': quant,
            'perplexity': perplexity,
            'perplexity_loss': perplexity_loss,
            'tk_labels': tk_labels,
            'qloss': qloss
        }

    def encode(self, input, quantizer: Quantizer):
        quantizer = self.quantizer if self.quantizer is not None else quantizer
        h = self.quant_conv(self.encoder(input))
        if self.args.trailing_norm:
            h = self.layer_norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        quant, emb_loss, info = quantizer(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def get_last_layer(self):
        return self.decoder.get_last_layer()
