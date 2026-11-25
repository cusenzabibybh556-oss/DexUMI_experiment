"""Single-camera ACT/CVAE using the pinned TronCamp DETR transformer."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .vendor.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


def position_encoding(length, dimension, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=dtype) * (-math.log(10000) / dimension))
    out = torch.zeros(length, dimension, device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(positions * frequency)
    out[:, 1::2] = torch.cos(positions * frequency)
    return out


def image_position(height, width, dimension, device, dtype):
    half = dimension // 2
    y = position_encoding(height, half, device, dtype)[:, None, :].expand(height, width, half)
    x = position_encoding(width, half, device, dtype)[None, :, :].expand(height, width, half)
    return torch.cat([y, x], dim=-1).permute(2, 0, 1).unsqueeze(0)


class ACT(nn.Module):
    def __init__(self, config, load_pretrained=None):
        super().__init__()
        self.config = config.validate()
        c, d = config, config.hidden_dim
        use_weights = c.pretrained_backbone if load_pretrained is None else load_pretrained
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if use_weights else None)
        # Use fixed BatchNorm statistics, as in ACT, while fine-tuning convolution weights.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_projection = nn.Conv2d(512, d, 1)
        self.state_projection = nn.Linear(c.state_dim, d)
        self.action_projection = nn.Linear(c.action_dim, d)
        self.encoder_state_projection = nn.Linear(c.state_dim, d)
        self.cls = nn.Embedding(1, d)
        encoder_layer = TransformerEncoderLayer(d, c.heads, c.feedforward_dim, c.dropout, "relu", False)
        self.latent_encoder = TransformerEncoder(encoder_layer, c.encoder_layers)
        self.latent_parameters = nn.Linear(d, c.latent_dim * 2)
        self.latent_projection = nn.Linear(c.latent_dim, d)
        self.transformer = Transformer(d_model=d, nhead=c.heads, num_encoder_layers=c.encoder_layers,
                                       num_decoder_layers=c.decoder_layers, dim_feedforward=c.feedforward_dim,
                                       dropout=c.dropout, return_intermediate_dec=False)
        self.queries = nn.Embedding(c.chunk_size, d)
        self.additional_positions = nn.Embedding(2, d)
        self.action_head = nn.Linear(d, c.action_dim)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, image, state, actions=None, is_pad=None):
        c = self.config
        if state.shape[-1] != c.state_dim or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image [B,3,H,W] and configured state dimension")
        batch = state.shape[0]
        mu = logvar = None
        if actions is not None:
            if actions.shape != (batch, c.chunk_size, c.action_dim) or is_pad is None:
                raise ValueError("Expected full action chunk and explicit padding mask")
            tokens = torch.cat([self.cls.weight[None].expand(batch, -1, -1),
                                self.encoder_state_projection(state)[:, None],
                                self.action_projection(actions)], dim=1).transpose(0, 1)
            pad = torch.cat([torch.zeros(batch, 2, device=state.device, dtype=torch.bool), is_pad], dim=1)
            pos = position_encoding(c.chunk_size + 2, c.hidden_dim, state.device, state.dtype)[:, None]
            cls_output = self.latent_encoder(tokens, pos=pos, src_key_padding_mask=pad)[0]
            mu, logvar = self.latent_parameters(cls_output).chunk(2, dim=-1)
            logvar = logvar.clamp(-20, 20)
            latent = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            latent = state.new_zeros(batch, c.latent_dim)
        feature = self.image_projection(self.backbone((image - self.rgb_mean) / self.rgb_std))
        pos = image_position(feature.shape[-2], feature.shape[-1], c.hidden_dim, feature.device, feature.dtype)
        hidden = self.transformer(feature, None, self.queries.weight, pos,
                                  self.latent_projection(latent), self.state_projection(state),
                                  self.additional_positions.weight)[0]
        return self.action_head(hidden), mu, logvar


def loss(prediction, target, is_pad, mu, logvar, kl_weight):
    valid = ~is_pad
    count = valid.sum() * target.shape[-1]
    if count == 0:
        raise ValueError("All-padding action chunks cannot train ACT")
    l1 = (F.l1_loss(prediction, target, reduction="none") * valid[..., None]).sum() / count
    kl = (-.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)).mean()
    return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl}
