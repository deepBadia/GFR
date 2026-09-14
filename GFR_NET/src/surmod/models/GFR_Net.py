"""Geometry-Frequency Response Network.

Maps geometry parameters and a normalized frequency axis to S-parameter outputs.
The network encodes geometry once, projects frequency with learnable Fourier
features, combines them, and decodes per-frequency predictions.
"""

import numpy as np
import torch
import torch.nn as nn

from surmod.utils.common import compute_loss


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GFR_Net(nn.Module):
    """Predicts frequency responses from geometry parameters and axis values."""

    def __init__(self, config, n_geom: int = 6):
        super().__init__()
        self.n_geom = n_geom
        self.name = "GFR-Net"

        model_cfg = config["model"]
        common_cfg = config["common"]

        self.n_out = common_cfg.get("n_out", 2)
        self.hidden_dim = model_cfg.get("hidden_dim", 512)
        self.dropout = model_cfg.get("dropout", 0.05)
        self.n_fourier = model_cfg.get("n_fourier", 64)
        self.n_geo_fourier = model_cfg.get("n_geo_fourier", 0)
        n_encoder_layers = model_cfg.get("n_encoder_layers", 0)
        self.use_film = model_cfg.get("use_film", False)
        self.n_geom_cont = model_cfg.get("n_geom_cont", n_geom)
        n_geom_disc = n_geom - self.n_geom_cont

        if self.n_geo_fourier > 0:
            self.geo_scales = nn.Parameter(
                torch.randn(self.n_geom_cont, self.n_geo_fourier) * 0.5
            )
            geo_input_dim = self.n_geom_cont + 2 * self.n_geom_cont * self.n_geo_fourier
        else:
            geo_input_dim = self.n_geom_cont if n_geom_disc > 0 else n_geom

        encoder_layers = [
            nn.Linear(geo_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        ]
        for _ in range(n_encoder_layers):
            encoder_layers.append(
                ResidualBlock(self.hidden_dim, expansion=2, dropout=self.dropout)
            )
        self.geom_encoder = nn.Sequential(*encoder_layers)

        if n_geom_disc > 0:
            self.disc_encoder = nn.Sequential(
                nn.Linear(n_geom_disc, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )

        self.freq_scales = nn.Parameter(torch.randn(self.n_fourier) * 0.5)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(self.hidden_dim, expansion=4, dropout=self.dropout)
                for _ in range(model_cfg.get("n_layers", 8))
            ]
        )

        freq_dim = 1 + 2 * self.n_fourier
        self.freq_proj = nn.Linear(freq_dim, self.hidden_dim)

        if self.use_film:
            self.film_gamma = nn.Sequential(
                nn.Linear(freq_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
            self.film_beta = nn.Linear(freq_dim, self.hidden_dim)

        self.use_freq_attention = model_cfg.get("use_freq_attention", False)
        n_attn_heads = model_cfg.get("n_attn_heads", 8)
        if self.use_freq_attention:
            self.freq_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=n_attn_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.freq_attn_norm = nn.LayerNorm(self.hidden_dim)

        n_out_layers = model_cfg.get("n_out_layers", 2)
        self.use_split_head = model_cfg.get("use_split_head", False)
        if self.use_split_head and self.n_out == 3:
            self.gain_head = nn.Sequential(
                *[
                    ResidualBlock(self.hidden_dim, expansion=2, dropout=self.dropout)
                    for _ in range(max(1, n_out_layers // 2))
                ],
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 1),
            )
            self.phase_head = nn.Sequential(
                *[
                    ResidualBlock(self.hidden_dim, expansion=2, dropout=self.dropout)
                    for _ in range(n_out_layers)
                ],
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 2),
            )
        else:
            self.out_blocks = nn.Sequential(
                *[
                    ResidualBlock(self.hidden_dim, expansion=2, dropout=self.dropout)
                    for _ in range(n_out_layers)
                ],
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.n_out),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _fourier_features(self, freq: torch.Tensor) -> torch.Tensor:
        scaled = freq.unsqueeze(-1) * self.freq_scales * 2 * np.pi
        return torch.cat(
            [freq.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)],
            dim=-1,
        )

    def _geo_fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.unsqueeze(-1) * self.geo_scales.unsqueeze(0) * 2 * np.pi
        return torch.cat(
            [
                x,
                torch.sin(scaled).flatten(1),
                torch.cos(scaled).flatten(1),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor, axis_norm: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_freq = len(axis_norm)

        if self.n_geom_cont < x.shape[-1]:
            x_cont = x[..., : self.n_geom_cont]
            x_disc = x[..., self.n_geom_cont :]
            x_geo = (
                self._geo_fourier_features(x_cont)
                if self.n_geo_fourier > 0
                else x_cont
            )
            h = self.geom_encoder(x_geo) + self.disc_encoder(x_disc)
        else:
            x_geo = self._geo_fourier_features(x) if self.n_geo_fourier > 0 else x
            h = self.geom_encoder(x_geo)

        for block in self.blocks:
            h = block(h)

        freq_raw = self._fourier_features(axis_norm)
        freq_feat = self.freq_proj(freq_raw)

        if self.use_film:
            gamma = self.film_gamma(freq_raw) + 0.5
            beta = self.film_beta(freq_raw)
            h = h.unsqueeze(1) * gamma.unsqueeze(0) + beta.unsqueeze(0)
            h = h + freq_feat.unsqueeze(0)
        else:
            h = h.unsqueeze(1).expand(batch_size, n_freq, -1)
            h = h + freq_feat.unsqueeze(0)

        if self.use_freq_attention:
            h_normed = self.freq_attn_norm(h)
            h_attn, _ = self.freq_attn(h_normed, h_normed, h_normed)
            h = h + h_attn

        h_flat = h.reshape(batch_size * n_freq, -1)
        if self.use_split_head and self.n_out == 3:
            gain = self.gain_head(h_flat)
            phase = self.phase_head(h_flat)
            out = torch.cat([gain, phase], dim=-1).reshape(batch_size, n_freq, self.n_out)
        else:
            out = self.out_blocks(h_flat).reshape(batch_size, n_freq, self.n_out)
        return out


MODEL_CLASS = GFR_Net


def process_batch(**kwargs):
    """Compute loss for one training or validation batch."""
    model = kwargs["model"]
    bx = kwargs["bx"]
    by = kwargs["by"]
    bw = kwargs["bw"]
    axis_norm = kwargs["axis_norm"]
    loss_accumulator = kwargs["train_sse"]
    weight_accumulator = kwargs["train_total_weight"]
    mask = kwargs.get("mask")
    config = kwargs["config"]

    pred = model(bx, axis_norm)
    loss, batch_loss_sum, batch_total_w = compute_loss(
        pred,
        by,
        weights=bw,
        mask=mask,
        config=config,
        training=model.training,
    )

    loss_accumulator += batch_loss_sum.item()
    weight_accumulator += batch_total_w.item()
    return loss, loss_accumulator, weight_accumulator, pred