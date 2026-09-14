"""Active-GFR-Net: GFR-Net with MC-dropout uncertainty for active learning.

Uncertainty is estimated by running multiple stochastic forward passes with a
higher dropout rate than training. Scores drive which frequency points or
geometries are queried next in the active-learning loop.
"""

import numpy as np
import torch
import torch.nn as nn

from .GFR_Net import GFR_Net, process_batch

__all__ = ["ActiveGFRNet", "MODEL_CLASS", "process_batch"]


class ActiveGFRNet(GFR_Net):
    """GFR-Net extended with Monte Carlo dropout uncertainty estimation."""

    def __init__(self, config, n_geom: int = 6):
        super().__init__(config, n_geom)
        self.name = "Active-GFR-Net"

        model_cfg = config["model"]
        self.train_dropout = model_cfg.get("dropout", 0.05)
        self.mc_dropout_rate = model_cfg.get("mc_dropout", 0.20)
        self.uncertainty_batch_size = model_cfg.get("uncertainty_batch_size", 512)

    def _set_dropout_rate(self, rate: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def mc_predict(
        self,
        x: torch.Tensor,
        axis_norm: torch.Tensor,
        n_mc: int = 20,
        batch_size: int = None,
    ):
        """Run Monte Carlo dropout forward passes and return mean and std predictions."""
        if batch_size is None:
            batch_size = self.uncertainty_batch_size

        was_training = self.training
        self.train()

        mean_preds = []
        std_preds = []

        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                x_batch = x[start:end]
                mc_preds = []

                for _ in range(n_mc):
                    self._set_dropout_rate(self.mc_dropout_rate)
                    mc_preds.append(super().forward(x_batch, axis_norm))
                    self._set_dropout_rate(self.train_dropout)

                mc_stack = torch.stack(mc_preds, dim=0)
                mean_preds.append(mc_stack.mean(dim=0))
                std_preds.append(mc_stack.std(dim=0, unbiased=False))

        self.train(was_training)
        return torch.cat(mean_preds, dim=0), torch.cat(std_preds, dim=0)

    def uncertainty_score_per_point(
        self,
        x: torch.Tensor,
        axis_norm: torch.Tensor,
        n_mc: int = 20,
        batch_size: int = None,
        config: dict = None,
    ) -> np.ndarray:
        """Return per-geometry, per-axis uncertainty scores with shape (n_samples, n_axis)."""
        if batch_size is None:
            batch_size = self.uncertainty_batch_size

        mean_pred, std = self.mc_predict(x, axis_norm, n_mc, batch_size)
        n_out = std.shape[-1]
        complex_format = (
            config.get("common", {}).get("complex_format") if config else None
        )

        if n_out == 3 and complex_format == "polar":
            gain_std = std[..., 0]
            phase_std = (std[..., 1] + std[..., 2]) / 2.0
            amp_proxy = torch.sigmoid(mean_pred[..., 0])
            score = gain_std + amp_proxy * phase_std
        else:
            score = std.mean(dim=-1)

        return score.cpu().numpy()

    def uncertainty_score(
        self,
        x: torch.Tensor,
        axis_norm: torch.Tensor,
        n_mc: int = 20,
        batch_size: int = None,
    ) -> np.ndarray:
        """Return one uncertainty score per geometry by averaging across axis points."""
        per_point = self.uncertainty_score_per_point(
            x,
            axis_norm,
            n_mc=n_mc,
            batch_size=batch_size,
            config=None,
        )
        return per_point.mean(axis=1)


MODEL_CLASS = ActiveGFRNet