"""Define MAP likelihood terms for deterministic RGC firing-rate observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from diffusion_retina import StaticRGCEncoder


class DeterministicRateLikelihood(nn.Module):
    """Evaluate a Gaussian-surrogate likelihood for deterministic RGC rates.

    Args:
        encoder (StaticRGCEncoder): Fixed differentiable static RGC encoder.
        rate_noise_std (float): Standard deviation controlling likelihood strength.
    """

    def __init__(self, encoder:StaticRGCEncoder, rate_noise_std:float=1.0):
        super().__init__()

        if rate_noise_std <= 0:
            raise ValueError("rate_noise_std must be positive.")
        
        self.encoder = encoder
        self.rate_noise_std = float(rate_noise_std)

    def loss(self, image: torch.Tensor, observed_rates: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute the deterministic RGC rate-matching loss.

        Args:
            image (torch.Tensor): Grayscale images shaped (batch, 1, height, width).
            observed_rates (Sequence[torch.Tensor]): Observed rates for each mosaic.

        Returns:
            torch.Tensor: Scalar negative log likelihood surrogate.
        """

        predicted_rates = self.encoder(image)
        if len(predicted_rates) != len(observed_rates):
            raise ValueError("Observed rates must contain one tensor per RGC mosaic.")
        
        losses = []
        for predicted, observed in zip(predicted_rates, observed_rates):
            observed = observed.to(device=image.device, dtype=image.dtype)
            if observed.shape != predicted.shape:
                raise ValueError(
                    f"Rate shape mismatch: expected {tuple(predicted.shape)}, got {tuple(observed.shape)}."
                )
            losses.append((predicted - observed).square().mean())
        return torch.stack(losses).mean() / (2 * self.rate_noise_std**2)

    def gradient( self, image:torch.Tensor, observed_rates:Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiate the likelihood with respect to an input image.

        Args:
            image (torch.Tensor): Grayscale images shaped (batch, 1, height, width).
            observed_rates (Sequence[torch.Tensor]): Observed rates for each mosaic.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image gradient and detached scalar loss.
        """

        image = image.requires_grad_(True)
        loss = self.loss(image, observed_rates)
        (gradient,) = torch.autograd.grad(loss, image)

        return gradient, loss.detach()
