"""Discretization transform for autoregressive modeling of continuous data.

This module provides functionality to convert continuous coordinate data
into discrete bins, similar to how PixelCNN treats pixel intensities.
"""

import torch
import torch.nn as nn


class DiscretizeTransform(nn.Module):
    """Transform that discretizes continuous values into bins.

    Similar to how PixelCNN discretizes pixel values (0-255), this transform
    converts continuous molecular coordinates into discrete bins for
    autoregressive modeling.

    Args:
        num_bins: Number of discrete bins to use (default: 256)
        min_val: Minimum value for discretization range (default: -4.0)
        max_val: Maximum value for discretization range (default: 4.0)

    The values are first clipped to [min_val, max_val], then linearly mapped
    to bins [0, num_bins-1].
    """

    def __init__(
        self,
        num_bins: int = 256,
        min_val: float = -4.0,
        max_val: float = 4.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_width = (max_val - min_val) / num_bins

        # Storage for empirical within-bin offsets (data-driven sampling)
        # offsets_per_bin[b] contains list of fractional offsets u in [0, 1] for bin b
        self._empirical_offsets: list[torch.Tensor] | None = None
        self._empirical_offsets_initialized = False

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to discrete bin indices.

        Args:
            x: Continuous values, shape (batch, num_atoms, dim) or (batch, seq_len)

        Returns:
            Discrete bin indices, same shape as input, dtype long
        """
        # Clip values to range
        x_clipped = torch.clamp(x, self.min_val, self.max_val)

        # Linear map to [0, num_bins-1]
        x_normalized = (x_clipped - self.min_val) / (self.max_val - self.min_val)
        x_discrete = (x_normalized * (self.num_bins - 1)).round().long()

        return x_discrete

    def undiscretize(self, x_discrete: torch.Tensor) -> torch.Tensor:
        """Convert discrete bin indices back to continuous values.

        Args:
            x_discrete: Discrete bin indices, shape (batch, num_atoms, dim) or (batch, seq_len)

        Returns:
            Continuous values (bin centers), same shape as input, dtype float
        """
        # Map back to continuous space (use bin centers)
        x_normalized = x_discrete.float() / (self.num_bins - 1)
        x_continuous = x_normalized * (self.max_val - self.min_val) + self.min_val

        return x_continuous

    def undiscretize_with_noise(self, x_discrete: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """Convert discrete bins to continuous with uniform noise within bin.

        This helps avoid artifacts from using only bin centers during generation.

        Args:
            x_discrete: Discrete bin indices
            noise_scale: Scale of uniform noise (1.0 = full bin width)

        Returns:
            Continuous values with uniform noise within each bin
        """
        # Get bin centers
        x_continuous = self.undiscretize(x_discrete)

        # Add uniform noise within bin
        noise = (torch.rand_like(x_continuous) - 0.5) * self.bin_width * noise_scale

        return x_continuous + noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass discretizes the input."""
        return self.discretize(x)

    def inverse(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Inverse pass undiscretizes the input."""
        if add_noise:
            return self.undiscretize_with_noise(x)
        return self.undiscretize(x)

    def compute_empirical_offsets(self, x: torch.Tensor, max_samples_per_bin: int = 10000) -> None:
        """Compute empirical within-bin offsets from training data.

        For each bin, stores the fractional offsets u = (x - bin_left) / bin_width
        observed in the training data. These can later be sampled from instead of
        using uniform noise.

        Args:
            x: Training data continuous values, shape (N,) or (N, D) flattened to 1D
            max_samples_per_bin: Maximum number of offsets to store per bin (for memory)
        """
        x_flat = x.reshape(-1).cpu()

        # Clip to valid range
        x_clipped = torch.clamp(x_flat, self.min_val, self.max_val - 1e-6)

        # Compute bin indices
        bin_indices = ((x_clipped - self.min_val) / self.bin_width).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Compute fractional offsets within each bin: u = (x - bin_left) / bin_width
        bin_lefts = self.min_val + bin_indices.float() * self.bin_width
        offsets = (x_clipped - bin_lefts) / self.bin_width
        offsets = torch.clamp(offsets, 0.0, 1.0)  # Ensure in [0, 1]

        # Group offsets by bin
        self._empirical_offsets = []
        for b in range(self.num_bins):
            mask = bin_indices == b
            bin_offsets = offsets[mask]

            if len(bin_offsets) > max_samples_per_bin:
                # Subsample to limit memory
                perm = torch.randperm(len(bin_offsets))[:max_samples_per_bin]
                bin_offsets = bin_offsets[perm]

            if len(bin_offsets) == 0:
                # No training samples in this bin - fall back to uniform [0, 1]
                bin_offsets = torch.tensor([0.5])

            self._empirical_offsets.append(bin_offsets)

        self._empirical_offsets_initialized = True

    def undiscretize_with_empirical_offsets(self, x_discrete: torch.Tensor) -> torch.Tensor:
        """Convert discrete bins to continuous using empirical within-bin offsets.

        Instead of uniform noise, samples offsets from the empirical distribution
        observed in training data. This can produce more realistic continuous
        values that better match the training distribution.

        Args:
            x_discrete: Discrete bin indices

        Returns:
            Continuous values sampled using empirical offsets
        """
        if not self._empirical_offsets_initialized:
            raise RuntimeError(
                "Empirical offsets not initialized. Call compute_empirical_offsets() "
                "with training data first, or use undiscretize_with_noise() instead."
            )

        device = x_discrete.device
        shape = x_discrete.shape
        x_flat = x_discrete.reshape(-1)

        # Get bin left edges
        bin_lefts = self.min_val + x_flat.float() * self.bin_width

        # Sample offsets from empirical distribution for each bin
        sampled_offsets = torch.zeros_like(bin_lefts)
        for b in range(self.num_bins):
            mask = x_flat == b
            if mask.sum() > 0:
                # Sample with replacement from empirical offsets for this bin
                # Match dtype and device of the output tensor
                bin_offsets = self._empirical_offsets[b].to(device=device, dtype=sampled_offsets.dtype)
                num_samples = mask.sum().item()
                indices = torch.randint(0, len(bin_offsets), (num_samples,), device=device)
                sampled_offsets[mask] = bin_offsets[indices]

        # Reconstruct continuous values: x = bin_left + u * bin_width
        x_continuous = bin_lefts + sampled_offsets * self.bin_width

        return x_continuous.reshape(shape)

    def has_empirical_offsets(self) -> bool:
        """Check if empirical offsets have been computed."""
        return self._empirical_offsets_initialized


class MuLawDiscretizeTransform(nn.Module):
    """Mu-law discretization for better resolution near zero.

    Mu-law companding provides higher resolution for small values,
    which may be useful for molecular coordinates where precision
    near equilibrium positions matters more.

    Args:
        num_bins: Number of discrete bins
        mu: Mu-law parameter (higher = more compression)
        min_val: Minimum value for range
        max_val: Maximum value for range
    """

    def __init__(
        self,
        num_bins: int = 256,
        mu: float = 255.0,
        min_val: float = -4.0,
        max_val: float = 4.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mu = mu
        self.min_val = min_val
        self.max_val = max_val

    def _mu_law_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mu-law compression."""
        # Normalize to [-1, 1]
        x_norm = 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1
        x_norm = torch.clamp(x_norm, -1, 1)

        # Apply mu-law
        sign = torch.sign(x_norm)
        x_mu = sign * torch.log1p(self.mu * torch.abs(x_norm)) / torch.log1p(torch.tensor(self.mu))

        return x_mu

    def _mu_law_decode(self, x_mu: torch.Tensor) -> torch.Tensor:
        """Apply mu-law expansion."""
        sign = torch.sign(x_mu)
        x_norm = sign * (torch.exp(torch.abs(x_mu) * torch.log1p(torch.tensor(self.mu))) - 1) / self.mu

        # Map back to original range
        x = (x_norm + 1) / 2 * (self.max_val - self.min_val) + self.min_val

        return x

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to discrete bin indices using mu-law."""
        x_mu = self._mu_law_encode(x)

        # Map from [-1, 1] to [0, num_bins-1]
        x_discrete = ((x_mu + 1) / 2 * (self.num_bins - 1)).round().long()

        return x_discrete

    def undiscretize(self, x_discrete: torch.Tensor) -> torch.Tensor:
        """Convert discrete bin indices back to continuous values."""
        # Map from [0, num_bins-1] to [-1, 1]
        x_mu = x_discrete.float() / (self.num_bins - 1) * 2 - 1

        # Apply inverse mu-law
        x_continuous = self._mu_law_decode(x_mu)

        return x_continuous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discretize(x)

    def inverse(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        return self.undiscretize(x)
