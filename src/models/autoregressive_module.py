"""Autoregressive module for molecular coordinate generation.

This module implements an autoregressive model (similar to PixelCNN/GPT)
for generating molecular coordinates by discretizing coordinates into bins
and predicting each coordinate sequentially.
"""

import logging

import torch
import torch.nn.functional as F

from src.data.transforms.discretize import DiscretizeTransform
from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule

logger = logging.getLogger(__name__)


class AutoregressiveLitModule(TransferableBoltzmannGeneratorLitModule):
    """Autoregressive model for molecular coordinate generation.

    This model discretizes continuous 3D coordinates into bins and learns
    to predict each coordinate autoregressively, similar to how PixelCNN
    generates images pixel by pixel.

    Key differences from flow matching:
    - Uses discrete representation (cross-entropy loss instead of MSE)
    - Generates one coordinate at a time (no parallel generation)
    - Exact likelihood computation (no ODE integration needed)

    Args:
        num_bins: Number of discrete bins for coordinate values
        discretize_min: Minimum value for discretization range
        discretize_max: Maximum value for discretization range
        temperature: Sampling temperature for generation
        top_k: Top-k sampling parameter (None to disable)
        top_p: Nucleus sampling parameter (None to disable)
    """

    def __init__(
        self,
        num_bins: int = 256,
        discretize_min: float = -3.0,
        discretize_max: float = 3.0,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_mu_law: bool = False,
        label_smoothing: float = 0.0,
        use_empirical_offsets: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the autoregressive module.

        :param net: The autoregressive network (e.g., CausalTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_bins: Number of discrete bins for coordinates.
        :param discretize_min: Minimum value for discretization.
        :param discretize_max: Maximum value for discretization.
        :param temperature: Sampling temperature.
        :param top_k: Top-k sampling (None to disable).
        :param top_p: Nucleus sampling (None to disable).
        :param label_smoothing: Label smoothing factor (0.0 = no smoothing).
        :param use_empirical_offsets: If True, use data-driven within-bin offsets
            instead of uniform noise during inference. Offsets are computed from
            training data on first validation/test run.
        """
        super().__init__(*args, **kwargs)

        if "strict_loading" in kwargs:
            self.strict_loading = kwargs["strict_loading"]

        # Initialize discretization transform
        self.discretize = DiscretizeTransform(
            num_bins=num_bins,
            min_val=discretize_min,
            max_val=discretize_max,
        )

        self.save_hyperparameters(
            "num_bins",
            "discretize_min",
            "discretize_max",
            "temperature",
            "top_k",
            "top_p",
            "use_mu_law",
            "label_smoothing",
            "use_empirical_offsets",
            logger=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        encodings: dict | None = None,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: Discrete token indices, shape (batch, seq_len)
        :param encodings: Optional conditional embeddings
        :param mask: Optional node mask
        :param use_cache: Whether to use KV-cache (for generation)
        :return: Logits over bins, shape (batch, seq_len, num_bins)
        """
        logits, _ = self.net(x, encodings=encodings, node_mask=mask, use_cache=use_cache)
        return logits

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        Computes cross-entropy loss for next-token prediction.

        :param batch: Dictionary containing 'x' (coordinates), 'encodings', 'mask'
        :return: A tensor of losses.
        """
        # Get continuous coordinates: (batch, num_atoms, 3)
        x_continuous = batch["x"]

        num_samples = x_continuous.shape[0]
        num_dims = self.datamodule.hparams.num_dimensions

        encodings = batch.get("encodings", None)
        mask = batch.get("mask", None)

        # Flatten to (batch, num_atoms * 3)
        x_flat = x_continuous.reshape(num_samples, -1)

        # Discretize coordinates
        x_discrete = self.discretize.discretize(x_flat)  # (batch, seq_len)

        # Create input sequence (shifted right) for teacher forcing
        # Input: [START, x_0, x_1, ..., x_{n-2}]
        # Target: [x_0, x_1, ..., x_{n-1}]
        # We use 0 as the start token (could also use a dedicated start token)
        x_input = torch.cat(
            [torch.zeros(num_samples, 1, dtype=torch.long, device=x_discrete.device), x_discrete[:, :-1]], dim=1
        )

        # Get logits for all positions
        logits = self.forward(x_input, encodings=encodings, mask=mask)

        # Compute cross-entropy loss with optional label smoothing
        # logits: (batch, seq_len, num_bins)
        # targets: (batch, seq_len)
        loss = F.cross_entropy(
            logits.reshape(-1, self.hparams.num_bins),
            x_discrete.reshape(-1),
            reduction="none",
            label_smoothing=self.hparams.label_smoothing,
        ).reshape(num_samples, -1)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to coordinate level
            mask_expanded = mask.repeat_interleave(num_dims, dim=1)
            loss = loss * mask_expanded.float()
            # Average over non-masked positions
            loss = loss.sum(dim=-1) / mask_expanded.float().sum(dim=-1).clamp(min=1)
        else:
            loss = loss.mean(dim=-1)

        loss = loss.mean()

        return loss

    def compute_log_likelihood(
        self,
        x_continuous: torch.Tensor,
        encodings: dict | None = None,
        mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Compute exact log likelihood of samples.

        Unlike flow matching which requires ODE integration to compute likelihood,
        autoregressive models give exact likelihoods directly.

        Args:
            x_continuous: Continuous coordinates, shape (batch, num_atoms, 3)
            encodings: Optional conditional embeddings
            mask: Optional node mask
            temperature: Sampling temperature used to generate samples (default: 1.0).
                        If samples were generated with temperature T, set this to T
                        to get the correct likelihood under the distribution sampled from.
            top_k: Top-k filtering (None to disable). If samples were generated with
                   top-k filtering, set this to the same value.
            top_p: Nucleus (top-p) sampling (None to disable). If samples were generated
                   with top-p filtering, set this to the same value.

        Returns:
            Log likelihood per sample, shape (batch,)
        """
        num_samples = x_continuous.shape[0]
        num_dims = self.datamodule.hparams.num_dimensions

        # Flatten and discretize
        x_flat = x_continuous.reshape(num_samples, -1)
        x_discrete = self.discretize.discretize(x_flat)

        # Create input sequence (shifted right)
        x_input = torch.cat(
            [torch.zeros(num_samples, 1, dtype=torch.long, device=x_discrete.device), x_discrete[:, :-1]], dim=1
        )

        # Get logits
        logits = self.forward(x_input, encodings=encodings, mask=mask)

        # Apply temperature to logits (scale by 1/T before softmax)
        # This gives the likelihood under the temperature-scaled distribution
        scaled_logits = logits / temperature

        # Apply top-k filtering if used
        if top_k is not None:
            v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            scaled_logits = scaled_logits.clone()  # avoid in-place modification issues
            scaled_logits[scaled_logits < v[:, :, [-1]]] = float("-inf")

        # Apply nucleus (top-p) sampling filtering if used
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
            sorted_indices_to_remove[:, :, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
            scaled_logits = scaled_logits.clone()
            scaled_logits[indices_to_remove] = float("-inf")

        # Compute log probabilities under the filtered, temperature-scaled distribution
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        # Gather log probs for actual tokens
        # log_probs: (batch, seq_len, num_bins)
        # x_discrete: (batch, seq_len)
        log_prob_tokens = torch.gather(log_probs, dim=-1, index=x_discrete.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, seq_len)

        # Sum log probs (apply mask if needed)
        if mask is not None:
            mask_expanded = mask.repeat_interleave(num_dims, dim=1)
            log_prob_tokens = log_prob_tokens * mask_expanded.float()

        log_likelihood = log_prob_tokens.sum(dim=-1)  # (batch,)

        return log_likelihood

    def evaluate(
        self,
        sequence,
        true_samples,
        permutations,
        encodings,
        energy_fn,
        tica_model,
        prefix: str = "val",
        proposal_generator=None,
        output_dir=None,
    ):
        """Evaluate the model on a validation/test set."""
        results = super().evaluate(
            sequence,
            true_samples,
            permutations,
            encodings,
            energy_fn,
            tica_model,
            prefix,
            proposal_generator,
            output_dir,
        )

        return results

    @torch.no_grad()
    def generate_samples(
        self,
        batch_size: int,
        permutations: dict[str, torch.Tensor] | None = None,
        encodings: dict[str, torch.Tensor] | None = None,
        dummy_ll: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model autoregressively.

        :param batch_size: The batch size to use for generating samples.
        :param permutations: Optional permutation dictionaries (unused for AR models).
        :param encodings: Optional conditional encodings (e.g., atom types).
        :param dummy_ll: If True, return dummy log likelihood (faster).
        :return: A tuple containing (samples, log_likelihood, prior_samples).
                 Note: prior_samples is a dummy tensor for AR models since there's no prior.
        """
        if encodings is None:
            num_atoms = self.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        num_dims = self.datamodule.hparams.num_dimensions

        local_batch_size = batch_size // self.trainer.world_size

        if encodings is not None:
            encodings = {
                key: tensor.unsqueeze(0).repeat(local_batch_size, 1).to(self.device)
                for key, tensor in encodings.items()
            }

        # Generate discrete tokens autoregressively
        generated_discrete = self._generate_autoregressive(
            batch_size=local_batch_size,
            num_atoms=num_atoms,
            encodings=encodings,
        )  # (batch, seq_len)

        # Convert discrete tokens back to continuous coordinates
        # Use empirical offsets if enabled and initialized, otherwise uniform noise
        if self.hparams.use_empirical_offsets and self.discretize.has_empirical_offsets():
            samples_flat = self.discretize.undiscretize_with_empirical_offsets(generated_discrete)
        else:
            samples_flat = self.discretize.undiscretize_with_noise(generated_discrete)
        samples = samples_flat.reshape(local_batch_size, num_atoms, num_dims)

        # Compute log likelihood (or return dummy)
        if dummy_ll:
            log_p = torch.zeros(local_batch_size, device=self.device)
        else:
            log_p = self.compute_log_likelihood(
                samples,
                encodings,
                temperature=self.hparams.temperature,
                top_k=self.hparams.top_k,
                top_p=self.hparams.top_p,
            )

        # Create dummy "prior samples" for compatibility
        # (AR models don't have a prior in the same sense as flow models)
        prior_samples = torch.randn_like(samples)

        # Gather across devices
        samples = self.all_gather(samples).reshape(batch_size, *samples.shape[1:])
        log_p = self.all_gather(log_p).reshape(-1)
        prior_samples = self.all_gather(prior_samples).reshape(-1, *prior_samples.shape[1:])

        return samples, log_p, prior_samples

    @torch.no_grad()
    def _generate_autoregressive(
        self,
        batch_size: int,
        num_atoms: int,
        encodings: dict | None = None,
    ) -> torch.Tensor:
        """Generate discrete tokens autoregressively with KV-cache for efficiency.

        Args:
            batch_size: Number of samples to generate
            num_atoms: Number of atoms per sample
            encodings: Optional conditional embeddings

        Returns:
            Discrete tokens, shape (batch, num_atoms * num_dims)
        """
        # Use the network's built-in generate method which has KV-cache support
        return self.net.generate(
            batch_size=batch_size,
            num_atoms=num_atoms,
            encodings=encodings,
            temperature=self.hparams.temperature,
            top_k=self.hparams.top_k,
            top_p=self.hparams.top_p,
            device=self.device,
            use_cache=True,  # Enable KV-cache for faster generation
        )

    @torch.no_grad()
    def batched_generate_samples_no_ll(
        self, total_size: int, batch_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples without computing likelihood (faster)."""
        return super().batched_generate_samples(total_size, batch_size, dummy_ll=True)

    def _initialize_empirical_offsets(self) -> None:
        """Initialize empirical within-bin offsets from training data.

        Called automatically before first validation/test if use_empirical_offsets=True.
        """
        if not self.hparams.use_empirical_offsets:
            return

        if self.discretize.has_empirical_offsets():
            logger.info("Empirical offsets already initialized, skipping.")
            return

        logger.info("Computing empirical within-bin offsets from training data...")

        # Get training dataset - try different attribute names used by different datamodules
        train_data = None
        if hasattr(self.datamodule, "data_train"):
            train_data = self.datamodule.data_train
        elif hasattr(self.datamodule, "train_dataset"):
            train_data = self.datamodule.train_dataset

        if train_data is None:
            logger.warning("Training dataset not available, cannot compute empirical offsets.")
            return

        # Collect all training coordinates
        # Handle both raw tensor data and dataset objects
        all_coords = []
        if hasattr(train_data, "data"):
            # TensorDataset with .data attribute containing raw tensor
            raw_data = train_data.data
            # Normalize the data the same way it's normalized during training
            if hasattr(self.datamodule, "normalize"):
                raw_data = self.datamodule.normalize(raw_data)
            all_coords.append(raw_data.reshape(-1))
        else:
            # Iterate through dataset
            for i in range(len(train_data)):
                sample = train_data[i]
                if isinstance(sample, dict):
                    x = sample.get("x", sample.get("coords", None))
                else:
                    x = sample[0] if isinstance(sample, tuple | list) else sample
                if x is not None:
                    all_coords.append(x.reshape(-1))

        if len(all_coords) == 0:
            logger.warning("No coordinates found in training data.")
            return

        all_coords = torch.cat(all_coords, dim=0)
        logger.info(f"Computing offsets from {len(all_coords):,} coordinate values...")

        self.discretize.compute_empirical_offsets(all_coords)
        logger.info("Empirical offsets computed successfully.")

    def on_validation_start(self) -> None:
        """Initialize empirical offsets before first validation."""
        super().on_validation_start() if hasattr(super(), "on_validation_start") else None
        self._initialize_empirical_offsets()

    def on_test_start(self) -> None:
        """Initialize empirical offsets before testing."""
        super().on_test_start() if hasattr(super(), "on_test_start") else None
        self._initialize_empirical_offsets()


if __name__ == "__main__":
    _ = AutoregressiveLitModule(None, None, None, None)
