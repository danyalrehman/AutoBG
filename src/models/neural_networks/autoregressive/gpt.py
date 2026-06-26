"""GPT-style Transformer for autoregressive modeling of molecular coordinates.

This module implements a GPT-2 style transformer architecture for autoregressive
generation of discretized molecular coordinates. Converted from the original
TensorFlow implementation to PyTorch with modern features.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check if Flash Attention is available (PyTorch 2.0+)
FLASH_ATTENTION_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
_FLASH_ATTENTION_WARNING_SHOWN = False


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Input with positional embeddings added
        """
        return x + self.pe[: x.size(1)]


class GPTAttention(nn.Module):
    """GPT-style multi-head attention with KV-cache and Flash Attention support.

    Uses combined QKV projection like the original GPT-2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        self.dropout_p = dropout

        # Warn if flash attention was requested but not available
        global _FLASH_ATTENTION_WARNING_SHOWN
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE and not _FLASH_ATTENTION_WARNING_SHOWN:
            logger.warning(
                "Flash Attention requested but not available. "
                "Falling back to standard attention. "
                "Upgrade to PyTorch 2.0+ for Flash Attention support."
            )
            _FLASH_ATTENTION_WARNING_SHOWN = True

        # Combined QKV projection (GPT-2 style)
        self.c_attn = nn.Linear(dim, 3 * self.inner_dim)
        self.c_proj = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask (only needed when not using flash attention)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass with causal masking and optional KV-cache.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional additional mask (batch, seq_len)
            past_kv: Optional tuple of (past_k, past_v) from previous forward pass
            use_cache: Whether to return KV cache for next step

        Returns:
            Tuple of (output, present_kv) where present_kv is None if use_cache=False
        """
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.c_attn(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # Use Flash Attention when available and no custom mask is needed
        if self.use_flash_attention and attention_mask is None:
            is_causal = past_kv is None
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)
        else:
            # Standard attention with explicit masking
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if past_kv is not None:
                pass  # No causal mask needed for single token generation
            else:
                causal_mask = self.causal_mask[:seq_len, :kv_seq_len]
                attn = attn.masked_fill(causal_mask, float("-inf"))

            # Apply additional attention mask if provided
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(~attn_mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        out = self.c_proj(out)
        return out, present_kv


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU is a gated linear unit with Swish activation, shown to improve
    transformer performance in LLMs (Shazeer, 2020; LLaMA, PaLM).
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate projection
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # Value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


class GPTMLP(nn.Module):
    """GPT-style MLP with GELU or SwiGLU activation."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        hidden_dim = dim * expansion

        if use_swiglu:
            # SwiGLU uses 2/3 of the hidden dim to maintain parameter count
            swiglu_hidden = int(2 * hidden_dim / 3)
            self.c_fc = SwiGLU(dim, swiglu_hidden, bias=True)
            self.c_proj = nn.Linear(swiglu_hidden, dim)
        else:
            self.c_fc = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
            )
            self.c_proj = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GPTBlock(nn.Module):
    """GPT-style transformer block with pre-norm architecture."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        head_dim: int | None = None,
        expansion: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        use_rms_norm: bool = True,
        use_flash_attention: bool = True,
        use_swiglu: bool = True,
    ):
        super().__init__()

        if use_rms_norm:
            self.ln_1 = RMSNorm(dim)
            self.ln_2 = RMSNorm(dim)
        else:
            self.ln_1 = nn.LayerNorm(dim)
            self.ln_2 = nn.LayerNorm(dim)

        self.attn = GPTAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = GPTMLP(dim=dim, expansion=expansion, dropout=dropout, use_swiglu=use_swiglu)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm architecture (GPT-2 style)
        attn_out, present_kv = self.attn(self.ln_1(x), attention_mask, past_kv, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    """GPT-style Transformer for autoregressive coordinate generation.

    This model predicts the next coordinate token given all previous tokens,
    outputting a distribution over discrete bins. Based on the GPT-2 architecture.

    Args:
        num_bins: Number of discrete bins for each coordinate (vocabulary size)
        input_dim: Dimension of each input position (typically 3 for 3D coordinates)
        channels: Hidden dimension of the transformer (n_embd)
        num_layers: Number of transformer blocks (n_layer)
        num_heads: Number of attention heads (n_head)
        max_num_atoms: Maximum number of atoms (positions)
        head_dim: Dimension of each attention head
        expansion: Feed-forward expansion factor
        dropout: Dropout probability
        pos_embed_type: Type of positional embedding ("learned" or "sinusoidal")
        use_rms_norm: Whether to use RMSNorm instead of LayerNorm
        use_swiglu: Whether to use SwiGLU activation instead of GELU
        coordinate_embedding_type: How to embed coordinates ("embedding" or "linear")
        cond_embed: Optional conditional embedding module (e.g., for atom types)
        use_flash_attention: Whether to use Flash Attention
        use_bf16: Whether to use bfloat16 precision for forward pass. Default: False
    """

    def __init__(
        self,
        num_bins: int = 256,
        input_dim: int = 3,
        channels: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_num_atoms: int = 64,
        head_dim: int | None = None,
        expansion: int = 4,
        dropout: float = 0.0,
        pos_embed_type: str = "learned",
        use_rms_norm: bool = True,
        use_swiglu: bool = True,
        coordinate_embedding_type: str = "embedding",
        cond_embed: nn.Module | None = None,
        use_flash_attention: bool = True,
        use_bf16: bool = False,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.input_dim = input_dim
        self.channels = channels
        self.max_seq_len = max_num_atoms * input_dim
        self.use_bf16 = use_bf16

        # Token embedding (wte in GPT-2)
        if coordinate_embedding_type == "embedding":
            self.wte = nn.Embedding(num_bins, channels)
        else:
            self.wte = nn.Linear(num_bins, channels)

        self.coordinate_embedding_type = coordinate_embedding_type

        # Positional embeddings (wpe in GPT-2)
        if pos_embed_type == "learned":
            self.wpe = nn.Embedding(self.max_seq_len, channels)
        else:
            self.wpe = SinusoidalPositionalEmbedding(channels, self.max_seq_len)
        self.pos_embed_type = pos_embed_type

        # Coordinate dimension embedding (which of x, y, z)
        self.coord_dim_embed = nn.Embedding(input_dim, channels)

        # Atom position embedding (which atom in the sequence)
        self.atom_pos_embed = nn.Embedding(max_num_atoms, channels)

        # Optional conditional embedding (e.g., atom types)
        self.cond_embed = cond_embed

        # Dropout
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                GPTBlock(
                    dim=channels,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    expansion=expansion,
                    dropout=dropout,
                    max_seq_len=self.max_seq_len,
                    use_rms_norm=use_rms_norm,
                    use_flash_attention=use_flash_attention,
                    use_swiglu=use_swiglu,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm (ln_f in GPT-2)
        if use_rms_norm:
            self.ln_f = RMSNorm(channels)
        else:
            self.ln_f = nn.LayerNorm(channels)

        # Output projection (tied weights with wte in original GPT-2, but separate here)
        self.lm_head = nn.Linear(channels, num_bins, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        encodings: dict | None = None,
        node_mask: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Forward pass to get logits for next token prediction.

        Args:
            x: Discrete token indices, shape (batch, seq_len)
            encodings: Optional dictionary with conditional embeddings (e.g., atom types)
            node_mask: Optional mask for padding, shape (batch, num_atoms)
            past_kvs: Optional list of (past_k, past_v) tuples for each layer
            use_cache: Whether to return KV cache for incremental generation
            position_offset: Position offset for positional embeddings (used with KV cache)

        Returns:
            Tuple of (logits, present_kvs) where present_kvs is None if use_cache=False
        """
        if self.use_bf16:
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                return self._forward_impl(x, encodings, node_mask, past_kvs, use_cache, position_offset)
        else:
            return self._forward_impl(x, encodings, node_mask, past_kvs, use_cache, position_offset)

    def _forward_impl(
        self,
        x: torch.Tensor,
        encodings: dict | None = None,
        node_mask: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Internal forward implementation."""
        batch_size, seq_len = x.shape
        device = x.device

        # Token embedding
        if self.coordinate_embedding_type == "embedding":
            h = self.wte(x)  # (batch, seq_len, channels)
        else:
            x_onehot = F.one_hot(x, num_classes=self.num_bins).float()
            h = self.wte(x_onehot)

        # Positional embeddings
        positions = torch.arange(position_offset, position_offset + seq_len, device=device)
        if self.pos_embed_type == "learned":
            h = h + self.wpe(positions)
        else:
            h = h + self.wpe.pe[position_offset : position_offset + seq_len]

        # Add coordinate dimension embeddings (which of x, y, z)
        coord_dims = positions % self.input_dim
        h = h + self.coord_dim_embed(coord_dims)

        # Add atom position embeddings
        atom_positions = positions // self.input_dim
        h = h + self.atom_pos_embed(atom_positions)

        # Add conditional embeddings if provided
        if self.cond_embed is not None and encodings is not None:
            cond = self.cond_embed(encodings)  # (batch, num_atoms, channels)
            cond = cond.repeat_interleave(self.input_dim, dim=1)  # (batch, total_seq_len, channels)
            h = h + cond[:, position_offset : position_offset + seq_len, :]

        h = self.drop(h)

        # Create attention mask from node_mask if provided
        attention_mask = None
        if node_mask is not None:
            full_attention_mask = node_mask.repeat_interleave(self.input_dim, dim=1)
            attention_mask = full_attention_mask[:, : position_offset + seq_len]

        # Pass through transformer blocks
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, present_kv = block(h, attention_mask, past_kv, use_cache)
            if use_cache:
                present_kvs.append(present_kv)

        # Final layer norm and projection
        h = self.ln_f(h)
        logits = self.lm_head(h)

        return logits, present_kvs

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_atoms: int,
        encodings: dict | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: torch.device = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate samples with KV-cache for efficiency.

        Args:
            batch_size: Number of samples to generate
            num_atoms: Number of atoms (positions)
            encodings: Optional conditional embeddings
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            device: Device to generate on
            use_cache: Whether to use KV-cache for faster generation

        Returns:
            Generated discrete tokens, shape (batch, num_atoms * input_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        seq_len = num_atoms * self.input_dim

        # Start with start token (0)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        past_kvs = None

        for i in range(seq_len):
            # Get logits for next position
            if use_cache and past_kvs is not None:
                logits, past_kvs = self.forward(
                    generated[:, -1:],
                    encodings,
                    past_kvs=past_kvs,
                    use_cache=True,
                    position_offset=i,
                )
                logits = logits[:, -1, :]
            else:
                logits, past_kvs = self.forward(
                    generated,
                    encodings,
                    use_cache=use_cache,
                    position_offset=0,
                )
                logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        # Remove the initial start token
        return generated[:, 1:]


class GPTSinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for continuous/integer values (e.g., positions, sequence length)."""

    def __init__(self, embed_size: int, div_value: float = 10000.0):
        super().__init__()
        self.embed_size = embed_size
        self.div_value = div_value
        assert self.embed_size % 2 == 0, "embed_size must be even."

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        position = data.float().unsqueeze(-1)  # Shape: [..., 1]
        div_term = torch.exp(
            torch.arange(0, self.embed_size, 2, device=data.device) * -(math.log(self.div_value) / self.embed_size)
        )  # Shape: [embed_size // 2]
        sinusoid_inp = position * div_term  # Shape: [..., embed_size // 2]
        pos_embedding = torch.zeros(*sinusoid_inp.shape[:-1], self.embed_size, device=data.device)
        pos_embedding[..., 0::2] = torch.sin(sinusoid_inp)
        pos_embedding[..., 1::2] = torch.cos(sinusoid_inp)
        return pos_embedding


class ConditionalEmbedder(nn.Module):
    """Embedder for conditional information in the transferable setting.

    Embeds atom type, amino acid type, amino acid position, and sequence length
    to condition the autoregressive model on the molecular structure.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_atom_emb: int = 64,
        num_residue_emb: int = 32,
        sinusoid_div_value: float = 10000.0,
    ):
        """Initialize the conditional embedder.

        Args:
            hidden_dim: Hidden dimension for embeddings
            output_dim: Output dimension after projection
            num_atom_emb: Number of atom type embeddings (54 unique atom types + padding)
            num_residue_emb: Number of residue type embeddings (20 amino acids + padding)
            sinusoid_div_value: Divisor for sinusoidal position embeddings
        """
        super().__init__()

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=hidden_dim)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=hidden_dim)
        self.residue_pos_embed = GPTSinusoidalEmbedding(embed_size=hidden_dim, div_value=sinusoid_div_value)
        self.seq_len_embed = GPTSinusoidalEmbedding(embed_size=hidden_dim, div_value=sinusoid_div_value)

        self.proj = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, encodings: dict) -> torch.Tensor:
        """Embed conditional information.

        Args:
            encodings: Dictionary containing:
                - 'atom_type': Atom type indices, shape (batch, num_atoms)
                - 'aa_type': Amino acid type indices, shape (batch, num_atoms)
                - 'aa_pos': Amino acid position indices, shape (batch, num_atoms)
                - 'seq_len': Sequence length, shape (batch, 1)

        Returns:
            Embeddings of shape (batch, num_atoms, output_dim)
        """
        atom_type = encodings.get("atom_type")
        aa_type = encodings.get("aa_type")
        aa_pos = encodings.get("aa_pos")
        seq_len = encodings.get("seq_len")

        if atom_type is None:
            raise ValueError("Expected 'atom_type' in encodings")

        # Get embeddings
        atom_emb = self.atom_embed(atom_type)  # (batch, num_atoms, hidden_dim)
        residue_emb = self.residue_embed(aa_type)  # (batch, num_atoms, hidden_dim)
        pos_emb = self.residue_pos_embed(aa_pos)  # (batch, num_atoms, hidden_dim)

        # Expand seq_len embedding to match num_atoms
        num_atoms = atom_type.shape[1]
        seq_len_emb = self.seq_len_embed(seq_len).expand(-1, num_atoms, -1)  # (batch, num_atoms, hidden_dim)

        # Concatenate all embeddings and project
        x = torch.cat([atom_emb, residue_emb, pos_emb, seq_len_emb], dim=-1)
        return self.proj(x)
