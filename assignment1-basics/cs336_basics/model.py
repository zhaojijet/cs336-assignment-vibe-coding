import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Root Mean Square Normalization.
    x: Input tensor
    weight: Scale parameter
    eps: Epsilon for numerical stability
    """
    # Calculate RMS: sqrt(mean(x^2))
    # Keep dim for broadcasting
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid Linear Unit (SiLU) / Swish activation.
    SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Softmax function.
    Exp(x) / Sum(Exp(x))
    """
    # For numerical stability, subtract max
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Linear layer: xW^T
    x: Input tensor (..., d_in)
    weight: Weight tensor (d_out, d_in)
    """
    return F.linear(x, weight)


def embedding(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup.
    tokens: Input indices (...,)
    weight: Embedding matrix (vocab_size, d_model)
    """
    return F.embedding(tokens, weight)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy loss.
    logits: (batch_size, vocab_size)
    targets: (batch_size,)
    """
    return F.cross_entropy(logits, targets)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scalar Dot Product Attention.
    Args:
        q: (..., seq_len, d_k)
        k: (..., seq_len, d_k)
        v: (..., seq_len, d_v)
        mask: optional boolean mask
    """
    d_k = q.size(-1)
    # (..., seq_len, d_k) @ (..., d_k, seq_len) -> (..., seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_probs = softmax(scores, dim=-1)
    # (..., seq_len, seq_len) @ (..., seq_len, d_v) -> (..., seq_len, d_v)
    return torch.matmul(attn_probs, v)


def rotary_pos_emb(
    x: torch.Tensor, theta: float, positions: torch.Tensor = None
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE).
    x: (..., seq_len, d_head)
    theta: frequency scaling parameter
    positions: (..., seq_len) optional position indices. If None, inference from shape.
    """
    # d_head must be even
    d = x.shape[-1]
    seq_len = x.shape[-2]
    assert d % 2 == 0

    if positions is None:
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    else:
        positions = positions.float()

    # Create frequencies
    # freqs = 1 / (theta ** (2 * i / d)) for i in range(d/2)
    indices = torch.arange(0, d, 2, device=x.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (indices / d))

    # freqs: (d/2,)
    # angles: (..., seq_len, d/2)
    angles = torch.einsum("...s, f -> ...sf", positions, freqs)

    # Repeat angles for sin/cos: [theta0, theta0, theta1, theta1, ...]
    angles = torch.repeat_interleave(angles, 2, dim=-1)  # (..., seq_len, d)

    if x.ndim > angles.ndim:
        # x is likely (batch, heads, seq, dim), angles is (batch, seq, dim)
        # or x is (batch, seq, dim), angles is (seq, dim)
        # We want to match dims from the right.
        # If x.ndim == 4 and angles.ndim == 3, unsqueeze(1) makes it (b, 1, s, d)
        angles = angles.unsqueeze(-3)

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    def rotate_every_two(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(x.shape[:-2] + (-1,))

    return (x * cos) + (rotate_every_two(x) * sin)


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multi-Head Self Attention.
    in_features: (batch, seq_len, d_in)
    weights: (d_model, d_in) if d_model==d_in.
    """
    batch_size, seq_len, d_in = in_features.shape
    d_head = d_model // num_heads

    # 1. Project Q, K, V
    q = linear(in_features, q_proj_weight)  # (b, s, d_model)
    k = linear(in_features, k_proj_weight)
    v = linear(in_features, v_proj_weight)

    # 2. Split heads
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

    # 3. Attention
    attn_out = scaled_dot_product_attention(q, k, v, mask=mask)  # (b, h, s, d)

    # 4. Concat heads
    attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

    # 5. Output projection
    return linear(attn_out, o_proj_weight)


def swiglu(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU activation.
    x: Input tensor (..., d_model)
    w1: Gate weight (d_ff, d_model)
    w2: Output weight (d_model, d_ff)
    w3: Up weight (d_ff, d_model)
    """
    x_w1 = linear(x, w1_weight)
    x_w3 = linear(x, w3_weight)
    gate = silu(x_w1) * x_w3
    return linear(gate, w2_weight)


def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    d_model: int,
    num_heads: int,
    d_ff: int,
    theta: float,
    mask: torch.Tensor | None = None,
    rope_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Transformer Block with Pre-Norm and RoPE.
    """
    # 1. Layer Norm 1
    # Note: weights keys must match what's expected.
    # We assume keys like "ln1.weight", "attn.q_proj.weight", etc. are passed in a way that maps correctly.
    # The adapter passes a flat dict with specific keys.

    # Attention Block
    h = rms_norm(x, weights["ln1.weight"], eps=1e-5)

    # MHA logic inline to support RoPE
    # Project Q, K, V
    q = linear(h, weights["attn.q_proj.weight"])
    k = linear(h, weights["attn.k_proj.weight"])
    v = linear(h, weights["attn.v_proj.weight"])

    # Split heads
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

    # Apply RoPE to Q and K
    q = rotary_pos_emb(q, theta, rope_positions)
    k = rotary_pos_emb(k, theta, rope_positions)

    # Attention
    attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
    attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
    attn_out = linear(attn_out, weights["attn.output_proj.weight"])

    # Residual
    x = x + attn_out

    # 2. Layer Norm 2 + FFN (SwiGLU)
    h = rms_norm(x, weights["ln2.weight"], eps=1e-5)
    ffn_out = swiglu(
        h,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
    )

    # Residual
    x = x + ffn_out
    return x


def transformer_lm(
    in_indices: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
) -> torch.Tensor:
    """
    Transformer Language Model.
    """
    # 1. Embedding
    x = embedding(in_indices, weights["token_embeddings.weight"])

    # Causal Mask
    s = in_indices.shape[-1]
    mask = torch.tril(torch.ones(s, s, device=x.device)).bool()

    # Position IDs for RoPE
    pos_ids = torch.arange(s, device=x.device)

    # 2. Transformer Blocks
    for i in range(num_layers):
        # Extract weights for this layer
        # The weights dictionary has keys like "layers.0.ln1.weight"
        layer_weights = {
            k.split(f"layers.{i}.")[-1]: v
            for k, v in weights.items()
            if k.startswith(f"layers.{i}.")
        }

        x = transformer_block(
            x,
            layer_weights,
            d_model,
            num_heads,
            d_ff,
            rope_theta,
            mask=mask,
            rope_positions=pos_ids,
        )

    # 3. Final Layer Norm
    x = rms_norm(x, weights["ln_final.weight"], eps=1e-5)

    # 4. Output Projection
    logits = linear(x, weights["lm_head.weight"])
    return logits
