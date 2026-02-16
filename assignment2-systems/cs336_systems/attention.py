import torch


class FlashAttentionFunctionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # q: (batch_size, n_heads, seq_len_q, d_head)
        # k: (batch_size, n_heads, seq_len_k, d_head)
        # v: (batch_size, n_heads, seq_len_k, d_head)

        # Computes:
        # S = Q @ K^T / sqrt(d)
        # P = softmax(S)
        # O = P @ V

        d_head = q.shape[-1]
        scale = d_head**-0.5

        # (batch_size, n_heads, seq_len_q, seq_len_k)
        S = torch.einsum("...qd,...kd->...qk", q, k) * scale

        if is_causal:
            seq_len_q = q.shape[-2]
            seq_len_k = k.shape[-2]
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1
            ).bool()
            S.masked_fill_(mask, float("-inf"))

        P = torch.softmax(S, dim=-1)

        # (batch_size, n_heads, seq_len_q, d_head)
        O = torch.einsum("...qk,...kd->...qd", P, v)

        # Save for backward: q, k, v, L
        # L is logsumexp(S, dim=-1)
        # Tests require L to be saved with shape (batch, n_heads, seq_len_q)
        # Actually tests check for shape (batch_size, n_queries) if 2D, but inputs are 4D (batch, n_heads, seq_len, d)
        # The tests flatten batch and heads? No, let's look at test_attention.py again.
        # It checks `t.shape == (q.shape[0], q.shape[1])`. Wait, q shape in tests is (batch, queries, d). 3D.
        # My implementation supports 4D (includes heads) which is more general.
        # But let's see. logic should be robust to dims.

        L = torch.logsumexp(S, dim=-1)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        # Recompute P and S?
        # S = Q @ K^T / scale
        # We can recompute S.
        S = torch.einsum("...qd,...kd->...qk", q, k) * scale
        if is_causal:
            seq_len_q = q.shape[-2]
            seq_len_k = k.shape[-2]
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1
            ).bool()
            S.masked_fill_(mask, float("-inf"))

        P = torch.softmax(S, dim=-1)

        # dV = P^T @ dO
        dV = torch.einsum("...qk,...qd->...kd", P, dO)

        # dP = dO @ V^T
        dP = torch.einsum("...qd,...kd->...qk", dO, v)

        # dS = P * (dP - rowsum(dP * P))
        # D = rowsum(dO * O)
        D = torch.sum(dO * O, dim=-1, keepdim=True)
        dS = P * (dP - D)

        # dQ = dS @ K * scale
        dQ = torch.einsum("...qk,...kd->...qd", dS, k) * scale

        # dK = dS^T @ Q * scale
        dK = (
            torch.einsum("...qk,...qd->...kd", dS, q) * scale
        )  # Transpose dS means switch q and k dims
        # Wait: dS is (..., q, k). dK needs (..., k, d).
        # dK = (dS)^T @ Q = (..., k, q) @ (..., q, d) = (..., k, d). Correct.
        # But einsum above: dS is ...qk. q is ...qd. result ...kd.
        # indices: q (query dim), k (key dim), d (head dim).
        # dK_kd = sum_q dS_qk * Q_qd.
        # einsum('...qk,...qd->...kd', dS, q) matches this.

        return dQ, dK, dV, None


class FlashAttentionFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
