import torch
from torch.optim import Optimizer
import math
from typing import Iterable


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad.data

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Correct bias
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # Apply weight decay
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it >= cosine_cycle_iters:
        return min_learning_rate

    # Cosine decay
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p.grad is not None, parameters))

    max_l2_norm = float(max_l2_norm)

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0
    )

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
