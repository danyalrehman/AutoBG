"""Muon optimizer implementation.

Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer designed for
training neural networks. It applies Newton-Schulz orthogonalization to the
momentum to improve training dynamics.

Reference:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
    https://kellerjordan.github.io/posts/muon/
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Compute the zeroth power / orthogonalization of G using Newton-Schulz iteration.

    This computes G @ (G.T @ G)^(-1/2) which orthogonalizes the columns of G.
    The Newton-Schulz iteration converges to the orthogonal matrix closest to G.

    Args:
        G: Input matrix to orthogonalize.
        steps: Number of Newton-Schulz iterations.
        eps: Small constant for numerical stability.

    Returns:
        Orthogonalized matrix with the same shape as G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Reshape to 2D for the orthogonalization
    original_shape = G.shape
    G = G.reshape(-1, G.shape[-1])

    # Scale G to have roughly unit Frobenius norm per row
    X = G.bfloat16()
    X /= X.norm() + eps

    # Newton-Schulz iteration
    if X.shape[0] > X.shape[1]:
        # Tall matrix: iterate on X.T @ X
        for _ in range(steps):
            A = X.T @ X
            B = b * A + c * A @ A
            X = a * X + X @ B
    else:
        # Wide or square matrix: iterate on X @ X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    return X.to(G.dtype).reshape(original_shape)


class Muon(Optimizer):
    """Muon optimizer - MomentUm Orthogonalized by Newton-schulz.

    Muon applies Newton-Schulz orthogonalization to the momentum buffer before
    applying updates. This helps maintain better conditioning of the updates
    throughout training.

    For parameters with 2+ dimensions, Muon applies orthogonalized momentum updates.
    For 1D parameters (biases, LayerNorm, etc.), it falls back to standard AdamW.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum factor for Muon updates (default: 0.95).
        betas: Coefficients for computing running averages of gradient and its square
            for AdamW fallback on 1D params (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability in AdamW (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.0).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iteration steps (default: 5).
        adamw_lr_ratio: Learning rate ratio for AdamW fallback on 1D params.
            If None, uses the same lr (default: 0.1).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr_ratio: float | None = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr_ratio=adamw_lr_ratio,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adamw_lr_ratio = group["adamw_lr_ratio"]

            # Compute effective AdamW learning rate
            adamw_lr = lr * adamw_lr_ratio if adamw_lr_ratio is not None else lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if p.ndim >= 2:
                        # Muon momentum buffer for 2D+ params
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        # AdamW state for 1D params
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if p.ndim >= 2:
                    # Muon update for matrices (2D+)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        # Nesterov momentum: use grad + momentum * buf
                        update = grad + momentum * buf
                    else:
                        update = buf

                    # Apply Newton-Schulz orthogonalization
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Scale by sqrt of fan-in for proper scaling
                    scale = max(1, p.shape[0] / p.shape[1]) ** 0.5

                    # Apply weight decay (decoupled, like AdamW)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.add_(update, alpha=-lr * scale)

                else:
                    # AdamW update for vectors (1D)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    step = state["step"]
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    # Compute step size
                    step_size = adamw_lr / bias_correction1

                    # Compute denominator
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                    # Apply weight decay (decoupled)
                    if weight_decay != 0:
                        p.mul_(1 - adamw_lr * weight_decay)

                    # Apply update
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
