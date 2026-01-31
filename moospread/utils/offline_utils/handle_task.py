import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def one_hot(a: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert an integer tensor to a float32 one-hot tensor using PyTorch.

    Args:
        a: Long/Int tensor of arbitrary shape containing class indices.
        num_classes: Total number of classes.
    Returns:
        Tensor with shape ``a.shape + (num_classes,)`` and dtype float32.
    """
    if torch.is_floating_point(a):
        raise ValueError("cannot one-hot encode non-integers (got floating dtype)")
    return F.one_hot(a.to(torch.long), num_classes=num_classes).to(torch.float32)


def help_to_logits(
    x: torch.Tensor, num_classes: int, soft_interpolation: float = 0.6
) -> torch.Tensor:
    """Convert integer labels to *logit* representation with a soft uniform prior.

    Mirrors the original NumPy implementation but in PyTorch.

    Args:
        x: Int tensor of shape ``(n_samples, 1)``.
        num_classes: Number of classes for this position.
        soft_interpolation: Interpolate between one-hot (Dirac) and uniform prior.
    Returns:
        Float tensor of shape ``(n_samples, 1, num_classes-1)``.
    """
    if torch.is_floating_point(x):
        raise ValueError("cannot convert non-integers to logits")

    device = x.device

    # one-hot (n, 1, C)
    one_hot_x = one_hot(x, num_classes=num_classes)  # float32

    # uniform prior to interpolate with
    uniform_prior = torch.full_like(one_hot_x, 1.0 / float(num_classes), device=device)

    # convex combination
    soft_x = soft_interpolation * one_hot_x + (1.0 - soft_interpolation) * uniform_prior

    # convert to log probabilities
    log_p = torch.log(soft_x)

    # remove one degree of freedom: subtract the first component and drop it
    return (log_p[:, :, 1:] - log_p[:, :, :1]).to(torch.float32)


def offdata_to_logits(
    x: torch.Tensor,
    num_classes_on_each_position: List[int],
    soft_interpolation: float = 0.6,
) -> torch.Tensor:
    """Convert a sequence of categorical integers to concatenated logits.

    For each sequence position ``i`` with ``k_i`` classes, we form a length ``k_i-1``
    logit vector (after removing the redundant degree of freedom) and then concatenate
    all positions along the last axis.

    Args:
        x: Int tensor of shape ``(n_samples, seq_len)``.
        num_classes_on_each_position: list of class counts per position.
        soft_interpolation: Interpolation factor in ``[0, 1]``.
    Returns:
        Float tensor of shape ``(n_samples, sum_i (k_i - 1))`` (matches the original).
    """
    # Original code adds +1, then for positions with exactly 1 class, adds another +1
    # to introduce a dummy class. We reproduce that behavior.
    num_classes = [c + 1 for c in num_classes_on_each_position]
    num_classes = [c + 1 if c == 1 else c for c in num_classes]

    logits = []
    seq_len = len(num_classes)
    for i in range(seq_len):
        temp_x = x[:, i].reshape(-1, 1)
        logits.append(help_to_logits(temp_x, num_classes[i], soft_interpolation))

    # concatenate along the last dim, then squeeze (to mimic NumPy's .squeeze())
    out = torch.cat(logits, dim=2).squeeze()
    return out


def help_to_integers(x: torch.Tensor, true_num_of_classes: int) -> torch.Tensor:
    """Convert per-position logits back to class integers.

    Args:
        x: Float tensor of shape ``(n_samples, 1, k-1)`` for a position with ``k`` classes.
        true_num_of_classes: The (possibly dummy-adjusted) number of classes ``k``.
    Returns:
        Int tensor of shape ``(n_samples, 1)`` with the selected classes.
    """
    if not torch.is_floating_point(x):
        raise ValueError("cannot convert non-floats to integers")

    # Special-case: if k == 1 (RFP-Exact-v0 path), always return zeros
    if true_num_of_classes == 1:
        return torch.zeros(x.shape[:-1], dtype=torch.int64, device=x.device)

    # Pad a leading zero component and take argmax along class dim
    # x shape: (n, 1, k-1) -> pad to (n, 1, k)
    x_padded = F.pad(x, pad=(1, 0))  # pad last dim: (left=1, right=0)
    return torch.argmax(x_padded, dim=-1).to(torch.int64)


def offdata_to_integers(x: torch.Tensor, num_classes_on_each_position: List[int]) -> torch.Tensor:
    """Invert ``to_logits``: recover integer classes for each sequence position.

    Note: This follows the original slicing behavior where the concatenated
    per-position logits are packed along the second dimension (after a squeeze).

    Args:
        x: Float tensor with shape ``(n_samples, total_concatenated)`` where
           ``total_concatenated = sum_i (k_i - 1)`` after the same class-count
           adjustments used in ``to_logits``.
        num_classes_on_each_position: List of true class counts per position.
    Returns:
        Int tensor of shape ``(n_samples, seq_len)``.
    """
    # Reproduce the same class-count adjustment as in to_logits
    true_num_classes = [c + 1 for c in num_classes_on_each_position]
    num_classes = [c + 1 if c == 1 else c for c in true_num_classes]

    integers = []
    start = 0
    for k in num_classes:
        width = k - 1
        # Slice along dim=1 to match the original implementation
        temp_x = x[:, start : start + width].reshape(-1, 1, width)
        integers.append(help_to_integers(temp_x, k))
        start += width

    # Concatenate along seq dimension (dim=1)
    return torch.cat(integers, dim=1)


def offdata_z_score_normalize(x: torch.Tensor, 
                              mean: Optional[torch.Tensor] = None,
                              std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score normalize features columnwise (match NumPy semantics).

    Args:
        x: Float tensor of shape ``(n_samples, n_dim)``.
    Returns:
        (x_norm, mean, std) where each has shape compatible with broadcasting.
    """
    if not torch.is_floating_point(x):
        raise ValueError("cannot normalize discrete design values")

    if mean is not None and std is not None:
        x_norm = (x - mean.to(x.device)) / std.to(x.device)
        return x_norm, mean, std
    
    mean = torch.mean(x, dim=0)
    # NumPy's np.std uses population std (ddof=0) by default -> unbiased=False
    std = torch.std(x, dim=0, unbiased=False)
    eps = 1e-6
    std_safe = torch.clamp(std, min=eps)
    x_norm = (x - mean) / std_safe
    return x_norm, mean, std_safe


def offdata_z_score_denormalize(x: torch.Tensor, 
                                x_mean: torch.Tensor, 
                                x_std: torch.Tensor) -> torch.Tensor:
    """Invert z-score normalization.

    Args:
        x: Float tensor ``(n_samples, n_dim)``.
        x_mean: Mean used during normalization.
        x_std: Std used during normalization.
    Returns:
        Denormalized tensor.
    """
    if not torch.is_floating_point(x):
        raise ValueError("cannot denormalize discrete design values")
    return x * x_std.to(x.device) + x_mean.to(x.device)


def offdata_min_max_normalize(x: torch.Tensor,
                              min_val: Optional[torch.Tensor] = None,
                              max_val: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize features columnwise.

    Args:
        x: Tensor of shape ``(n_samples, n_dim)``.
    Returns:
        (x_norm, x_min, x_max)
    """
    
    if min_val is not None and max_val is not None:
        x_norm = (x - min_val.to(x.device)) / (max_val.to(x.device) - min_val.to(x.device))
        return x_norm, min_val, max_val
    
    x_min = torch.min(x, dim=0).values
    x_max = torch.max(x, dim=0).values
    eps = 1e-6
    x_max_x_min_safe = torch.clamp(x_max - x_min, min=eps)
    x_norm = (x - x_min) / x_max_x_min_safe
    return x_norm, x_max-x_max_x_min_safe, x_max


def offdata_min_max_denormalize(x: torch.Tensor, 
                                x_min: torch.Tensor, 
                                x_max: torch.Tensor) -> torch.Tensor:
    """Invert min-max normalization.

    Args:
        x: Tensor of shape ``(n_samples, n_dim)``.
        x_min: Per-dimension min.
        x_max: Per-dimension max.
    Returns:
        Denormalized tensor.
    """
    return x * (x_max.to(x.device) - x_min.to(x.device)) + x_min.to(x.device)
