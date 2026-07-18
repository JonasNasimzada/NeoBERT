import torch

from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    Float64 inputs produce complex128 frequencies; all other supported real
    input dtypes produce complex64 frequencies.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    if dim <= 0 or dim % 2 != 0:
        raise ValueError("RoPE dimension must be a positive even integer")
    if end < 0:
        raise ValueError("RoPE sequence length must be nonnegative")
    real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    positions = torch.arange(0, dim, 2, dtype=real_dtype, device=device)
    inverse_frequencies = 1.0 / (theta ** (positions / dim))
    token_positions = torch.arange(end, dtype=real_dtype, device=device)
    angles = torch.outer(token_positions, inverse_frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """

    ndim = x.ndim
    if ndim < 2:
        raise ValueError("RoPE input must have at least two dimensions")
    expected_shape = (x.shape[1], x.shape[-1])
    if freqs_cis.shape != expected_shape:
        raise ValueError(
            f"RoPE frequencies must have shape {expected_shape}, got {tuple(freqs_cis.shape)}"
        )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if xq.shape != xk.shape:
        raise ValueError("RoPE query and key must have the same shape")
    if xq.device != xk.device or xq.dtype != xk.dtype:
        raise ValueError("RoPE query and key must have the same dtype and device")
    if not torch.is_floating_point(xq):
        raise ValueError("RoPE query and key must use a real floating-point dtype")
    if xq.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even final dimension")

    compute_dtype = torch.float64 if xq.dtype == torch.float64 else torch.float32
    complex_dtype = torch.complex128 if compute_dtype == torch.float64 else torch.complex64
    xq_pairs = xq.to(compute_dtype).reshape(*xq.shape[:-1], -1, 2).contiguous()
    xk_pairs = xk.to(compute_dtype).reshape(*xk.shape[:-1], -1, 2).contiguous()
    xq_ = torch.view_as_complex(xq_pairs)
    xk_ = torch.view_as_complex(xk_pairs)
    frequencies = freqs_cis.to(device=xq.device, dtype=complex_dtype)
    frequencies = reshape_for_broadcast(frequencies, xq_)
    xq_out = torch.view_as_real(xq_ * frequencies).flatten(-2)
    xk_out = torch.view_as_real(xk_ * frequencies).flatten(-2)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_real, k_real = apply_rotary_emb(xq.real.contiguous(), xk.real.contiguous(), freqs_cis)
    q_imag, k_imag = apply_rotary_emb(xq.imag.contiguous(), xk.imag.contiguous(), freqs_cis)
    return torch.complex(q_real, q_imag).to(xq.dtype), torch.complex(k_real, k_imag).to(xk.dtype)
