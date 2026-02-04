import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

BLOCK_M = 128
ALLOW_TF32 = True

# FP8 constants
FP8_E4M3_MAX = 448.0  # Max value for E4M3 format

# INT4 constants
INT4_MIN = -8
INT4_MAX = 7


def quantize_fp8_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 E4M3 with per-row scaling.

    Args:
        x: Input tensor of shape (..., K)

    Returns:
        x_fp8: Quantized tensor in float8_e4m3fn
        x_scale: Per-row scale factors (for last dim)
    """
    # Compute per-row max absolute value
    row_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # Scale to FP8 range
    scale = row_max / FP8_E4M3_MAX
    # Quantize
    x_scaled = x / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def dequantize_fp8_rowwise(x_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Dequantize FP8 tensor with per-row scaling.

    Args:
        x_fp8: FP8 tensor
        scale: Per-row scale factors
        dtype: Output dtype

    Returns:
        Dequantized tensor
    """
    return x_fp8.to(dtype) * scale.unsqueeze(-1).to(dtype)


# ============================================================================
# QAT (Quantization-Aware Training) Functions with STE
# ============================================================================

class FakeQuantizeFP8RowwiseSTE(torch.autograd.Function):
    """
    Fake quantization for FP8 with Straight-Through Estimator (STE).

    Forward: quantize → dequantize (simulates quantization noise)
    Backward: STE (gradients pass through unchanged)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Quantize to FP8
        x_fp8, scale = quantize_fp8_rowwise(x)
        # Immediately dequantize back
        x_fake_quant = dequantize_fp8_rowwise(x_fp8, scale, dtype=x.dtype)
        return x_fake_quant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradients through unchanged
        return grad_output


def fake_quantize_fp8_rowwise(x: torch.Tensor) -> torch.Tensor:
    """
    Apply fake FP8 quantization with STE for QAT.

    Args:
        x: Input tensor

    Returns:
        Fake-quantized tensor (same dtype as input, but with quantization noise)
    """
    return FakeQuantizeFP8RowwiseSTE.apply(x)


class FakeQuantizeINT4RowwiseSTE(torch.autograd.Function):
    """
    Fake quantization for INT4 with Straight-Through Estimator (STE).

    Uses symmetric quantization with range [-8, 7] (4-bit signed).
    Per-row (group) scaling for dynamic range adaptation.

    Forward: quantize → dequantize (simulates quantization noise)
    Backward: STE (gradients pass through unchanged)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
        original_shape = x.shape
        dtype = x.dtype

        # Reshape for group-wise quantization if needed
        if group_size > 0 and x.shape[-1] > group_size:
            # Reshape to (..., num_groups, group_size)
            num_groups = x.shape[-1] // group_size
            remainder = x.shape[-1] % group_size
            if remainder != 0:
                # Pad to make divisible
                pad_size = group_size - remainder
                x = torch.nn.functional.pad(x, (0, pad_size))
                num_groups = x.shape[-1] // group_size
            x = x.view(*x.shape[:-1], num_groups, group_size)
            grouped = True
        else:
            grouped = False

        # Compute per-row (or per-group) scale
        row_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = row_max / INT4_MAX  # Scale to [-8, 7] range

        # Quantize: scale down, round, clamp to INT4 range
        x_scaled = x / scale
        x_int = torch.round(x_scaled).clamp(INT4_MIN, INT4_MAX)

        # Dequantize: scale back up
        x_fake_quant = x_int * scale

        # Reshape back if grouped
        if grouped:
            x_fake_quant = x_fake_quant.view(*original_shape[:-1], -1)
            # Remove padding if added
            if x_fake_quant.shape[-1] != original_shape[-1]:
                x_fake_quant = x_fake_quant[..., :original_shape[-1]]

        return x_fake_quant.to(dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # STE: pass gradients through unchanged
        return grad_output, None


def fake_quantize_int4_rowwise(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Apply fake INT4 quantization with STE for QAT.

    Args:
        x: Input tensor
        group_size: Group size for quantization (0 for per-row)

    Returns:
        Fake-quantized tensor (same dtype as input, but with quantization noise)
    """
    return FakeQuantizeINT4RowwiseSTE.apply(x, group_size)



@triton.jit
def _compute_expert_block(
    E_idx, E_mask,
    M_in_idx,
    N_block, N_mask,
    X_ptr, stride_xm, stride_xk,
    W_ptr, stride_we, stride_wk, stride_wn,
    K,
    acc,
    no_k_mask,
    BLOCK_K,
    allow_tf32=True,
):

    K_block = tl.arange(0, BLOCK_K)
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we
    iters = tl.cdiv(K, BLOCK_K)

    for K_block_id in range(iters):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])

        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc = tl.dot(x, w, acc, allow_tf32=allow_tf32)
    return acc


@triton.jit
def _compute_expert_block_fp8(
    E_idx, E_mask,
    M_in_idx,
    N_block, N_mask,
    X_ptr, X_scale_ptr, stride_xm, stride_xk,
    W_ptr, W_scale_ptr, stride_we, stride_wk, stride_wn, stride_ws,
    K,
    acc,
    no_k_mask,
    BLOCK_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FP8 version of expert block computation with rowwise dequantization.

    X_ptr: FP8 input tensor
    X_scale_ptr: Input row scales (per-token)
    W_ptr: FP8 weight tensor
    W_scale_ptr: Weight row scales (per-expert, per-output-row)
    stride_ws: Stride for weight scales (expert dimension)
    """
    K_block = tl.arange(0, BLOCK_K)

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we

    # Load weight scales for this expert's output rows (N dimension)
    # W_scale shape: (E, N) where N is output dimension
    W_scale_ptrs = W_scale_ptr + E_idx * stride_ws + N_block
    w_scale = tl.load(W_scale_ptrs, mask=N_mask)  # (BLOCK_N,)

    # Load input scales for the tokens in this block
    x_scale = tl.load(X_scale_ptr + M_in_idx, mask=E_mask)  # (BLOCK_M,)

    iters = tl.cdiv(K, BLOCK_K)

    # Compute this expert's contribution separately
    expert_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for K_block_id in range(iters):
        if no_k_mask:
            x_fp8 = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            w_fp8 = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x_fp8 = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w_fp8 = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])

        # Dequantize to float32 for computation
        x = x_fp8.to(tl.float32)
        w = w_fp8.to(tl.float32)

        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        expert_acc = tl.dot(x, w, expert_acc, allow_tf32=True)

    # Apply scales only to this expert's contribution: expert_acc[m, n] *= x_scale[m] * w_scale[n]
    scale_matrix = x_scale[:, None] * w_scale[None, :]
    expert_acc = expert_acc * scale_matrix

    # Add to main accumulator
    acc = acc + expert_acc
    return acc


def _scatter2scatter_fp8_configs():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]


@triton.autotune(configs=_scatter2scatter_fp8_configs(), key=['M', 'N', 'K'])
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _scatter2scatter_fp8(
    X_ptr, X_scale_ptr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    W_ptr, W_scale_ptr, stride_we, stride_wk: tl.constexpr, stride_wn: tl.constexpr, stride_ws,
    Y_ptr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    B_ptr, stride_be: tl.constexpr, stride_bn: tl.constexpr,
    grouped_idx_ptr, expert_idxs_ptr,
    FAN_OUT: tl.constexpr,
    M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    x_grouped: tl.constexpr, y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    """
    FP8 version of scatter2scatter kernel.

    X_ptr: FP8 input tensor
    X_scale_ptr: Input row scales
    W_ptr: FP8 weight tensor
    W_scale_ptr: Weight row scales (E, N)
    """
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT

    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < (FAN_OUT * M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_k_mask = K % BLOCK_K == 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)
    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        E_M_idx = M_idx
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = E_M_idx // FAN_OUT
        acc = _compute_expert_block_fp8(
            E_idx, E_mask,
            M_in_idx, N_block, N_mask,
            X_ptr, X_scale_ptr, stride_xm, stride_xk,
            W_ptr, W_scale_ptr, stride_we, stride_wk, stride_wn, stride_ws,
            K,
            acc,
            no_k_mask,
            BLOCK_K,
            BLOCK_M,
            BLOCK_N,
        )

    if B_ptr is not None:
        B_blk_ptrs = B_ptr + E_idxs[:, None] * stride_be + N_block[None, :] * stride_bn
        acc += tl.load(B_blk_ptrs, mask=M_boundary_mask[:, None] & N_mask[None, :])

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])


def scatter2scatter_fp8(
    X_fp8: torch.Tensor, X_scale: torch.Tensor,
    W_fp8: torch.Tensor, W_scale: torch.Tensor,
    sorted_expert_idxs: torch.Tensor, sorted_scattered_idxs: torch.Tensor, k: int,
    b: Optional[torch.Tensor] = None,
    x_grouped: bool = False, y_grouped: bool = False,
    out: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 scatter2scatter operation.

    Args:
        X_fp8: FP8 input tensor (M, K) in float8_e4m3fn
        X_scale: Input row scales (M,)
        W_fp8: FP8 weight tensor (E, K, N) in float8_e4m3fn
        W_scale: Weight row scales (E, N)
        sorted_expert_idxs: Sorted expert indices
        sorted_scattered_idxs: Sorted scattered indices
        k: top-k value
        b: Optional bias (E, N)
        x_grouped: Whether input is already grouped
        y_grouped: Whether output should be grouped
        out: Optional output buffer
        output_dtype: Output dtype (default float32)

    Returns:
        Output tensor (L, N) in output_dtype
    """
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X_fp8.size(0) * k

    y_dim = W_fp8.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        output = torch.empty((L_scattered, y_dim), device=X_fp8.device, dtype=output_dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        output = out

    scatter2scatter_fp8_compileable(
        output, W_fp8, W_scale, X_fp8, X_scale, k,
        sorted_expert_idxs, sorted_scattered_idxs,
        b, x_grouped, y_grouped
    )
    return output


@torch.library.custom_op("scattermoe::scatter2scatter_fp8", mutates_args={"output"})
def scatter2scatter_fp8_compileable(
        output: torch.Tensor,
        W_fp8: torch.Tensor,
        W_scale: torch.Tensor,
        X_fp8: torch.Tensor,
        X_scale: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        b: Optional[torch.Tensor],
        x_grouped: bool, y_grouped: bool) -> None:
    def grid(META):
        grid_num = (
            triton.cdiv(sorted_expert_idxs.size(0), META["BLOCK_M"]) *
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid_num

    if b is None:
        stride_be = stride_bk = 0
    else:
        stride_be, stride_bk = b.stride()

    _scatter2scatter_fp8[grid](
        X_fp8, X_scale, X_fp8.stride(0), X_fp8.stride(1),
        W_fp8, W_scale, W_fp8.stride(0), W_fp8.stride(1), W_fp8.stride(2), W_scale.stride(0),
        output, output.stride(0), output.stride(1),
        b, stride_be, stride_bk,
        grouped_idx_ptr=sorted_scattered_idxs,
        expert_idxs_ptr=sorted_expert_idxs,
        FAN_OUT=k,
        M=X_fp8.size(0),
        K=X_fp8.size(1),
        N=output.size(1), E=W_fp8.size(0),
        BLOCK_M=BLOCK_M,
        ACC_TYPE=tl.float32,
        x_grouped=x_grouped, y_grouped=y_grouped,
    )


def _scatter2scatter_configs():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=_scatter2scatter_configs(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _scatter2scatter(
    X_ptr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    W_ptr, stride_we, stride_wk: tl.constexpr, stride_wn: tl.constexpr,
    Y_ptr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    B_ptr, stride_be: tl.constexpr, stride_bn: tl.constexpr,
    grouped_idx_ptr, expert_idxs_ptr,
    # block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    # OUT_M,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr, y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT

    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < (FAN_OUT * M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_k_mask = K % BLOCK_K == 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)
    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        E_M_idx = M_idx
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = E_M_idx // FAN_OUT
        acc = _compute_expert_block(
            E_idx, E_mask,
            M_in_idx, N_block, N_mask,
            X_ptr, stride_xm, stride_xk,
            W_ptr, stride_we, stride_wk, stride_wn,
            K,
            acc,
            no_k_mask,
            BLOCK_K,
            allow_tf32=allow_tf32,
        )

    if B_ptr is not None:
        B_blk_ptrs = B_ptr + E_idxs[:, None] * stride_be + N_block[None, :] * stride_bn
        acc += tl.load(B_blk_ptrs, mask=M_boundary_mask[:, None] & N_mask[None, :])

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])

def scatter2scatter(X, W, sorted_expert_idxs, sorted_scattered_idxs, k,
                    b=None,
                    x_grouped=False, y_grouped=False,
                    out=None):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    # Pre-kernel setup
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        output = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        output = out

    scatter2scatter_compileable(output, W, X, k, sorted_expert_idxs, sorted_scattered_idxs,
                                b, x_grouped, y_grouped)
    return output


@torch.library.custom_op("scattermoe::scatter2scatter", mutates_args={"output"})
def scatter2scatter_compileable(
        output: torch.Tensor,
        W: torch.Tensor,
        X: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        b: Optional[torch.Tensor],
        x_grouped: bool, y_grouped: bool) -> None:
    def grid(META):
        grid_num = (
            triton.cdiv(sorted_expert_idxs.size(0), META["BLOCK_M"]) *
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid_num

    if b is None:
        b = None
        stride_be = stride_bk = 0
    else:
        stride_be, stride_bk = b.stride()

    _scatter2scatter[grid](
        # X_ptr, stride_xm, stride_xk,
        X, X.stride(0), X.stride(1),
        # W_ptr, stride_we, stride_wk, stride_wn,
        W, W.stride(0), W.stride(1), W.stride(2),
        # Y_ptr, stride_ym, stride_yn,
        output, output.stride(0), output.stride(1),
        # B_ptr, stride_be, stride_bk
        b, stride_be, stride_bk,
        grouped_idx_ptr=sorted_scattered_idxs,
        expert_idxs_ptr=sorted_expert_idxs,
        # block_start_idx_ptr=padded_block_idxs,
        FAN_OUT=k,
        M=X.size(0),
        K=X.size(1),
        N=output.size(1), E=W.size(0),
        BLOCK_M=BLOCK_M,
        ACC_TYPE=tl.float32,
        allow_tf32=ALLOW_TF32,
        x_grouped=x_grouped, y_grouped=y_grouped,
    )


def _config_XtY():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'BLOCK_M': 32}, num_stages=4, num_warps=4),
    ]

def group_bwd_W(DY, X, expert_offsets, E, has_bias=False):
    DWt = torch.zeros((E, DY.size(-1), X.size(-1)), device=DY.device, dtype=DY.dtype)
    DW = DWt.permute(0, 2, 1)
    if has_bias:
        Db = torch.zeros((E, DY.size(-1)), device=DY.device, dtype=DY.dtype)
    else:
        Db = None
    groupXtY_compileable(E, DW, Db, DY, X, expert_offsets)
    return DW, Db


@torch.library.custom_op("scattermoe::groupXtY", mutates_args={"DW"})
def groupXtY_compileable(
        E: int,
        DW: torch.Tensor,
        Db: Optional[torch.Tensor],
        DY: torch.Tensor,
        X: torch.Tensor,
        expert_offsets: torch.Tensor) -> None:
    def grid(META):
        grid = (
            E * triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid
    
    if Db is None:
        stride_dbe = 0
        stride_dbn = 0
    else:
        stride_dbe, stride_dbn = Db.stride()

    _groupXtY[grid](
        # DY_ptr, stride_dym, stride_dyk,
        DY, DY.stride(0), DY.stride(1),
        # X_ptr, stride_xm, stride_xn,
        X, X.stride(0), X.stride(1),
        # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
        DW, DW.stride(0), DW.stride(1), DW.stride(2),
        # Db_ptr, stride_dwe, stride_dbn,
        Db, stride_dbe, stride_dbn,
        # expert_offsets_ptr,
        expert_offsets,
        # K: tl.constexpr, N: tl.constexpr,
        M=DY.size(0), N=DY.size(-1), K=X.size(-1),
        # ACC_TYPE: tl.constexpr,
        ACC_TYPE=tl.float32,
        allow_tf32=ALLOW_TF32
    )


@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _groupXtY(
    DY_ptr, stride_dym, stride_dyk,
    X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn,
    Db_ptr, stride_dbe, stride_dbn,
    expert_offsets_ptr,
    M, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    # pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)


    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk
        if (Db_ptr is not None) and (K_block_id == 0):
            _xty_and_bias(
                E_idx, start_idx, end_idx,
                M_block,
                K_block, K_mask, N_block, N_mask, 
                dy_blk_ptrs, stride_dym,
                xt_blk_ptrs, stride_xm,
                DW_ptr, stride_dwe, stride_dwk, stride_dwn,
                Db_ptr, stride_dbe, stride_dbn,
                BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                allow_tf32, NO_K_MASK, NO_N_MASK,
                compute_bias=True
            )
        else:
            _xty_and_bias(
                E_idx, start_idx, end_idx,
                M_block,
                K_block, K_mask, N_block, N_mask, 
                dy_blk_ptrs, stride_dym,
                xt_blk_ptrs, stride_xm,
                DW_ptr, stride_dwe, stride_dwk, stride_dwn,
                Db_ptr, stride_dbe, stride_dbn,
                BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                allow_tf32, NO_K_MASK, NO_N_MASK,
                compute_bias=False
            )


@triton.jit
def _xty_and_bias(
        E_idx, start_idx, end_idx,
        M_block,
        K_block, K_mask, N_block, N_mask, 
        dy_blk_ptrs, stride_dym,
        xt_blk_ptrs, stride_xm,
        DW_ptr, stride_dwe, stride_dwk, stride_dwn,
        Db_ptr, stride_dbe, stride_dbn,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
        allow_tf32, NO_K_MASK, NO_N_MASK,
        compute_bias: tl.constexpr
    ):

    if compute_bias:
        db_acc = tl.zeros((BLOCK_N,), dtype=ACC_TYPE)
    else:
        db_acc = None

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
    for i in range(0, iters):
        M_mask = (i * BLOCK_M + M_block) < end_idx
        if NO_K_MASK:
            xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
        else:
            xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
        if NO_N_MASK:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
        else:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
            
        acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

        xt_blk_ptrs += BLOCK_M * stride_xm
        dy_blk_ptrs += BLOCK_M * stride_dym

        if compute_bias:
            db_acc += tl.sum(dy, axis=0)

    DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
    acc = acc.to(DW_blk_ptrs.dtype.element_ty)
    tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])
    if compute_bias:
        Db_blk_ptrs =  Db_ptr + E_idx * stride_dbe + N_block * stride_dbn
        tl.store(Db_blk_ptrs, db_acc, mask=N_mask)



def _config_grouping():
    return [
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]

def group(A, sorted_expert_idxs, coeff=None, fan_out=1, out=None):
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N
    if out is not None:
        Y = out
    else:
        Y = torch.empty((N, K), dtype=A.dtype, device=A.device)
    group_compileable(A, K, N, Y, coeff, coeff is not None, fan_out, sorted_expert_idxs)
    return Y


@torch.library.custom_op("scattermoe::group", mutates_args={"Y"})
def group_compileable(
        A: torch.Tensor,
        K: int,
        N: int,
        Y: torch.Tensor,
        coeff: torch.Tensor, has_coeff: bool,
        fan_out: int,
        sorted_expert_idxs: torch.Tensor) -> None:
    def grid(META):
        grid_num = (triton.cdiv(META['N'], META['BLOCK_N']),)
        return grid_num
    _group[grid](
        # A_ptr, stride_an, stride_ai,
        A, A.stride(0), A.stride(1), has_coeff, coeff, fan_out,
        # Y_ptr, stride_yn, stride_yk,
        Y, Y.stride(0), Y.stride(1),
        # grouped_idx_ptr,
        sorted_expert_idxs,
        # N: tl.constexpr, K: tl.constexpr,
        N, K
    )


@triton.autotune(configs=_config_grouping(), key=['K'])
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0
})
@triton.jit
def _group(
    src_ptr, stride_sn, stride_sk, has_coeff: tl.constexpr, coeff_ptr, FAN_OUT: tl.constexpr,
    tgt_ptr, stride_tn, stride_ti,
    grouped_idx_ptr,
    N, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NO_K_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)

    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :] * stride_ti

    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]

    iters = tl.cdiv(K, BLOCK_K)
    for i in range(0, iters):
        if NO_K_MASK or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])

        else:
            K_mask = (i * BLOCK_K + K_blk) < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=mask)
        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti
