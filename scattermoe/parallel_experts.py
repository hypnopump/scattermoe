import torch
import torch.nn as nn
from . import kernels
from typing import Optional, Tuple

@torch.library.custom_op("scattermoe::bincount", mutates_args={})
def compileable_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
        return x.bincount(minlength=minlength)

@compileable_bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, dtype=torch.long, device=x.device)

@torch.compile
def flatten_sort_count(expert_idxs: torch.Tensor, num_experts: int):
    with torch.no_grad():
        flattened_expert_idxs = expert_idxs.flatten()
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)
        expert_counts = compileable_bincount(flattened_expert_idxs, minlength=num_experts)
        expert_offsets = expert_counts.cumsum(-1)
        return sorted_expert_idxs, sorted_scattered_idxs, expert_offsets



class ParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, expert_weights: torch.Tensor, k: int,
        sorted_expert_idxs: torch.Tensor, sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_biases: Optional[torch.Tensor]=None,
        gates: Optional[torch.Tensor]=None,
        grouped_in: bool =False, grouped_out: bool=False,
    ):
        with torch.device(x.device):
            output = kernels.ops.scatter2scatter(
                X=x, W=expert_weights,
                b=expert_biases, k=k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=grouped_in, y_grouped=grouped_out
            )
            if gates is not None:
                output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
                output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
            else:
                output_expanded = None

            ctx.save_for_backward(
                x, expert_weights,
                expert_biases,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded
            )
            ctx.grouped_in = grouped_in
            ctx.grouped_out = grouped_out
            ctx.k = k
        return output
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        with torch.device(grad_out.device):
            (x, expert_weights, expert_biases,
             sorted_expert_idxs,
             sorted_scattered_idxs,
             expert_offsets,
             gates, output_expanded) = ctx.saved_tensors
            k = ctx.k
            grouped_in = ctx.grouped_in
            grouped_out = ctx.grouped_out
            # print("backward")

            if gates is not None:
                # calculate gates gradient
                # d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
                d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                grouped_grad_out = output_expanded.flatten(0, 1) # reuse expanded buffer later
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None

            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = kernels.ops.group(grad_out, sorted_scattered_idxs,
                                                     fan_out=gate_fan, coeff=gates_flat,
                                                     out=grouped_grad_out)
            if grouped_in:
                grouped_x = x
                d_expanded_input = None
            else:
                grouped_x = kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
                d_expanded_input = grouped_x

            d_weights, d_biases = kernels.ops.group_bwd_W(
                DY=grouped_grad_out, X=grouped_x,
                expert_offsets=expert_offsets,
                E=expert_weights.size(0),
                has_bias=expert_biases is not None
            )


            d_expanded_input = kernels.ops.scatter2scatter(
                X=grouped_grad_out, x_grouped=True,
                W=expert_weights.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
                y_grouped=grouped_in,
                out=d_expanded_input # Reuse grouped_x buffer
            )

            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)
        # print("backward end.")
        return (
            # x, expert_weights,
            d_input, d_weights,
            # k, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            None, None, None, None, 
            # bias, gates
            d_biases, d_gates,
            # grouped_in, grouped_out,
            None, None
        )

def parallel_linear(inputs, expert_weights, k,
                    sorted_expert_idxs, sorted_scattered_idxs,
                    expert_offsets,
                    expert_biases=None,
                    gates=None, grouped_in=False, grouped_out=False):
    results = ParallelLinear.apply(inputs, expert_weights, k,
                                   sorted_expert_idxs, sorted_scattered_idxs,
                                   expert_offsets,
                                   expert_biases,
                                   gates, grouped_in, grouped_out)
    return results


class ParallelLinearFP8(torch.autograd.Function):
    """
    FP8 version of ParallelLinear with rowwise quantization.

    Forward pass uses FP8 weights and dynamically quantizes activations.
    Backward pass uses full precision for gradient computation.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights_fp8: torch.Tensor,
        expert_weights_scale: torch.Tensor,
        expert_weights_fp32: torch.Tensor,  # Full precision for backward
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_biases: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ):
        with torch.device(x.device):
            # Quantize input activations dynamically
            x_fp8, x_scale = kernels.ops.quantize_fp8_rowwise(x)

            output = kernels.ops.scatter2scatter_fp8(
                X_fp8=x_fp8, X_scale=x_scale,
                W_fp8=expert_weights_fp8, W_scale=expert_weights_scale,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=k,
                b=expert_biases,
                x_grouped=grouped_in, y_grouped=grouped_out,
                output_dtype=x.dtype,
            )

            if gates is not None:
                output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
                output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
            else:
                output_expanded = None

            ctx.save_for_backward(
                x, expert_weights_fp32,
                expert_biases,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                gates,
                output_expanded
            )
            ctx.grouped_in = grouped_in
            ctx.grouped_out = grouped_out
            ctx.k = k
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Backward uses full precision weights for accurate gradients
        with torch.device(grad_out.device):
            (x, expert_weights_fp32, expert_biases,
             sorted_expert_idxs,
             sorted_scattered_idxs,
             expert_offsets,
             gates, output_expanded) = ctx.saved_tensors
            k = ctx.k
            grouped_in = ctx.grouped_in
            grouped_out = ctx.grouped_out

            if gates is not None:
                d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                grouped_grad_out = output_expanded.flatten(0, 1)
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None

            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = kernels.ops.group(grad_out, sorted_scattered_idxs,
                                                     fan_out=gate_fan, coeff=gates_flat,
                                                     out=grouped_grad_out)
            if grouped_in:
                grouped_x = x
                d_expanded_input = None
            else:
                grouped_x = kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
                d_expanded_input = grouped_x

            d_weights, d_biases = kernels.ops.group_bwd_W(
                DY=grouped_grad_out, X=grouped_x,
                expert_offsets=expert_offsets,
                E=expert_weights_fp32.size(0),
                has_bias=expert_biases is not None
            )

            d_expanded_input = kernels.ops.scatter2scatter(
                X=grouped_grad_out, x_grouped=True,
                W=expert_weights_fp32.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
                y_grouped=grouped_in,
                out=d_expanded_input
            )

            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        return (
            d_input, None, None, d_weights,  # x, fp8_weights, scale, fp32_weights
            None, None, None, None,  # k, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets
            d_biases, d_gates,  # bias, gates
            None, None  # grouped_in, grouped_out
        )


def parallel_linear_fp8(
    inputs, expert_weights_fp8, expert_weights_scale, expert_weights_fp32, k,
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
    expert_biases=None, gates=None, grouped_in=False, grouped_out=False
):
    """
    FP8 parallel linear forward with full-precision backward.
    """
    results = ParallelLinearFP8.apply(
        inputs, expert_weights_fp8, expert_weights_scale, expert_weights_fp32, k,
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
        expert_biases, gates, grouped_in, grouped_out
    )
    return results

class ParallelExperts(nn.Module):
    """
    Parallel expert linear layer for Mixture of Experts.

    Args:
        num_experts: Number of experts
        input_size: Input dimension
        output_size: Output dimension
        bias: Whether to use bias
        fp8: Use FP8 storage mode (quantize weights to FP8 buffer)
        qat: QAT mode - "fp8" for FP8 QAT, "int4" for INT4 QAT, None for disabled
        qat_group_size: Group size for INT4 QAT quantization (default: 32)
    """
    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        fp8: bool = False,
        qat: Optional[str] = None,
        qat_group_size: int = 32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_size))
        else:
            self.bias = None

        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.fp8 = fp8
        self.qat = qat
        self.qat_group_size = qat_group_size

        # Validate QAT mode
        if qat is not None and qat not in ("fp8", "int4"):
            raise ValueError(f"qat must be 'fp8', 'int4', or None, got '{qat}'")

        # FP8 buffers (registered but not initialized until quantize_weights is called)
        if fp8:
            # Weight shape is (E, output_size, input_size), permuted to (E, input_size, output_size)
            # So FP8 weight will be (E, input_size, output_size) and scale is (E, output_size)
            self.register_buffer('weight_fp8', None)
            self.register_buffer('weight_scale', None)

        self.reset_parameters()

    def extra_repr(self):
        fp8_str = ', fp8=True' if self.fp8 else ''
        qat_str = f', qat={self.qat!r}' if self.qat else ''
        return 'num_experts={}, input_size={}, output_size={}{}{}'.format(
            self.num_experts, self.input_size, self.output_size, fp8_str, qat_str)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weights(self) -> None:
        """
        Quantize weights to FP8 format (for fp8=True mode, not QAT).
        Call this after loading weights or during training to update FP8 cache.
        """
        if not self.fp8:
            raise RuntimeError("FP8 mode not enabled for this module")

        # Permute weight from (E, output_size, input_size) to (E, input_size, output_size)
        weight_permuted = self.weight.permute(0, 2, 1).contiguous()

        # Quantize along K dimension for each (e, n) slice
        # This means each output neuron has its own scale
        E, K, N = weight_permuted.shape
        weight_permuted_for_scale = weight_permuted.permute(0, 2, 1).contiguous()  # (E, N, K)
        weight_flat_n = weight_permuted_for_scale.reshape(E * N, K)
        weight_fp8_n, weight_scale_n = kernels.ops.quantize_fp8_rowwise(weight_flat_n)
        # Reshape: weight_fp8 should be (E, K, N) for kernel
        self.weight_fp8 = weight_fp8_n.reshape(E, N, K).permute(0, 2, 1).contiguous()
        # Scale is (E, N)
        self.weight_scale = weight_scale_n.reshape(E, N)

    def _get_weight_for_forward(self) -> torch.Tensor:
        """
        Get the weight tensor for forward pass, applying QAT fake quantization if enabled.
        """
        weight_permuted = self.weight.permute(0, 2, 1)  # (E, input_size, output_size)

        if self.qat == "fp8":
            # Apply fake FP8 quantization with STE
            # Quantize along the output dimension (last dim)
            original_shape = weight_permuted.shape
            weight_flat = weight_permuted.reshape(-1, original_shape[-1])
            weight_fake_quant = kernels.ops.fake_quantize_fp8_rowwise(weight_flat)
            return weight_fake_quant.reshape(original_shape)

        elif self.qat == "int4":
            # Apply fake INT4 quantization with STE
            original_shape = weight_permuted.shape
            weight_flat = weight_permuted.reshape(-1, original_shape[-1])
            weight_fake_quant = kernels.ops.fake_quantize_int4_rowwise(weight_flat, self.qat_group_size)
            return weight_fake_quant.reshape(original_shape)

        else:
            return weight_permuted

    def forward(self, inputs, k, sorted_expert_idxs, sorted_scattered_idxs,
                expert_offsets,
                gates=None, grouped_in=False, grouped_out=False):

        if self.fp8 and self.weight_fp8 is not None:
            # FP8 storage mode: use FP8 kernel
            results = parallel_linear_fp8(
                inputs, self.weight_fp8, self.weight_scale, self.weight.permute(0, 2, 1), k,
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
                expert_biases=self.bias,
                gates=gates, grouped_in=grouped_in, grouped_out=grouped_out
            )
        else:
            # Standard mode or QAT mode
            # QAT applies fake quantization to weights (with STE for gradients)
            weight = self._get_weight_for_forward()
            results = parallel_linear(
                inputs, weight, k,
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
                expert_biases=self.bias,
                gates=gates, grouped_in=grouped_in, grouped_out=grouped_out
            )
        return results
