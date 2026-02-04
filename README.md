# scattermoe
Triton-based implementation of Sparse Mixture-of-Experts (SMoE) on GPUs.
ScatterMoE builds upon existing implementations, and overcoming some of the limitations to improve inference, training speed, and memory footprint. 
This implementation achieves this by avoiding padding and making excessive copies of the input.
We also fuse expert linear transforms and reordering operations with `ParallelLinear`, a module that can be used to extend the concept of SMoEs.

This implementation is lightweight (~700 lines).
It will work within an FSDP or pipeline parallel framework, but does not include any additional multi-node training infrastructure code.
You can find the report [here](https://arxiv.org/abs/2403.08245)

## Installation
```sh
# Check all is working well.
PYTHONPATH=. pytest tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```

## Usage
```python
from scattermoe.mlp import MLP

# Initialise module...
mlp = MLP(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
)

# Calling module...
Y = mlp(
    X,         # input tensor
    k_weights, # top-k weights from router
    k_idxs     # top-k indices from router
)
```

## Additional Features

### Activation Types

The MLP supports both gated and non-gated activations via string names:

**Gated activations** (3 weight matrices: `w2(act(gate) * h)` where `[h, gate] = w1(x)`):
- `"swiglu"` (default), `"geglu"`, `"reglu"`

**Non-gated activations** (2 weight matrices: `w2(act(w1(x)))`):
- `"relu"`, `"relu2"`, `"gelu"`, `"silu"`, `"tanh"`

```python
mlp = MLP(768, 3072, num_experts=8, top_k=2, activation="swiglu")  # default
mlp = MLP(768, 3072, num_experts=8, top_k=2, activation="relu2")   # ReLU squared
mlp = MLP(768, 3072, num_experts=8, top_k=2, activation="geglu")   # GeGLU gated
```

You can also pass an `nn.Module` instance for custom activations.

### Virtual Experts

Virtual experts participate in routing but perform an identity operation instead of MLP computation. This allows tokens to "skip" the MLP while still being part of the routing distribution, useful for load balancing or allowing the model to learn when MLP computation is unnecessary.

```python
# 8 real experts + 2 virtual experts = 10 total for routing
mlp = MLP(768, 3072, num_experts=8, top_k=2, num_virtual_experts=2)

# Router should output indices in [0, mlp.total_num_experts)
# Indices [0, 7] route to real experts, [8, 9] route to virtual (identity)
```

### EmbeddingMLP

`EmbeddingMLP` replaces the gate projection with per-expert embedding lookups based on token vocabulary indices: `w2(act(w1(x)) * embd(token_idx))`.

```python
from scattermoe.mlp import EmbeddingMLP

mlp = EmbeddingMLP(
    input_size=768,
    hidden_size=3072,
    num_experts=8,
    top_k=2,
    vocab_size=32000,
    activation="relu2",  # only non-gated activations supported
)

Y = mlp(X, k_weights, k_idxs, token_idxs)  # token_idxs: vocabulary indices
```

`EmbeddingMLP` also supports `num_virtual_experts`.

### Quantization-Aware Training (QAT)

ScatterMoE supports quantization-aware training for expert weights, allowing models to learn to adapt to quantization constraints during training. This uses fake quantization with Straight-Through Estimator (STE) - forward pass simulates quantization noise while gradients flow through unchanged to update full-precision master weights.

**Available QAT modes:**

| Mode | Parameter | Description |
|------|-----------|-------------|
| FP8 | `qat="fp8"` | FP8 E4M3 fake quantization with per-row scaling |
| INT4 | `qat="int4"` | INT4 fake quantization with group-wise scaling |

```python
# FP8 QAT
mlp = MLP(768, 3072, num_experts=8, top_k=2, qat="fp8")

# INT4 QAT with default group size (32)
mlp = MLP(768, 3072, num_experts=8, top_k=2, qat="int4")

# INT4 QAT with custom group size
mlp = MLP(768, 3072, num_experts=8, top_k=2, qat="int4", qat_group_size=64)
```

**How it works:**
- **Forward pass**: Weights are quantized then immediately dequantized (fake quantization), introducing quantization noise that the model learns to handle
- **Backward pass**: STE passes gradients through the quantization operation unchanged, allowing full-precision weight updates
- **Result**: Model learns weight distributions that are robust to quantization, enabling better post-training quantization or direct deployment with quantized weights

### FP8 Storage Mode

For inference or when you want to store weights in FP8 format (separate from QAT), use `fp8=True`:

```python
mlp = MLP(768, 3072, num_experts=8, top_k=2, fp8=True)

# After loading/initializing weights, quantize them:
mlp.quantize_weights()

# Forward pass uses FP8 Triton kernel
Y = mlp(X, k_weights, k_idxs)
```

This mode stores weights in FP8 buffers with per-row scales and uses a custom FP8 Triton kernel for the forward pass, while backward uses full-precision weights for accurate gradients.

**QAT vs FP8 Storage:**
- `qat="fp8"`: Keeps full-precision weights, simulates FP8 during forward. Use during training.
- `fp8=True`: Actually stores FP8 weights in buffers. Requires `quantize_weights()` call. Better for inference.

## Bibtex
If you use ScatterMoE in your project, cite us!
```bibtex
@article{tan2024scattered,
  title={Scattered Mixture-of-Experts Implementation},
  author={Tan, Shawn and Shen, Yikang and Panda, Rameswar and Courville, Aaron},
  journal={arXiv preprint arXiv:2403.08245},
  year={2024}
}
```

Enjoy!
----

###  Version 0.3.0

- Refactored away padded indices
- Allow bias
- Inject MoE implementation into the following implementations:
  - `transformers.models.gpt_oss.modeling_gpt_oss` 
  - `transformers.models.granitemoehybrid.modeling_granitemoehybrid`
Just do this:
```sh
git clone git@github.com:shawntan/scattermoe.git
cd scattermoe
pip install -e .
```
```python
import transformers
import scattermoe.utils.replace_moe # put this line after wherever you import transformers
```

###  Version 0.2.0

- Made compileable.
