import torch
from torch import nn
from typing import Union

from .parallel_experts import ParallelExperts, flatten_sort_count


class ReluSquared(nn.Module):
    """ReLU² activation: relu(x)^2"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


# Registry of non-gated activations (use 2 weight matrices: w2(act(w1(x))))
NONGATED_ACTIVATIONS = {
    "relu": nn.ReLU,
    "relu2": ReluSquared,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}

# Registry of gated activations (use gated structure: w2(act(gate) * h))
GATED_ACTIVATIONS = {
    "swiglu": nn.SiLU,  # SwiGLU: swish(gate) * h
    "geglu": nn.GELU,   # GeGLU: gelu(gate) * h
    "reglu": nn.ReLU,   # ReGLU: relu(gate) * h
}


def get_activation(activation: Union[str, nn.Module, None]) -> tuple[nn.Module, bool]:
    """
    Get activation module and whether it's gated.

    Args:
        activation: Either a string name (e.g., "swiglu", "relu2") or an nn.Module

    Returns:
        Tuple of (activation_module, is_gated)
    """
    if activation is None:
        return nn.Identity(), False

    if isinstance(activation, nn.Module):
        return activation, False

    if isinstance(activation, str):
        activation_lower = activation.lower()

        if activation_lower in GATED_ACTIVATIONS:
            return GATED_ACTIVATIONS[activation_lower](), True
        elif activation_lower in NONGATED_ACTIVATIONS:
            return NONGATED_ACTIVATIONS[activation_lower](), False
        else:
            available = list(GATED_ACTIVATIONS.keys()) + list(NONGATED_ACTIVATIONS.keys())
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {available}"
            )

    raise TypeError(f"activation must be str or nn.Module, got {type(activation)}")


class MLP(nn.Module):
    """
    Expert MLP that supports both gated and non-gated activations.

    Gated activations (swiglu, geglu, reglu):
        w2(act(gate) * h) where [h, gate] = w1(x)
        Uses 3 effective weight matrices (first layer outputs 2x hidden_size)

    Non-gated activations (relu, relu2, gelu, silu, tanh):
        w2(act(w1(x)))
        Uses 2 weight matrices

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension (intermediate size)
        num_experts: Number of experts with parameters
        top_k: Number of experts to route to per token
        bias: Whether to use bias in linear layers
        activation: Activation name string (default: "swiglu") or nn.Module
            Gated: "swiglu", "geglu", "reglu"
            Non-gated: "relu", "relu2", "gelu", "silu", "tanh"
        num_virtual_experts: Number of virtual experts that perform identity operation.
            Virtual experts have indices [num_experts, num_experts + num_virtual_experts).
            They participate in routing but perform no computation (identity function).
            Useful for load balancing while allowing some tokens to skip MLP computation.

    Example:
        >>> mlp = MLP(768, 3072, num_experts=8, top_k=2)  # default swiglu
        >>> mlp = MLP(768, 3072, num_experts=8, top_k=2, activation="relu2")
        >>> mlp = MLP(768, 3072, num_experts=8, top_k=2, num_virtual_experts=2)  # 8 real + 2 virtual
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias: bool = False,
        activation: Union[str, nn.Module] = "swiglu",
        num_virtual_experts: int = 0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_virtual_experts = num_virtual_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.top_k = min(top_k, num_experts + num_virtual_experts)

        # Resolve activation and determine if gated
        if isinstance(activation, str):
            act_module, is_gated = get_activation(activation)
            self.activation = act_module
            self._activation_name = activation
            self._is_gated = is_gated
        else:
            self.activation = activation
            self._activation_name = type(activation).__name__
            self._is_gated = False

        # Create expert layers based on gating
        if self._is_gated:
            # Gated: first layer outputs 2x hidden_size
            self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size, bias=bias)
        else:
            # Non-gated: standard hidden_size
            self.experts = ParallelExperts(num_experts, input_size, hidden_size, bias=bias)

        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias=bias)

    @property
    def total_num_experts(self) -> int:
        """Total number of experts including virtual experts (for router configuration)."""
        return self.num_experts + self.num_virtual_experts

    def extra_repr(self):
        gated_str = "gated" if self._is_gated else "nongated"
        virtual_str = f', virtual={self.num_virtual_experts}' if self.num_virtual_experts > 0 else ''
        return f'k={self.top_k}, activation={self._activation_name}, type={gated_str}{virtual_str}'

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])

        # Handle virtual experts (identity operation)
        if self.num_virtual_experts > 0:
            # Identify virtual expert assignments (indices >= num_experts)
            virtual_mask = expert_idxs >= self.num_experts  # (num_tokens, top_k)

            # Sum of routing weights to virtual experts per token
            virtual_weights = (expert_p * virtual_mask).sum(dim=-1, keepdim=True)  # (num_tokens, 1)

            # Virtual contribution: identity weighted by routing weights
            y_virtual = x * virtual_weights  # (num_tokens, input_size)

            # For real expert processing: remap virtual indices to 0 and zero their weights
            # This ensures the kernel doesn't access out-of-bounds expert weights
            real_expert_idxs = expert_idxs.clone()
            real_expert_idxs[virtual_mask] = 0  # Map to expert 0 (contribution will be zeroed)

            real_expert_p = expert_p.clone()
            real_expert_p[virtual_mask] = 0  # Zero weight for virtual assignments
        else:
            real_expert_idxs = expert_idxs
            real_expert_p = expert_p
            y_virtual = 0

        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(real_expert_idxs, num_experts=self.num_experts)

        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_out=True
        )

        if self._is_gated:
            h, gates = h.chunk(2, dim=-1)
            h = self.activation(gates) * h
        else:
            h = self.activation(h)

        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            gates=real_expert_p,
        )

        # Add virtual expert contribution (identity)
        y = y + y_virtual

        y = y.view(*x_shape[:-1], y.size(-1))
        return y


class EmbeddingMLP(nn.Module):
    """
    Embedding-gated MLP for MoE: w2(act(w1(x)) * embd(token_idx))

    Replaces the gate projection with an embedding lookup per token.
    Each expert has its own embedding table. The gate is retrieved by
    token vocabulary index rather than computed from the input.

    Uses 2 weight matrices + 1 embedding per expert.
    Only supports non-gated activations (relu, relu2, gelu, silu, tanh).

    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension (intermediate size)
        num_experts: Number of experts with parameters
        top_k: Number of experts to route to per token
        vocab_size: Vocabulary size for the embedding
        bias: Whether to use bias in linear layers
        activation: Activation name string (default: "relu2") or nn.Module
            Supported: "relu", "relu2", "gelu", "silu", "tanh"
        num_virtual_experts: Number of virtual experts that perform identity operation.
            Virtual experts have indices [num_experts, num_experts + num_virtual_experts).
            They participate in routing but perform no computation (identity function).

    Example:
        >>> mlp = EmbeddingMLP(768, 3072, num_experts=8, top_k=2, vocab_size=32000)
        >>> y = mlp(x, expert_weights, expert_idxs, token_idxs)
        >>> mlp = EmbeddingMLP(768, 3072, num_experts=8, top_k=2, vocab_size=32000, num_virtual_experts=2)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        vocab_size: int,
        bias: bool = False,
        activation: Union[str, nn.Module] = "relu2",
        num_virtual_experts: int = 0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_virtual_experts = num_virtual_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.top_k = min(top_k, num_experts + num_virtual_experts)

        # Resolve activation (only non-gated supported)
        if isinstance(activation, str):
            act_module, is_gated = get_activation(activation)
            if is_gated:
                raise ValueError(
                    f"'{activation}' is a gated activation. "
                    f"EmbeddingMLP only supports non-gated activations: {list(NONGATED_ACTIVATIONS.keys())}"
                )
            self.activation = act_module
            self._activation_name = activation
        else:
            self.activation = activation
            self._activation_name = type(activation).__name__

        # w1: up projection, w2: down projection
        self.experts = ParallelExperts(num_experts, input_size, hidden_size, bias=bias)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias=bias)

        # Embedding per expert: (num_experts, vocab_size, hidden_size)
        self.embedding = nn.Parameter(torch.empty(num_experts, vocab_size, hidden_size))
        self.reset_embedding_parameters()

    def reset_embedding_parameters(self):
        nn.init.normal_(self.embedding, std=0.02)

    @property
    def total_num_experts(self) -> int:
        """Total number of experts including virtual experts (for router configuration)."""
        return self.num_experts + self.num_virtual_experts

    def extra_repr(self):
        virtual_str = f', virtual={self.num_virtual_experts}' if self.num_virtual_experts > 0 else ''
        return (
            f'k={self.top_k}, activation={self._activation_name}, '
            f'vocab_size={self.vocab_size}{virtual_str}'
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_p: torch.Tensor,
        expert_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
    ):
        """
        Args:
            x: Input tensor (batch, seq, input_size) or (batch*seq, input_size)
            expert_p: Expert weights (batch*seq, top_k)
            expert_idxs: Expert indices (batch*seq, top_k)
            token_idxs: Vocabulary indices (batch*seq,)

        Returns:
            Output tensor with same shape as input
        """
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])

        # Handle virtual experts (identity operation)
        if self.num_virtual_experts > 0:
            # Identify virtual expert assignments (indices >= num_experts)
            virtual_mask = expert_idxs >= self.num_experts  # (num_tokens, top_k)

            # Sum of routing weights to virtual experts per token
            virtual_weights = (expert_p * virtual_mask).sum(dim=-1, keepdim=True)  # (num_tokens, 1)

            # Virtual contribution: identity weighted by routing weights
            y_virtual = x * virtual_weights  # (num_tokens, input_size)

            # For real expert processing: remap virtual indices to 0 and zero their weights
            real_expert_idxs = expert_idxs.clone()
            real_expert_idxs[virtual_mask] = 0

            real_expert_p = expert_p.clone()
            real_expert_p[virtual_mask] = 0
        else:
            real_expert_idxs = expert_idxs
            real_expert_p = expert_p
            y_virtual = 0

        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(real_expert_idxs, num_experts=self.num_experts)

        # h = w1(x)
        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_out=True
        )

        # Apply activation: h = act(h)
        h = self.activation(h)

        # Lookup embedding gate for each (token, expert) pair
        # sorted_scattered_idxs: maps sorted position -> flattened (token * top_k) position
        # Original token index = sorted_scattered_idxs // top_k
        original_token_idxs = sorted_scattered_idxs // self.top_k
        token_vocab_idxs = token_idxs[original_token_idxs]

        # Get embedding: embedding[expert_idx, vocab_idx, :]
        emb_gate = self.embedding[sorted_expert_idxs, token_vocab_idxs]

        # Gate: h = h * embd
        h = h * emb_gate

        # y = w2(h)
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            gates=real_expert_p,
        )

        # Add virtual expert contribution (identity)
        y = y + y_virtual

        y = y.view(*x_shape[:-1], y.size(-1))
        return y
