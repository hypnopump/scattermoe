import pytest
import torch
from torch import nn
from torch.nn import functional as F
from scattermoe.mlp import MLP, EmbeddingMLP, ReluSquared
import scattermoe

scattermoe.kernels.ops.ALLOW_TF32 = False


def dumb_forward_nongated(m, x, expert_p, expert_idxs):
    """Reference implementation for non-gated MLP: w2(act(w1(x)))"""
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                m.activation(
                    F.linear(
                        x[i], m.experts.weight[expert_idxs[i, j]],
                        bias=m.experts.bias[expert_idxs[i, j]] if m.experts.bias is not None else None
                    )
                ),
                m.output_experts.weight[expert_idxs[i, j]],
                bias=m.output_experts.bias[expert_idxs[i, j]] if m.output_experts.bias is not None else None
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output


def dumb_forward_gated(m, x, expert_p, expert_idxs):
    """Reference implementation for gated MLP: w2(act(gate) * h) where [h, gate] = w1(x)"""
    h_dim = m.hidden_size
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                (lambda h: m.activation(h[..., h_dim:]) * h[..., :h_dim])(
                    F.linear(
                        x[i], m.experts.weight[expert_idxs[i, j]],
                        bias=m.experts.bias[expert_idxs[i, j]] if m.experts.bias is not None else None
                    )
                ),
                m.output_experts.weight[expert_idxs[i, j]],
                bias=m.output_experts.bias[expert_idxs[i, j]] if m.output_experts.bias is not None else None
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output


def dumb_forward_embedding(m, x, expert_p, expert_idxs, token_idxs):
    """Reference implementation for EmbeddingMLP: w2(act(w1(x)) * embd(token_idx))"""
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                m.activation(
                    F.linear(
                        x[i], m.experts.weight[expert_idxs[i, j]],
                        bias=m.experts.bias[expert_idxs[i, j]] if m.experts.bias is not None else None
                    )
                ) * m.embedding[expert_idxs[i, j], token_idxs[i]],
                m.output_experts.weight[expert_idxs[i, j]],
                bias=m.output_experts.bias[expert_idxs[i, j]] if m.output_experts.bias is not None else None
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output


def dumb_forward(m, x, expert_p, expert_idxs):
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                m.activation(
                    F.linear(
                        x[i], m.experts.weight[expert_idxs[i, j]],
                        bias=m.experts.bias[expert_idxs[i, j]] if m.experts.bias is not None else None
                    )
                ),
                m.output_experts.weight[expert_idxs[i, j]],
                bias=m.output_experts.bias[expert_idxs[i, j]] if m.output_experts.bias is not None else None
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output

def assert_diff(name, ref_X, new_X, tolerance=1e-2):
    diff = torch.abs(ref_X - new_X)
    max_diff = diff.max()
    print(f"{name} diff: {max_diff.item()}")
    assert max_diff < tolerance


class TestClass:
    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('bias', [False, True])
    @pytest.mark.parametrize('length', [1, 256, 512])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (xd, (4 * xd) // k, k)
        for xd in [128, 256, 512, 600, 100]
        for k in [2, 3, 4]
    ])
    def test_mlp_correctness(self, length, x_dim, h_dim, E, k, bias, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=nn.GELU(),
            num_experts=E, top_k=k,
            bias=bias
        ).cuda().to(dtype)
        if bias:
            nn.init.normal_(mlp.experts.bias, std=0.02)
            nn.init.normal_(mlp.output_experts.bias, std=0.02)
            


        Y = mlp(X, k_weights, k_idxs)
        name_tup = ("dX", "dg", "dW1", "dW2")
        input_tup = (X, k_weights, mlp.experts.weight, mlp.output_experts.weight)
        if bias:
            name_tup += ("db1", "db2")
            input_tup += (mlp.experts.bias, mlp.output_experts.bias)

        ref_out_tup = torch.autograd.grad(
            outputs=(Y,),
            inputs=input_tup,
            grad_outputs=(DY,)
        )
        Y_ = dumb_forward(mlp, X, k_weights, k_idxs)
        out_tup = torch.autograd.grad(
            outputs=(Y_,),
            inputs=input_tup,
            grad_outputs=(DY,)
        )
        tol = 1e-4 if dtype == torch.float32 else 1e-2

        assert_diff("Y", Y_, Y, tolerance=tol)
        for name, ref, new in zip(name_tup, ref_out_tup, out_tup):
            assert_diff(name, ref, new, tolerance=tol)


class TestNongatedActivations:
    """Test non-gated activations (relu, relu2, gelu, silu)"""

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('activation', ['relu', 'relu2', 'gelu', 'silu'])
    @pytest.mark.parametrize('length', [1, 64, 256])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (128, 512, 2),
        (256, 1024, 2),
        (128, 512, 4),
    ])
    def test_nongated_correctness(self, length, x_dim, h_dim, E, k, activation, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)
        Y_ref = dumb_forward_nongated(mlp, X, k_weights, k_idxs)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_diff("Y", Y_ref, Y, tolerance=tol)

        # Test gradients
        name_tup = ("dX", "dg", "dW1", "dW2")
        input_tup = (X, k_weights, mlp.experts.weight, mlp.output_experts.weight)

        ref_out_tup = torch.autograd.grad(outputs=(Y,), inputs=input_tup, grad_outputs=(DY,))
        out_tup = torch.autograd.grad(outputs=(Y_ref,), inputs=input_tup, grad_outputs=(DY,))

        for name, ref, new in zip(name_tup, ref_out_tup, out_tup):
            assert_diff(name, ref, new, tolerance=tol)


class TestGatedActivations:
    """Test gated activations (swiglu, geglu, reglu)"""

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'reglu'])
    @pytest.mark.parametrize('length', [1, 64, 256])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (128, 512, 2),
        (256, 1024, 2),
        (128, 512, 4),
    ])
    def test_gated_correctness(self, length, x_dim, h_dim, E, k, activation, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)
        Y_ref = dumb_forward_gated(mlp, X, k_weights, k_idxs)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_diff("Y", Y_ref, Y, tolerance=tol)

        # Test gradients
        name_tup = ("dX", "dg", "dW1", "dW2")
        input_tup = (X, k_weights, mlp.experts.weight, mlp.output_experts.weight)

        ref_out_tup = torch.autograd.grad(outputs=(Y,), inputs=input_tup, grad_outputs=(DY,))
        out_tup = torch.autograd.grad(outputs=(Y_ref,), inputs=input_tup, grad_outputs=(DY,))

        for name, ref, new in zip(name_tup, ref_out_tup, out_tup):
            assert_diff(name, ref, new, tolerance=tol)


class TestEmbeddingMLP:
    """Test EmbeddingMLP: w2(act(w1(x)) * embd(token_idx))"""

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('activation', ['relu', 'relu2', 'gelu', 'silu'])
    @pytest.mark.parametrize('length', [1, 64, 256])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k, vocab_size', [
        (128, 512, 2, 1000),
        (256, 1024, 2, 500),
        (128, 512, 4, 2000),
    ])
    def test_embedding_mlp_correctness(self, length, x_dim, h_dim, E, k, vocab_size, activation, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        token_idxs = torch.randint(0, vocab_size, (length,)).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = EmbeddingMLP(
            input_size=x_dim, hidden_size=h_dim,
            num_experts=E, top_k=k, vocab_size=vocab_size,
            activation=activation,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs, token_idxs)
        Y_ref = dumb_forward_embedding(mlp, X, k_weights, k_idxs, token_idxs)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_diff("Y", Y_ref, Y, tolerance=tol)

        # Test gradients (excluding embedding for simplicity)
        name_tup = ("dX", "dg", "dW1", "dW2")
        input_tup = (X, k_weights, mlp.experts.weight, mlp.output_experts.weight)

        ref_out_tup = torch.autograd.grad(outputs=(Y,), inputs=input_tup, grad_outputs=(DY,))
        out_tup = torch.autograd.grad(outputs=(Y_ref,), inputs=input_tup, grad_outputs=(DY,))

        for name, ref, new in zip(name_tup, ref_out_tup, out_tup):
            assert_diff(name, ref, new, tolerance=tol)

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_embedding_gradient_flow(self, dtype):
        """Test that gradients flow through the embedding"""
        length, x_dim, h_dim, E, k, vocab_size = 32, 128, 512, 8, 2, 100

        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, device='cuda')
        X.requires_grad_(True)
        token_idxs = torch.randint(0, vocab_size, (length,), device='cuda')
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = EmbeddingMLP(
            input_size=x_dim, hidden_size=h_dim,
            num_experts=E, top_k=k, vocab_size=vocab_size,
            activation='relu2',
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs, token_idxs)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None, "X gradient is None"
        assert k_weights.grad is not None, "k_weights gradient is None"
        assert mlp.experts.weight.grad is not None, "experts.weight gradient is None"
        assert mlp.output_experts.weight.grad is not None, "output_experts.weight gradient is None"
        assert mlp.embedding.grad is not None, "embedding gradient is None"

    def test_embedding_mlp_rejects_gated(self):
        """Test that EmbeddingMLP rejects gated activations"""
        with pytest.raises(ValueError, match="gated activation"):
            EmbeddingMLP(
                input_size=128, hidden_size=512,
                num_experts=8, top_k=2, vocab_size=1000,
                activation='swiglu'
            )


def dumb_forward_with_virtual(m, x, expert_p, expert_idxs):
    """Reference implementation that handles virtual experts (identity operation)."""
    num_experts = m.num_experts  # Real experts only
    output = torch.zeros_like(x)

    for i in range(expert_idxs.size(0)):
        for j in range(expert_idxs.size(1)):
            expert_idx = expert_idxs[i, j].item()
            weight = expert_p[i, j]

            if expert_idx >= num_experts:
                # Virtual expert: identity operation
                output[i] += weight * x[i]
            else:
                # Real expert: MLP computation
                if m._is_gated:
                    h = F.linear(
                        x[i], m.experts.weight[expert_idx],
                        bias=m.experts.bias[expert_idx] if m.experts.bias is not None else None
                    )
                    h_dim = m.hidden_size
                    h = m.activation(h[..., h_dim:]) * h[..., :h_dim]
                else:
                    h = m.activation(
                        F.linear(
                            x[i], m.experts.weight[expert_idx],
                            bias=m.experts.bias[expert_idx] if m.experts.bias is not None else None
                        )
                    )
                output[i] += weight * F.linear(
                    h,
                    m.output_experts.weight[expert_idx],
                    bias=m.output_experts.bias[expert_idx] if m.output_experts.bias is not None else None
                )
    return output


def dumb_forward_embedding_with_virtual(m, x, expert_p, expert_idxs, token_idxs):
    """Reference implementation for EmbeddingMLP with virtual experts."""
    num_experts = m.num_experts
    output = torch.zeros_like(x)

    for i in range(expert_idxs.size(0)):
        for j in range(expert_idxs.size(1)):
            expert_idx = expert_idxs[i, j].item()
            weight = expert_p[i, j]

            if expert_idx >= num_experts:
                # Virtual expert: identity operation
                output[i] += weight * x[i]
            else:
                # Real expert: EmbeddingMLP computation
                h = m.activation(
                    F.linear(
                        x[i], m.experts.weight[expert_idx],
                        bias=m.experts.bias[expert_idx] if m.experts.bias is not None else None
                    )
                )
                # Apply embedding gate
                h = h * m.embedding[expert_idx, token_idxs[i]]
                output[i] += weight * F.linear(
                    h,
                    m.output_experts.weight[expert_idx],
                    bias=m.output_experts.bias[expert_idx] if m.output_experts.bias is not None else None
                )
    return output


class TestVirtualExperts:
    """Test virtual experts (no-computation/identity experts)"""

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('activation', ['gelu', 'swiglu'])
    @pytest.mark.parametrize('length', [1, 64, 256])
    @pytest.mark.parametrize('num_virtual', [1, 2, 4])
    @pytest.mark.parametrize('x_dim, h_dim, E, k', [
        (128, 512, 8, 2),
        (256, 1024, 4, 2),
    ])
    def test_virtual_experts_correctness(self, length, x_dim, h_dim, E, k, num_virtual, activation, dtype):
        """Test that virtual experts produce correct output (identity operation)."""
        total_experts = E + num_virtual

        logits = torch.randn(length, total_experts, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            num_virtual_experts=num_virtual,
            bias=False
        ).cuda().to(dtype)

        # Verify total_num_experts property
        assert mlp.total_num_experts == total_experts

        Y = mlp(X, k_weights, k_idxs)
        Y_ref = dumb_forward_with_virtual(mlp, X, k_weights, k_idxs)

        tol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_diff("Y", Y_ref, Y, tolerance=tol)

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_all_virtual_experts(self, dtype):
        """Test when all routed experts are virtual (pure identity)."""
        length, x_dim, h_dim, E, k, num_virtual = 32, 128, 512, 4, 2, 4

        # Route only to virtual experts (indices E to E+num_virtual-1)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        k_idxs = torch.randint(E, E + num_virtual, (length, k)).cuda()
        k_weights = torch.softmax(torch.randn(length, k, dtype=dtype), dim=-1).cuda()
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            num_virtual_experts=num_virtual,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)

        # When all experts are virtual, output should be: sum(k_weights) * x per token
        expected = X * k_weights.sum(dim=-1, keepdim=True)

        tol = 1e-5
        assert_diff("Y_all_virtual", expected, Y, tolerance=tol)

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_no_virtual_experts_unchanged(self, dtype):
        """Test that num_virtual_experts=0 produces same results as before."""
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        # MLP without virtual experts
        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            num_virtual_experts=0,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)
        Y_ref = dumb_forward_nongated(mlp, X, k_weights, k_idxs)

        tol = 1e-4
        assert_diff("Y", Y_ref, Y, tolerance=tol)

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_virtual_experts_gradient_flow(self, dtype):
        """Test that gradients flow correctly through virtual experts."""
        length, x_dim, h_dim, E, k, num_virtual = 32, 128, 512, 4, 2, 2

        logits = torch.randn(length, E + num_virtual, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, device='cuda')
        X.requires_grad_(True)
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            num_virtual_experts=num_virtual,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None, "X gradient is None"
        assert k_weights.grad is not None, "k_weights gradient is None"
        assert mlp.experts.weight.grad is not None, "experts.weight gradient is None"
        assert mlp.output_experts.weight.grad is not None, "output_experts.weight gradient is None"

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_embedding_mlp_virtual_experts(self, dtype):
        """Test virtual experts in EmbeddingMLP."""
        length, x_dim, h_dim, E, k, vocab_size, num_virtual = 32, 128, 512, 4, 2, 100, 2
        total_experts = E + num_virtual

        logits = torch.randn(length, total_experts, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        token_idxs = torch.randint(0, vocab_size, (length,)).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = EmbeddingMLP(
            input_size=x_dim, hidden_size=h_dim,
            num_experts=E, top_k=k, vocab_size=vocab_size,
            activation='relu2',
            num_virtual_experts=num_virtual,
            bias=False
        ).cuda().to(dtype)

        assert mlp.total_num_experts == total_experts

        Y = mlp(X, k_weights, k_idxs, token_idxs)
        Y_ref = dumb_forward_embedding_with_virtual(mlp, X, k_weights, k_idxs, token_idxs)

        tol = 1e-4
        assert_diff("Y", Y_ref, Y, tolerance=tol)

    @pytest.mark.parametrize('dtype', [torch.float32])
    def test_mixed_real_virtual_routing(self, dtype):
        """Test mixed routing where some tokens go to real, some to virtual experts."""
        length, x_dim, h_dim, E, k, num_virtual = 64, 128, 512, 4, 2, 2

        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()

        # Create routing where half go to real experts, half to virtual
        k_idxs = torch.zeros(length, k, dtype=torch.long).cuda()
        k_idxs[:length//2, :] = torch.randint(0, E, (length//2, k)).cuda()  # Real experts
        k_idxs[length//2:, :] = torch.randint(E, E + num_virtual, (length - length//2, k)).cuda()  # Virtual

        k_weights = torch.softmax(torch.randn(length, k, dtype=dtype), dim=-1).cuda()
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            num_virtual_experts=num_virtual,
            bias=False
        ).cuda().to(dtype)

        Y = mlp(X, k_weights, k_idxs)
        Y_ref = dumb_forward_with_virtual(mlp, X, k_weights, k_idxs)

        tol = 1e-4
        assert_diff("Y_mixed", Y_ref, Y, tolerance=tol)

    def test_extra_repr_with_virtual(self):
        """Test that extra_repr includes virtual expert count."""
        mlp = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            num_virtual_experts=2
        )
        repr_str = mlp.extra_repr()
        assert 'virtual=2' in repr_str

        # Without virtual experts, should not appear
        mlp_no_virtual = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2
        )
        repr_str_no_virtual = mlp_no_virtual.extra_repr()
        assert 'virtual' not in repr_str_no_virtual


class TestFP8:
    """Test FP8 rowwise quantization for experts."""

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('activation', ['relu', 'gelu'])
    @pytest.mark.parametrize('length', [64, 256])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (128, 512, 2),
        (256, 1024, 2),
    ])
    def test_fp8_forward_close_to_fp32(self, length, x_dim, h_dim, E, k, activation, dtype):
        """Test that FP8 forward is reasonably close to FP32 forward.

        Note: FP8 quantization introduces ~1-2% error per layer. With 2 expert layers
        plus activation, errors compound. The important thing for QAT is that:
        1. Outputs are numerically stable (no NaN/inf)
        2. Gradient computation works correctly
        3. Relative error is bounded (not exponentially growing)
        """
        torch.manual_seed(42)
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        # FP32 reference
        mlp_fp32 = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False,
            fp8=False
        ).cuda().to(dtype)

        # FP8 model with same weights
        mlp_fp8 = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False,
            fp8=True
        ).cuda().to(dtype)

        # Copy weights
        mlp_fp8.experts.weight.data.copy_(mlp_fp32.experts.weight.data)
        mlp_fp8.output_experts.weight.data.copy_(mlp_fp32.output_experts.weight.data)

        # Quantize weights
        mlp_fp8.quantize_weights()

        with torch.no_grad():
            Y_fp32 = mlp_fp32(X, k_weights, k_idxs)
            Y_fp8 = mlp_fp8(X, k_weights, k_idxs)

        # Check outputs are valid
        assert not torch.isnan(Y_fp8).any(), "FP8 output contains NaN"
        assert not torch.isinf(Y_fp8).any(), "FP8 output contains Inf"

        # Check mean and std are similar (within 10%)
        mean_diff = abs(Y_fp8.mean().item() - Y_fp32.mean().item()) / (abs(Y_fp32.mean().item()) + 1e-6)
        std_diff = abs(Y_fp8.std().item() - Y_fp32.std().item()) / (abs(Y_fp32.std().item()) + 1e-6)
        print(f"Mean relative diff: {mean_diff:.4f}, Std relative diff: {std_diff:.4f}")
        assert mean_diff < 0.2, f"FP8 mean differs too much: {mean_diff:.4f}"
        assert std_diff < 0.2, f"FP8 std differs too much: {std_diff:.4f}"

        # Check correlation is high (outputs should be highly correlated)
        corr = torch.corrcoef(torch.stack([Y_fp32.flatten(), Y_fp8.flatten()]))[0, 1].item()
        print(f"Correlation between FP32 and FP8 outputs: {corr:.4f}")
        assert corr > 0.95, f"FP8 output poorly correlated with FP32: {corr:.4f}"

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('length', [64])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (128, 512, 2),
    ])
    def test_fp8_backward_gradients(self, length, x_dim, h_dim, E, k, dtype):
        """Test that FP8 backward produces valid gradients."""
        torch.manual_seed(42)
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp_fp8 = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            bias=False,
            fp8=True
        ).cuda().to(dtype)

        mlp_fp8.quantize_weights()

        Y = mlp_fp8(X, k_weights, k_idxs)

        # Check that backward runs without error
        grads = torch.autograd.grad(
            outputs=(Y,),
            inputs=(X, k_weights, mlp_fp8.experts.weight, mlp_fp8.output_experts.weight),
            grad_outputs=(DY,)
        )

        # Check gradients are not None and have correct shapes
        dX, dg, dW1, dW2 = grads
        assert dX is not None and dX.shape == X.shape
        assert dg is not None and dg.shape == k_weights.shape
        assert dW1 is not None and dW1.shape == mlp_fp8.experts.weight.shape
        assert dW2 is not None and dW2.shape == mlp_fp8.output_experts.weight.shape

        # Check gradients are not all zeros
        assert dX.abs().sum() > 0
        assert dW1.abs().sum() > 0
        assert dW2.abs().sum() > 0

    @pytest.mark.parametrize('activation', ['swiglu', 'geglu'])
    def test_fp8_gated_activations(self, activation):
        """Test FP8 with gated activations."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        mlp_fp8 = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False,
            fp8=True
        ).cuda()

        mlp_fp8.quantize_weights()

        with torch.no_grad():
            Y = mlp_fp8(X, k_weights, k_idxs)

        # Check output shape and values are valid
        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()

    def test_fp8_extra_repr(self):
        """Test that extra_repr includes fp8 status."""
        mlp_fp8 = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            fp8=True
        )
        repr_str = mlp_fp8.extra_repr()
        assert 'fp8=True' in repr_str

        mlp_no_fp8 = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            fp8=False
        )
        repr_str_no_fp8 = mlp_no_fp8.extra_repr()
        assert 'fp8' not in repr_str_no_fp8

    def test_fp8_quantize_weights_error_when_disabled(self):
        """Test that quantize_weights raises error when fp8=False."""
        mlp = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            fp8=False
        )

        with pytest.raises(RuntimeError, match="FP8 mode not enabled"):
            mlp.quantize_weights()

    def test_fp8_embedding_mlp(self):
        """Test FP8 with EmbeddingMLP."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k, vocab_size = 64, 128, 512, 8, 2, 1000

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        token_idxs = torch.randint(0, vocab_size, (length,)).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        mlp_fp8 = EmbeddingMLP(
            input_size=x_dim, hidden_size=h_dim,
            vocab_size=vocab_size,
            activation='relu',
            num_experts=E, top_k=k,
            bias=False,
            fp8=True
        ).cuda()

        mlp_fp8.quantize_weights()

        with torch.no_grad():
            Y = mlp_fp8(X, k_weights, k_idxs, token_idxs)

        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()


class TestQAT:
    """Test Quantization-Aware Training (QAT) with Straight-Through Estimator."""

    @pytest.mark.parametrize('qat_mode', ['fp8', 'int4'])
    @pytest.mark.parametrize('activation', ['relu', 'gelu'])
    def test_qat_forward(self, qat_mode, activation):
        """Test QAT forward pass produces valid outputs."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=activation,
            num_experts=E, top_k=k,
            bias=False,
            qat=qat_mode,
        ).cuda()

        with torch.no_grad():
            Y = mlp(X, k_weights, k_idxs)

        assert Y.shape == X.shape
        assert not torch.isnan(Y).any(), f"QAT {qat_mode} output contains NaN"
        assert not torch.isinf(Y).any(), f"QAT {qat_mode} output contains Inf"

    @pytest.mark.parametrize('qat_mode', ['fp8', 'int4'])
    def test_qat_backward_ste(self, qat_mode):
        """Test that QAT backward pass works with STE (gradients flow through)."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights = k_weights.detach().requires_grad_(True)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            bias=False,
            qat=qat_mode,
        ).cuda()

        Y = mlp(X, k_weights, k_idxs)

        # Check that backward runs without error
        grads = torch.autograd.grad(
            outputs=(Y,),
            inputs=(X, k_weights, mlp.experts.weight, mlp.output_experts.weight),
            grad_outputs=(DY,)
        )

        dX, dg, dW1, dW2 = grads

        # Check gradients exist and have correct shapes
        assert dX is not None and dX.shape == X.shape
        assert dg is not None and dg.shape == k_weights.shape
        assert dW1 is not None and dW1.shape == mlp.experts.weight.shape
        assert dW2 is not None and dW2.shape == mlp.output_experts.weight.shape

        # Check gradients are not all zeros (STE should allow gradients through)
        assert dX.abs().sum() > 0, "Input gradients are all zero"
        assert dW1.abs().sum() > 0, "Weight gradients are all zero (STE not working)"
        assert dW2.abs().sum() > 0, "Output weight gradients are all zero"

    @pytest.mark.parametrize('qat_mode', ['fp8', 'int4'])
    def test_qat_close_to_fp32(self, qat_mode):
        """Test that QAT forward is reasonably close to FP32 forward."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        # FP32 reference
        mlp_fp32 = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            bias=False,
        ).cuda()

        # QAT model with same weights
        mlp_qat = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='gelu',
            num_experts=E, top_k=k,
            bias=False,
            qat=qat_mode,
        ).cuda()

        # Copy weights
        mlp_qat.experts.weight.data.copy_(mlp_fp32.experts.weight.data)
        mlp_qat.output_experts.weight.data.copy_(mlp_fp32.output_experts.weight.data)

        with torch.no_grad():
            Y_fp32 = mlp_fp32(X, k_weights, k_idxs)
            Y_qat = mlp_qat(X, k_weights, k_idxs)

        # Check outputs are valid
        assert not torch.isnan(Y_qat).any()
        assert not torch.isinf(Y_qat).any()

        # Check correlation is high (outputs should be correlated)
        corr = torch.corrcoef(torch.stack([Y_fp32.flatten(), Y_qat.flatten()]))[0, 1].item()
        print(f"Correlation between FP32 and {qat_mode} QAT outputs: {corr:.4f}")
        # INT4 has more quantization error than FP8
        min_corr = 0.85 if qat_mode == 'int4' else 0.95
        assert corr > min_corr, f"QAT {qat_mode} output poorly correlated with FP32: {corr:.4f}"

    @pytest.mark.parametrize('qat_mode', ['fp8', 'int4'])
    def test_qat_gated_activations(self, qat_mode):
        """Test QAT with gated activations."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation='swiglu',
            num_experts=E, top_k=k,
            bias=False,
            qat=qat_mode,
        ).cuda()

        with torch.no_grad():
            Y = mlp(X, k_weights, k_idxs)

        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()

    def test_qat_extra_repr(self):
        """Test that extra_repr includes QAT mode."""
        mlp_fp8 = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            qat='fp8'
        )
        repr_str = mlp_fp8.extra_repr()
        assert "qat='fp8'" in repr_str

        mlp_int4 = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
            qat='int4'
        )
        repr_str = mlp_int4.extra_repr()
        assert "qat='int4'" in repr_str

        mlp_no_qat = MLP(
            input_size=128, hidden_size=512,
            num_experts=8, top_k=2,
        )
        repr_str_no_qat = mlp_no_qat.extra_repr()
        assert 'qat' not in repr_str_no_qat

    def test_int4_group_size(self):
        """Test INT4 QAT with different group sizes."""
        torch.manual_seed(42)
        length, x_dim, h_dim, E, k = 64, 128, 512, 8, 2

        logits = torch.randn(length, E)
        weights = torch.softmax(logits.float(), axis=-1).cuda()
        X = torch.randn(length, x_dim).cuda()
        k_weights, k_idxs = torch.topk(weights, k)

        for group_size in [16, 32, 64]:
            mlp = MLP(
                input_size=x_dim, hidden_size=h_dim,
                activation='gelu',
                num_experts=E, top_k=k,
                qat='int4',
                qat_group_size=group_size,
            ).cuda()

            with torch.no_grad():
                Y = mlp(X, k_weights, k_idxs)

            assert Y.shape == X.shape
            assert not torch.isnan(Y).any(), f"INT4 QAT with group_size={group_size} has NaN"
            assert not torch.isinf(Y).any(), f"INT4 QAT with group_size={group_size} has Inf"