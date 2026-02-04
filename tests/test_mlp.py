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