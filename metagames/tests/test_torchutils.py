"""Test torch utilities."""
import numpy as np
import pytest
import torch

from metagames import torchutils


class _Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.param = torch.nn.Parameter(torch.from_numpy(value))

    def forward(self):
        return self.param

    def data_param(self):
        return self.param


class _NestedConstant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.param = _Constant(value)

    def forward(self):
        return self.param()

    def data_param(self):
        return self.param.data_param()


@pytest.fixture(params=[_Constant, _NestedConstant])
def constant_cls(request):
    return request.param


def test_differentiable_gradient_step(constant_cls):
    # Have f(a, b)
    # Apply gradient step to b, with step size α:
    #   c = b + α * ∂f(a,b)/∂b
    #
    # Want total gradient w.r.t a:
    #   df(a,c)/da = ∂f(a,c)/∂a + ∂f(a,c)/∂c * ∂c/∂a
    #              = ∂f(a,c)/∂a + α * ∂f(a,c)/∂c * ∂f(a,b)/∂a∂b
    #
    # With f(a, b) = a*b we get:
    #
    # ∂f(a,b)/∂a = b
    # ∂f(a,c)/∂a = c = b + αa
    # df(a,c)/da = c + α * a * 1 = b + 2αa
    #
    # Setting a = -1, b = 3, α = 1 the values are
    #
    # ∂f(a,b)/∂a = 3
    # ∂f(a,c)/∂a = 2
    # df(a,c)/da = 1
    #
    # With the last one being the answer we want.
    a = constant_cls(np.array(-1.0))
    b = constant_cls(np.array(3.0))
    step_size = 1

    def f(x, y):
        return x * y

    loss = -f(a(), b())
    c = torchutils.differentiable_gradient_step(b, loss, step_size=step_size)
    fac = f(a(), c())

    a.zero_grad()
    fac.backward()
    assert a.data_param().grad.data.numpy() == 1.0


@pytest.mark.parametrize('num_steps', [0, 1, 2])
def test_differentiable_gradient_descent(constant_cls, num_steps):
    # Have f(a, b) = a * b
    # Apply gradient step to b0, with step size α:
    #   b1 = b0 + α * ∂f(a,b0)/∂b
    #
    # df(a,b1)/da = ∂f(a,b1)/∂a + ∂f(a,b1)/∂c * ∂b1/∂a
    #             = b0 + 2αa
    #
    # With further steps:
    #   bn = b(n-1) + α * ∂f(a,b(n-1))/∂b = b0 + αna
    #
    # df(a,bn)/da = b0 + 2αna
    a = constant_cls(np.array(-1.0))
    b = constant_cls(np.array(3.0))
    step_size = 1

    def f(x, y):
        return x * y

    def loss_fn(bn):
        return -f(a(), bn())

    c = torchutils.differentiable_gradient_descent(b, loss_fn, num_steps=num_steps, step_size=step_size)
    fac = f(a(), c())

    a.zero_grad()
    fac.backward()
    assert a.data_param().grad.data.numpy() == 3 - 2 * num_steps
