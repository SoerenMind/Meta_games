#!/usr/bin/env python
"""Test pytorch gradient through gradient."""
import argparse
import copy
import shutil
import sys

import numpy as np
import torch


def parse_args(args=None):
    """Parse command-line arguments.

    Args:
        args: A list of argument strings to use instead of sys.argv.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return parser.parse_args(args)


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.param = torch.nn.Parameter(torch.from_numpy(value))

    def forward(self):
        return self.param

    def data_param(self):
        return self.param


class NestedConstant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.param = Constant(value)

    def forward(self):
        return self.param()

    def data_param(self):
        return self.param.data_param()


def differentiable_gradient_descent_v1(module, loss_fn, step_size, inplace=False):
    """Differentiable loss-minimization gradient descent on module parameters.

    Args:
        module: Perform gradient descent on parameters of this module.
            A `torch.nn.Module`.
        loss_fn: The loss function. Takes `module` as input and returns
            a scalar tensor.
        step_size: The gradient descent step size.
        inplace: If true, modifies `values` in place. Otherwise, creates and
            returns a copy.

    Returns:
        module: The module with updated parameters after gradient descent.
            The original module object if `inplace` is True otherwise a copy.
    """
    if not inplace:
        module = copy.deepcopy(module)

    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)
    for (param_name, param), grad in zip(module._parameters.items(), gradients):
        module._parameters[param_name] = param - step_size * grad
    return module


def rsetattr(obj, name, value):
    name_parts = name.split(".")
    for part in name_parts[:-1]:
        obj = getattr(obj, part)
    return setattr(obj, name_parts[-1], value)


def rgetattr(obj, name):
    name_parts = name.split(".")
    for part in name_parts[:-1]:
        obj = getattr(obj, part)
    return getattr(obj, name_parts[-1])


def differentiable_gradient_descent_v2(module, loss_fn, step_size, inplace=False):
    if not inplace:
        module = copy.deepcopy(module)

    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)
    for (name, param), gradient in zip(module.named_parameters(), gradients):
        rsetattr(module, name, torch.nn.Parameter(param - step_size * gradient))
    return module


def differentiable_gradient_descent_v3(module, loss_fn, step_size, inplace=False):
    if not inplace:
        module = copy.deepcopy(module)

    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)
    for (name, param), gradient in zip(module.named_parameters(), gradients):
        rsetattr(module, name, param - step_size * gradient)
    return module


def differentiable_gradient_descent_v4(module, loss_fn, step_size, inplace=False):
    if not inplace:
        module = copy.deepcopy(module)

    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)
    for (name, param), gradient in zip(module.named_parameters(), gradients):
        param.data = param.data - step_size * gradient
    return module


def differentiable_gradient_descent_v5(module, loss_fn, step_size, inplace=False):
    """V1 but with support for nesting"""
    if not inplace:
        module = copy.deepcopy(module)

    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)
    for (name, param), gradient in zip(module.named_parameters(), gradients):
        name_parts = name.split(".")
        submodule = module
        for part in name_parts[:-1]:
            submodule = getattr(submodule, part)
        submodule._parameters[name_parts[-1]] = param - step_size * gradient
    return module


def differentiable_gradient_descent_v6(module, loss_fn, step_size, inplace=False):
    """Like v5 but calculates gradients on original module not copy."""
    loss = loss_fn(module)
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)

    if not inplace:
        module = copy.deepcopy(module)
    for (name, param), gradient in zip(module.named_parameters(), gradients):
        name_parts = name.split(".")
        submodule = module
        for part in name_parts[:-1]:
            submodule = getattr(submodule, part)
        # Modules do not allow assigning variables to parameters.
        # Instead we have to directly modify the private _parameters dict.
        # This might cause other breakage but at least it works for gradients.
        submodule._parameters[name_parts[-1]] = param - step_size * gradient
    return module


def check_differentiable_gradient(constant_cls, differentiable_gradient_descent):
    """Check that a function implementing differentiable gradient is correct.
    """
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

    def objective(x, y):
        return x * y

    def loss_fn(v):
        return -objective(a(), v())

    c = differentiable_gradient_descent(b, loss_fn=loss_fn, step_size=step_size)

    fac = objective(a(), c())
    a.zero_grad()
    fac.backward()
    grad = a.data_param().grad.data.numpy()
    print(f"Grad: {grad}   (expected 1)")
    return grad == 1.0


def main(args=None):
    """Run script.

    Args:
        args: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(args)
    classes = {"flat": Constant, "nested": NestedConstant}

    gradient_functions = {
        "V1": differentiable_gradient_descent_v1,
        "V2": differentiable_gradient_descent_v2,
        "V3": differentiable_gradient_descent_v3,
        "V4": differentiable_gradient_descent_v4,
        "V5": differentiable_gradient_descent_v5,
        "V6": differentiable_gradient_descent_v6,
    }
    for fn_name, gradient_fn in gradient_functions.items():
        for class_name, constant_cls in classes.items():
            print(f"{fn_name} - {class_name}")
            try:
                is_correct = check_differentiable_gradient(constant_cls, gradient_fn)
            except Exception as e:
                print("Exception:", str(e))
            else:
                print("Correct:", is_correct)
            print()


if __name__ == "__main__":
    try:
        _np = sys.modules["numpy"]
    except KeyError:
        pass
    else:
        _np.set_printoptions(linewidth=shutil.get_terminal_size().columns)
    main()
