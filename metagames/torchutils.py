"""General-purpose utilities involving torch."""
import copy

import torch


def differentiable_gradient_step(module, loss, step_size, inplace=False):
    """A differentiable loss-minimizing gradient descent step on module parameters.

    Args:
        module: Perform gradient descent on parameters of this module.
            A `torch.nn.Module`.
        loss: The loss to minimize.
        step_size: The gradient descent step size.
        inplace: If true, modifies `values` in place. Otherwise, creates and
            returns a copy.

    Returns:
        module: The module with updated parameters after gradient descent.
            The original module object if `inplace` is True otherwise a copy.

    Warning:
        The output module parameters are turned into tensors.
        The Module class tries to disallow this so there might be some
        negative consequences. Be cautious when using the result.
    """
    gradients = torch.autograd.grad(loss, module.parameters(), create_graph=True)

    if not inplace:
        module = copy.deepcopy(module)

    for (param_name, param), gradient in zip(module.named_parameters(), gradients):
        name_parts = param_name.split(".")
        submodule = module
        for part in name_parts[:-1]:
            submodule = getattr(submodule, part)
        # Modules do not allow assigning variables to parameters.
        # Instead we have to directly modify the private _parameters dict.
        # This might cause other problems but at least it works for gradients.
        submodule._parameters[name_parts[-1]] = param - step_size * gradient
    return module


def differentiable_gradient_descent(module, loss_fn, num_steps, step_size, inplace=False):
    """Differentiable loss-minimizing gradient descent on module parameters.

    Args:
        module: Perform gradient descent on parameters of this module.
            A `torch.nn.Module`.
        loss_fn: Loss function to minimize.
            Takes `module` as an argument and returns a tensor.
        num_steps: Number of gradient descent steps to apply.
        step_size: The gradient descent step size.
        inplace: If true, modifies `values` in place. Otherwise, creates and
            returns a copy.

    Returns:
        module: The module with updated parameters after gradient descent.
            The original module object if `inplace` is True otherwise a copy.

    Warning:
        The output module parameters are turned into tensors.
        The Module class tries to disallow this so there might be some
        negative consequences. Be cautious when using the result.
    """
    if not inplace:
        module = copy.deepcopy(module)
    for _ in range(num_steps):
        loss = loss_fn(module)
        module = differentiable_gradient_step(module, loss, step_size=step_size, inplace=True)
    return module
