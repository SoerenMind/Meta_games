"""Agents defined functionally as f(my_params, opponent_params).

The agents output a logit indicating the probability of playing
the second action in a binary-action game.
"""
import numpy as np
import torch
import torch.nn.functional as F


class OpenSourceBinaryGameAgent(torch.nn.Module):
    """Play a binary-action game given own and opponent's parameters."""

    def __init__(self, num_parameters, num_opponent_parameters=None, **kwargs):
        """Initialize an OpenSourceBinaryGameAgent

        Args:
            num_parameters: Number of agent parameters.
            num_opponent_parameters: Number of opponent parameters.
                Defaults to `num_parameters`.
            **kwargs: Ignore additional arguments to support common initialization
                of subclasses that have different initialization parameters.
        """
        del kwargs
        self.num_parameters = num_parameters
        if num_opponent_parameters is None:
            num_opponent_parameters = num_parameters
        self.num_opponent_parameters = num_opponent_parameters
        super().__init__()

    def forward(self, my_params, other_params):
        """Log-odds of playing the second action."""
        raise NotImplementedError


class SumAgent(OpenSourceBinaryGameAgent):
    """Play according to parameter sum with opponent.

    action_logit(y|x) = sum(x + y)
    """

    def forward(self, my_params, other_params):
        return torch.sum(my_params + other_params)


class DotProductAgent(OpenSourceBinaryGameAgent):
    """Play according to parameter dot product with opponent.

    action_logit(y|x) = x' * y
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_parameters != self.num_opponent_parameters:
            raise ValueError("{:s} requires num_opponent_parameters == num_parameters.".format(self.__class__.__name__))

    def forward(self, my_params, other_params):
        return torch.sum(my_params * other_params, -1)


class LinearAgent(OpenSourceBinaryGameAgent):
    """Bilinear combination of self & opponent.

    action_logit(y|x) = y' M x + b' x
    """

    def __init__(self, num_parameters, num_opponent_parameters=None, rand=None, dtype=torch.double, **kwargs):
        """Initialize a LinearAgent.

        Args:
            num_parameters: Number of agent parameters.
            num_opponent_parameters: Number of opponent parameters.
                Defaults to `num_parameters`.
            rand: Numpy random state. If None, uses a fixed simple initialization.
            dtype: Parameter data type.
        """
        super().__init__(num_parameters=num_parameters, num_opponent_parameters=num_opponent_parameters, **kwargs)

        if rand is None:
            if self.num_parameters != self.num_opponent_parameters:
                raise ValueError(
                    "Deterministic initialization only defined when opponent has same number of parameters."
                )
            weights = torch.eye(num_parameters, dtype=dtype)
            weights[1, 0] = 1.0
            bias = torch.zeros((num_parameters,), dtype=dtype)
            bias[0] = 1.0
        else:
            scale = 1 / np.sqrt(num_parameters)
            weights = torch.from_numpy(
                rand.normal(size=(self.num_opponent_parameters, self.num_parameters), scale=scale)
            )
            bias = torch.from_numpy(rand.normal(size=(self.num_parameters,), scale=scale))
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, my_params, other_params):
        return F.bilinear(other_params[None, :], my_params[None, :], self.weights[None, :, :]) + torch.squeeze(
            F.linear(my_params, self.bias[None, :]), dim=-1
        )


class SimpleLinearAgent(LinearAgent):
    def __init__(self, **kwargs):
        kwargs.pop("rand", None)
        super().__init__(**kwargs)


class SelfishAgent(OpenSourceBinaryGameAgent):
    """Ignores opponent.

    action_logit(y|x) = sum(x)
    """

    def forward(self, my_params, other_params):
        del other_params
        return torch.sum(my_params)


class _ConstantAgent(OpenSourceBinaryGameAgent):
    """Plays actions with a constant probability."""

    def __init__(self, action_logit, dtype=torch.double, **kwargs):
        super().__init__(**kwargs)
        self.action_logit = torch.tensor(action_logit, dtype=dtype, requires_grad=True)

    def forward(self, my_params, other_params):
        del my_params, other_params
        return self.action_logit


class Play0Agent(_ConstantAgent):
    def __init__(self, **kwargs):
        super().__init__(action_logit=-100.0, **kwargs)


class Play1Agent(_ConstantAgent):
    def __init__(self, **kwargs):
        super().__init__(action_logit=100.0, **kwargs)


class SimilarityAgent(OpenSourceBinaryGameAgent):
    """Play according to parameter L2 similarity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_parameters != self.num_opponent_parameters:
            raise ValueError("{:s} requires num_opponent_parameters == num_parameters.".format(self.__class__.__name__))

    def forward(self, my_params, other_params):
        return -torch.log(5 * torch.norm(my_params - other_params))


class SubspaceNeuralNetworkAgent(OpenSourceBinaryGameAgent):
    """Play according to the output of a NN on the opponents parameters.

    Network parameters are affine transformations of the agent's own parameters.
    """

    def __init__(
        self, num_parameters, num_opponent_parameters=None, layer_rel_sizes=(5,), dtype=torch.double, **kwargs
    ):
        """Initialize a NeuralNetworkAgent.

        Args:
            num_parameters: Number of agent parameters.
            num_opponent_parameters: Number of opponent parameters.
                Defaults to `num_parameters`.
            layer_rel_sizes: Hidden layer sizes, as multiples of num_parameters.
        """
        super().__init__(num_parameters=num_parameters, num_opponent_parameters=num_opponent_parameters, **kwargs)
        hyper_layers = []
        self.layer_sizes = []

        input_size = self.num_opponent_parameters
        # None for the final action logit layer with size 1
        for rel_size in tuple(layer_rel_sizes) + (None,):
            if rel_size is None:
                output_size = 1
            else:
                output_size = input_size * rel_size
            self.layer_sizes.append((input_size, output_size))
            hyper_layers.append(
                torch.nn.ModuleDict(
                    {
                        "weight": torch.nn.Linear(self.num_parameters, input_size * output_size).to(dtype),
                        "bias": torch.nn.Linear(self.num_parameters, output_size).to(dtype),
                    }
                )
            )
            input_size = output_size
        self.hyper_layers = torch.nn.ModuleList(hyper_layers)

    def forward(self, my_params, other_params):
        h = other_params
        for i, (hyper_layer, (in_size, out_size)) in enumerate(zip(self.hyper_layers, self.layer_sizes)):
            weight = hyper_layer["weight"](my_params).reshape(out_size, in_size)
            bias = hyper_layer["bias"](my_params)
            h = torch.nn.functional.linear(h, weight=weight, bias=bias)
            if i != len(self.layer_sizes) - 1:
                h = torch.nn.functional.relu(h)
        return torch.squeeze(h, dim=-1)
