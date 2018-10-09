"""Agents defined functionally as f(my_params, opponent_params)."""
import numpy as np
import torch
import torch.nn.functional as F


class BasePrisonersDilemmaAgent(torch.nn.Module):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()

    def forward(self, my_params, other_params):
        """Log-odds of cooperating."""
        raise NotImplementedError


class SumAgent(BasePrisonersDilemmaAgent):
    """Cooperates according to sum with opponent.

    Pr(x cooperates with y) is sigmoid(sum(x + y)).
    """

    def forward(self, my_params, other_params):
        return torch.sum(my_params + other_params)


class DotProductAgent(BasePrisonersDilemmaAgent):
    """Cooperates according to dot product with opponent.

    Pr(x cooperates with y) is sigmoid(dot(x, y)).
    """

    def forward(self, my_params, other_params):
        return torch.sum(my_params * other_params, -1)


class LinearAgent(BasePrisonersDilemmaAgent):
    """Bilinear combination of self & opponent.

    Pr(x cooperates with y) is sigmoid(y' M x + b' x).
    """

    def __init__(self, num_parameters, rand=None, dtype=torch.double, **kwargs):
        """Initialize a LinearAgent.

        Args:
            num_parameters: Number of agent and opponent parameters.
            rand: Numpy random state. If None, uses a fixed simple initialization.
        """
        super().__init__(**kwargs)
        if rand is None:
            weights = torch.eye(num_parameters, dtype=dtype)
            weights[1, 0] = 1.0
            bias = torch.zeros((num_parameters,), dtype=dtype)
            bias[0] = 1.0
        else:
            scale = 1 / np.sqrt(num_parameters)
            weights = torch.from_numpy(rand.normal(size=(num_parameters, num_parameters), scale=scale))
            bias = torch.from_numpy(rand.normal(size=(num_parameters,), scale=scale))
            print(weights.numpy())
            print(bias.numpy())
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


class SelfishAgent(BasePrisonersDilemmaAgent):
    """Ignores opponent.

    Pr(x cooperates with y) is sigmoid(sum(x)).
    """

    def forward(self, my_params, other_params):
        del other_params
        return torch.sum(my_params)


class _ConstantAgent(BasePrisonersDilemmaAgent):
    """Cooperates with constant probability."""

    def __init__(self, cooperate_logit, **kwargs):
        super().__init__(**kwargs)
        self.cooperate_logit = torch.tensor(cooperate_logit, requires_grad=True)

    def forward(self, my_params, other_params):
        del my_params, other_params
        return self.cooperate_logit


class CooperateAgent(_ConstantAgent):
    def __init__(self, **kwargs):
        super().__init__(cooperate_logit=100.0, **kwargs)


class DefectAgent(_ConstantAgent):
    def __init__(self, **kwargs):
        super().__init__(cooperate_logit=-100.0, **kwargs)


class CliqueAgent(BasePrisonersDilemmaAgent):
    """Cooperates with the other agent based on L2 similarity."""
    def forward(self, my_params, other_params):
        return - torch.log(5 * torch.norm(my_params - other_params))
