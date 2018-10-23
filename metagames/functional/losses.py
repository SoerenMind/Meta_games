"""Two-player binary-action game loss functions for functional agents.

The players output a logit indiciating the probability that they play the second action.
"""
import torch


class BaseLoss:
    """Loss function base class."""

    def __call__(self, *, utility, action_logit, opponent_action_logit, parameter_vector, opponent_parameter_vector):
        raise NotImplementedError


class UtilityLoss(BaseLoss):
    """Maximize utility."""

    def __call__(self, *, utility, **kwargs):
        del kwargs
        return -utility


class CopyLoss(BaseLoss):
    """Play the same action as the opponent."""

    def __call__(self, *, action_logit, opponent_action_logit, **kwargs):
        del kwargs
        return torch.nn.functional.mse_loss(action_logit, opponent_action_logit)


class Play0Loss(BaseLoss):
    """Play the first action."""

    def __call__(self, *, action_logit, **kwargs):
        del kwargs
        return action_logit


class Play1Loss(BaseLoss):
    """Play the second action."""

    def __call__(self, *, action_logit, **kwargs):
        del kwargs
        return -action_logit
