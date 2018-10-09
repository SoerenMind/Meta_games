"""Game definitions."""
import torch
import torch.nn.functional as F


def binary_game(player1_payoff_matrix, player1_action, player2_action):
    """Expected payoff to player 1 of a game with 2 players and 2 actions.

    Args:
        player1_payoff_matrix: A 2x2 matrix where cell [i, j] is the
            payoff to player 1 if player 1 plays action i and
            player 2 plays action j.
        player1_action: The probability that player 1 takes action 0.
        player2_action: The probability that player 2 takes action 0.

    Returns:
        The expected utility for player 1.
    """
    player1_distribution = torch.stack([player1_action, 1 - player1_action], -1)
    player2_distribution = torch.stack([player2_action, 1 - player2_action], -1)
    return F.bilinear(player1_distribution, player2_distribution, player1_payoff_matrix[None, :, :])


def prisoners_dilema(player1_action, player2_action):
    """Expected payoff to player 1 in the Prisoner's Dilemma game."""
    return binary_game(torch.tensor([[-1, -3], [0, -2]], dtype=player1_action.dtype), player1_action, player2_action)
