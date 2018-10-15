"""Game definitions."""
import numpy as np
import torch


def binary_game(player1_payoff_matrix, player1_action, player2_action):
    """Expected payoff to player 1 of a game with 2 players and 2 actions.

    Args:
        player1_payoff_matrix: A 2x2 matrix where cell [i, j] is the
            payoff to player 1 if player 1 plays action i and
            player 2 plays action j.
        player1_action: The probability that player 1 takes action 1.
        player2_action: The probability that player 2 takes action 1.

    Returns:
        The expected utility for player 1.
    """
    player1_distribution = torch.stack([1 - player1_action, player1_action], -1)
    player2_distribution = torch.stack([1 - player2_action, player2_action], -1)
    return torch.sum(torch.matmul(player1_distribution, player1_payoff_matrix) * player2_distribution, dim=-1)


PRISONERS_DILEMMA = np.array([[-2, 0], [-3, -1]], dtype=float)
