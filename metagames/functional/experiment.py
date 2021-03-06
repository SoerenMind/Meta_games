"""Experiments with functional-form agents."""
import collections
import itertools
import math
from copy import deepcopy

import numpy as np
import torch
import tqdm

from metagames import game

from . import data as mf_data


def scaled_normal_initializer(rand, num_parameters):
    """Initialize from a normal distribution scaled to give norm near 1."""
    return rand.normal(scale=1 / math.sqrt(max(num_parameters, 1)), size=(num_parameters,))


class PlayerSpecification(
    collections.namedtuple(
        "PlayerSpecification", ["agent", "initializer", "loss", "optimizer", "learning_rate", "step_rate",
                                "n_freeze_player_at", "lookahead", "name"]
    )
):
    """A player specification for an experiment.

    Attributes:
        agent: The player agent. Multiple players may share the same agent.
        initializer: Parameter initializer function.
            Maps (numpy_random_state, num_parameters) -> A numpy array of size (num_parameters,)
        loss: A callable that returns the agent's loss on a game outcome.
        optimizer: The player's optimizer class.
        learning_rate: The players's learning rate.
        step_rate: Number of times the player is updated for each global step.
        n_freeze_player_at: int, specifies when to stop updating parameters (opponents may continue). Usually inf.
        lookahead: bool, if True, player optimizes by simulating one opponent gradient step and using the gradient there
        name: An optional player name.
    """

    pass


class _ExperimentPlayer(collections.namedtuple("_ExperimentPlayer", ["spec", "parameters", "optimizer"])):
    """An initialized experiment player.

    Attributes:
        spec: The player specification. A `PlayerSpecification`.
        parameters: The player parameter vector as a torch Tensor.
        optimizer: The initialized optimizer over `parameters`.
    """

    pass


class Experiment:
    """Base experiment runner class.

    Subclasses must implement _make_player_opponents.
    """

    def __init__(self, payoff_matrix, dtype=None):
        """Initialize an experiment runner.

        Args:
            payoff_matrix: The game payoff matrix for the first player.
                An 2x2 numpy array where cell [i, j] is the payoff to the first player
                if they play action `i` and the opponent plays action `j`.
            dtype: Tensor data type.
        """
        if dtype is None:
            dtype = torch.double

        self.payoff_matrix = torch.tensor(payoff_matrix, dtype=dtype)


    def run(self, player_specifications, num_steps, seed=None, logger=None, progress_bar=False):
        data = {"player_specifications": player_specifications, "num_steps": num_steps, "seed": seed}

        players = self._initialize_players(player_specifications, seed=seed)
        steps = self.run_steps(player_specifications, seed=seed, max_steps=num_steps, players=players)

        if progress_bar:
            steps = tqdm.tqdm(steps, total=num_steps)

        if logger is not None:
            steps_data = []
            for step_index, step_data in enumerate(steps):
                logger(step_index, step_data)
                steps_data.append(step_data)
        else:
            steps_data = list(steps)
        data["steps"] = steps_data
        data['players'] = players
        return data

    def run_steps(self, player_specifications, max_steps=None, seed=None, players=None):
        """Yield experiment steps.

        Args:
            player_specifications: A list of `PlayerSpecification` describing the players.
                Multiple players may be specificied but it acts as a
                batch of independent players that play only against themselves.
            seed: Parameter initialization seed.
            max_steps: Optional maximum number of steps to run.

        Yields:
            For each step, a dictionary of step statistics.
        """
        if players is None:
            players = self._initialize_players(player_specifications, seed=seed)
        player_opponents = self._make_player_opponents(players)

        if max_steps:
            steps = range(max_steps)
        else:
            steps = itertools.count()

        for step in steps:
            step_statistics = {"player_updates": []}
            for player, opponents in player_opponents:
                if step >= player.spec.n_freeze_player_at:
                    player.parameters.requires_grad_(False)
                player_update_statistics = self._update_player(player, opponents)
                step_statistics["player_updates"].append(player_update_statistics)
            yield step_statistics, players

    def _make_player_opponents(self, players):
        """Make a list of (player, opponents) pairs."""
        raise NotImplementedError

    def _initialize_players(self, player_specifications, seed=None):
        """Initialize players from a list of player specifications.

        Args:
            player_specifications: The player speceifications.
                An iterable of PlayerSpecification instances.
            seed: Optional player parameter initialization seed.
                The per-player seed depends on this seed and their order in `player_specifications`.
        """
        rand = np.random.RandomState(seed)

        players = []
        for player_spec in player_specifications:
            # Create a per-player random generator to so that player parameters
            # do not depend on the number of random samples generated by other players.
            player_rand = np.random.RandomState(rand.randint(2 ** 32))
            parameters = torch.tensor(
                player_spec.initializer(player_rand, player_spec.agent.num_parameters), requires_grad=True
            )

            optimizer_kwargs = {}
            if player_spec.learning_rate is not None:
                optimizer_kwargs["lr"] = player_spec.learning_rate

            optimizer = player_spec.optimizer([parameters], **optimizer_kwargs)
            players.append(_ExperimentPlayer(spec=player_spec, optimizer=optimizer, parameters=parameters))
        return players

    def _eval_player(self, player, opponents):
        """Play a game against all opponents, just to return stats."""
        game_statistics = []
        for i, opponent in enumerate(opponents):
            game_statistics.append(self._play_game(
                agent=player.spec.agent,
                parameters=player.parameters,
                opponent_agent=opponent.spec.agent,
                opponent_parameters=opponent.parameters,
            ))
        return game_statistics

    def _play_game(self, agent, parameters, opponent_agent, opponent_parameters):
        """Play a game against an opponent."""
        action_logit = agent(parameters, opponent_parameters)
        opponent_action_logit = opponent_agent(opponent_parameters, parameters)
        action_probability = torch.sigmoid(action_logit)
        opponent_action_probability = torch.sigmoid(opponent_action_logit)

        # Utility for circling game wit dot agents
        if list(self.payoff_matrix.shape) == []:
            # TODO(sorenmind): Change sign for player 2
            utility = action_logit
        else:
            utility = game.binary_game(self.payoff_matrix, action_probability, opponent_action_probability)
        return {
            "utility": utility,
            "action_probability": action_probability,
            "opponent_action_probability": opponent_action_probability,
            "action_logit": action_logit,
            "opponent_action_logit": opponent_action_logit,
        }

    def _play_game_grad(self, player, opponent):
        """Play against an opponent and update player gradients."""

        agent = player.spec.agent
        parameters = player.parameters
        loss_fn = player.spec.loss()

        results = self._play_game(agent, parameters, opponent.spec.agent, opponent.parameters)
        statistics = {name: _tensor_data(value) for name, value in results.items()}

        loss = loss_fn(**results, parameter_vector=parameters, opponent_parameter_vector=opponent.parameters)
        opp_requires_grad = opponent.parameters.requires_grad
        # TODO: does this throw away the accumulated gradient? I think not.
        opponent.parameters.requires_grad_(False)
        loss.backward()
        opponent.parameters.requires_grad_(opp_requires_grad)
        return statistics

    def _play_game_lookahead_grad(self, player, opponent):
        """Play against an opponent and update player gradients with lookahead gradient of other player."""
        agent = player.spec.agent
        parameters = player.parameters
        loss_fn = player.spec.loss()

        # First record stats
        results_no_lookahead = self._play_game(agent, parameters, opponent.spec.agent, opponent.parameters)
        statistics = {name: _tensor_data(value) for name, value in results_no_lookahead.items()}

        # Let opponent do a naive update
        opp_copy = deepcopy(opponent)
        opp_copy.parameters.requires_grad_(True)
        opp_copy.optimizer.zero_grad()
        self._play_game_grad(opp_copy, player)
        opp_copy.optimizer.step()
        results_lookahead = self._play_game(agent, parameters, opp_copy.spec.agent, opp_copy.parameters)

        # TODO: zero_grad erases the gradients from playing against other players
        player.optimizer.zero_grad()    # May have accumulated from opponent loss.backward
        loss = loss_fn(**results_lookahead, parameter_vector=parameters, opponent_parameter_vector=opp_copy.parameters)
        loss.backward()

        return statistics

    def _update_player(self, player, opponents, update_steps=None):
        """Update a player from plays against a set of opponents.
        Player may take multiple update steps ('rounds').

        Args:
            player: The player to update. An instance of _ExperimentPlayer.
            opponents: The opponents. An iterable of _ExperimentPlayer.
            update_steps: Override the number of gradient descent steps performed.
                By default, uses player.spec.step_rate. This allows a player to update more
                often than others.
        """
        if update_steps is None:
            update_steps = player.spec.step_rate

        statistics = []
        # TODO: remove
        if player.spec.lookahead:
            accum_grad_fn = self._play_game_lookahead_grad
        else:
            accum_grad_fn = self._play_game_grad
        for _ in range(update_steps):
            player.optimizer.zero_grad()

            round_statistics = {"rounds": []}
            # Accumulate gradients against each opponent
            for opponent in opponents:
                game_statistics = accum_grad_fn(player, opponent=opponent)
                round_statistics["rounds"].append({"opponent": opponent.spec, **game_statistics})

            round_statistics["grad_norm"] = _tensor_data(torch.norm(player.parameters.grad))
            round_statistics["mean_utility"] = np.mean(
                [game_stats["utility"] for game_stats in round_statistics["rounds"]]
            )
            statistics.append(round_statistics)

            player.optimizer.step()

        return statistics


class SelfPlayExperiment(Experiment):
    """All players play against themselves only."""

    def __init__(self, payoff_matrix, self_aware=False, dtype=None):
        """Initialize a self-play experiment.

        Args:
            payoff_matrix: The game payoff matrix for the first player.
                An 2x2 numpy array where cell [i, j] is the payoff to the first player
                if they play action `i` and the opponent plays action `j`.
            self_aware: Whether the players are aware that they are playing against themselves.
                If True, gradient flows through the copied opponent parameters.
            dtype: Tensor data type.
        """
        super().__init__(payoff_matrix=payoff_matrix, dtype=dtype)
        self.self_aware = self_aware

    def _make_player_opponents(self, players):
        if self.self_aware:
            return [(player, (player,)) for player in players]
        else:
            # The detached version is a reference that has the same value as the original
            # parameters, so it is unnecessary to update on every gradient step.
            return [(player, (player._replace(parameters=player.parameters.detach()),)) for player in players]


class DuelExperiment(Experiment):
    """Two agents compete. They may have different parameter sizes."""

    def _make_player_opponents(self, players):
        # If there are exactly two players, they compete
        try:
            first, second = players
        except ValueError:
            pass
        else:
            return [(first, (second,)), (second, (first,))]

        # If there are more than 2 players, there must be exactly 2 unique agents.
        # Each players of one agent plays against all players of the other agent and
        # vice versa.
        agent_ids = set(id(player.spec.agent) for player in players)
        agent_players = {agent_id: [] for agent_id in agent_ids}
        for player in players:
            agent_players[id(player.spec.agent)].append(player)

        try:
            firsts, seconds = agent_players.values()
        except ValueError:
            raise ValueError("There must be exactly 2 agents.")
        return [(first, seconds) for first in firsts] + [(second, firsts) for second in seconds]


class FreeForAllExperiment(Experiment):
    """Every player competes against every other player."""

    def _make_player_opponents(self, players):
        return [(player, players[:i] + players[i + 1 :]) for (i, player) in enumerate(players)]


class ExperimentLogger:
    def __init__(self, log_every_n, substep_keys=("grad_norm",), round_keys=("utility",)):
        self.log_every_n = log_every_n
        self.substep_keys = substep_keys
        self.round_keys = round_keys
        self.recorded_data = {}

    def __call__(self, step_index, step_data):
        step_statistics = mf_data.experiment_single_step_statistics(
            step_data, substep_keys=self.substep_keys, round_keys=self.round_keys, statistic_types=("mean",)
        )
        mf_data.append_step_statistics(self.recorded_data, step_statistics)
        if not (step_index + 1) % self.log_every_n:
            for player, player_stats in self.recorded_data.items():
                for key, values in player_stats.items():
                    print("Step %d, Player %s, %s %f" % (step_index, player, key, np.mean(values["mean"])))
            self.recorded_data = {}


def _tensor_data(tensor):
    """The contents of a torch tensor as a numpy array."""
    return tensor.detach().cpu().numpy()
