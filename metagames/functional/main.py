"""High-level interface to open-source game experiments.
"""
import functools

import numpy as np
import torch

from . import agents
from . import experiment
from . import losses
from metagames import game

EXPERIMENTS = {
    "self_play_self_aware": functools.partial(experiment.SelfPlayExperiment, self_aware=True),
    "self_play_self_unaware": functools.partial(experiment.SelfPlayExperiment, self_aware=False),
}


GAMES = {"prisoners_dilemma": game.PRISONERS_DILEMMA}

AGENTS = {
    "dot": agents.DotProductAgent,
    "sum": agents.SumAgent,
    "selfish": agents.SelfishAgent,
    "defect": agents.Play0Agent,
    "cooperate": agents.Play1Agent,
    "linear": agents.LinearAgent,
    "simple-linear": agents.SimpleLinearAgent,
    "clique": agents.SimilarityAgent,
    "nn": agents.SubspaceNeuralNetworkAgent,
}

LOSSES = {
    "utility": losses.UtilityLoss,
    "copy": losses.CopyLoss,
    "defect": losses.Play0Loss,
    "cooperate": losses.Play1Loss,
}

OPTIMIZERS = {"grad": torch.optim.SGD, "adam": torch.optim.Adam, "lbfgs": torch.optim.LBFGS}


PARAMETER_INITIALIZERS = {"scaled_normal": experiment.scaled_normal_initializer, "zeros": lambda rand, n: np.zeros(n)}


def run_experiment(
    experiment, game, num_steps, agents_config, default_agent_config=None, agent_seed=None, parameter_seed=None
):
    """Run an open source game play experiment.

    Args:
        experiment: The experiment type. A key of `EXPERIMENTS`.
        game: The game. A key of `GAMES`.
        agents_config: A list of agent configuration dictionaries.
        default_agent_config: The default agent configuration to use when a key is missing
            from `agents_config`.
        agents_seed: The seed used to initialize the agent structure.
        parameter_seed: The seed used to initialize player parameter vectors.

    The agent configuration dictionary has the following keys:
    AgentConfig:
        agent: The agent type. A key of `AGENTS`.
        num_parameters: Size of the parameter vector.
        num_opponent_parameters: Optional size of the opponent's parameter vector.
            Defaults to `num_parameters`.
        initializer: The parameter vector initializer. A key of `PARAMETER_INITIALIZERS`.
        loss: The loss function. A key of `LOSSES`.
        learning_rate: The learning rate.
        optimizer: The optimizer to sue. A key of `OPTIMIZERS`.
        step_rate: Number of sub-steps per global step. Optional, defaults to 1.
        num_players: The number of players using this agent.
        name: An optional name for the agent.

    Returns:
        A dictionary of the run data.
    """

    payoff_matrix = GAMES[game]
    experiment_runner = EXPERIMENTS[experiment](payoff_matrix, dtype=torch.double)
    player_specifications = prepare_player_specifications(
        agents_config, default_config=default_agent_config, agent_seed=agent_seed
    )
    return experiment_runner.run(player_specifications=player_specifications, num_steps=num_steps, seed=parameter_seed)


def prepare_player_specifications(agents_config, default_config=None, agent_seed=None):
    """Create player specifications a list of agent configuration dictionaries.

    Args:
        agents_config: A list of agent configuration dictionaries.
        default_config: The default agent configuration to use when a key is missing
            from `agents_config`.
        agents_seed: The seed used to initialize the agent structure.

    The agent configuration dictionary has the following keys:
    AgentConfig:
        agent: The agent type. A key of `AGENTS`.
        num_parameters: Size of the parameter vector.
        num_opponent_parameters: Optional size of the opponent's parameter vector.
            Defaults to `num_parameters`.
        initializer: The parameter vector initializer. A key of `PARAMETER_INITIALIZERS`.
        loss: The loss function. A key of `LOSSES`.
        learning_rate: The learning rate.
        optimizer: The optimizer to sue. A key of `OPTIMIZERS`.
        step_rate: Number of sub-steps per global step. Optional, defaults to 1.
        num_players: The number of players using this agent.
        name: An optional name for the agent.

    Returns:
        A list of `PlayerSpecification` objects.
    """
    rand = np.random.RandomState(agent_seed)

    if default_config is None:
        default_config = {}

    def get_config(agent_config, key, final_default=KeyError):
        try:
            return agent_config[key]
        except KeyError:
            if final_default is KeyError:
                return default_config[key]
            else:
                return default_config.get(key, final_default)

    player_specifications = []
    for i, agent_config in enumerate(agents_config):
        num_parameters = get_config(agent_config, "num_parameters")
        num_opponent_parameters = get_config(agent_config, "num_opponent_parameters", None)
        agent_cls = AGENTS[get_config(agent_config, "agent")]
        agent = agent_cls(
            num_parameters=num_parameters,
            num_opponent_parameters=num_opponent_parameters,
            rand=np.random.RandomState(rand.randint(2 ** 32)),
            dtype=torch.double,
        )

        initializer = PARAMETER_INITIALIZERS[get_config(agent_config, "initializer")]
        loss = LOSSES[get_config(agent_config, "loss", None)]
        learning_rate = get_config(agent_config, "learning_rate", None)
        optimizer = OPTIMIZERS[get_config(agent_config, "optimizer")]
        step_rate = get_config(agent_config, "step_rate", 1)
        try:
            agent_name = agent_config["name"]
        except KeyError:
            agent_name = default_config.get('name', agent_cls.__name__)
        num_players = get_config(agent_config, "num_players", 1)

        for j in range(num_players):
            if num_players > 1:
                player_name = "{:s}_{:d}".format(agent_name, j)
            else:
                player_name = agent_name

            player_specifications.append(
                experiment.PlayerSpecification(
                    agent=agent,
                    initializer=initializer,
                    loss=loss,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    step_rate=step_rate,
                    name=player_name,
                )
            )
    return player_specifications
