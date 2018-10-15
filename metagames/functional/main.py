"""High-level interface to open-source game experiments.
"""
import numpy as np
import torch

from . import agents
from . import experiment
from . import losses

GAMES = {"prisoners_dilemma": np.array([[-1, -3], [0, 2]], dtype=float)}

AGENTS = {
    "dot": agents.DotProductAgent,
    "sum": agents.SumAgent,
    "selfish": agents.SelfishAgent,
    "cooperate": agents.Play1Agent,
    "defect": agents.Play0Agent,
    "linear": agents.LinearAgent,
    "simple-linear": agents.SimpleLinearAgent,
    "clique": agents.SimilarityAgent,
    "nn": agents.SubspaceNeuralNetworkAgent,
}

LOSSES = {
    "utility": losses.UtilityLoss,
    "copy": losses.CopyLoss,
    "cooperate": losses.Play1Loss,
    "defect": losses.Play0Loss,
}

OPTIMIZERS = {"grad": torch.optim.SGD, "adam": torch.optim.Adam, "lbfgs": torch.optim.LBFGS}


PARAMETER_INITIALIZERS = {"scaled_normal": experiment.scaled_normal_initializer, "zeros": lambda rand, n: np.zeros(n)}


def prepare_player_specifications(agents_config, default_config=None, agent_seed=None):
    """Create player specifications a list of agent configuration dictionaries.

    Args:
        agents_config: A list of agent configuration dictionaries. Dictionary keys:
            - agent (AGENTS key)
            - num_parameters
            - num_opponent_parameters
            - initializer (INITIALIZERS key)
            - loss (LOSSES key)
            - learning_rate
            - optimizer (OPTIMIZERS key)
            - step_rate
            - num_players
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
            rand=np.random.RandomState(rand.randint(2**32)),
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
            agent_name = f"{default_config.get('name', agent_cls.__name__)}_{i}"
        num_players = get_config(agent_config, "num_players", 1)

        for j in range(num_players):
            if num_players > 1:
                player_name = f"{agent_name}_{j}"
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
