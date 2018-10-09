#!/usr/bin/env python
"""Open Source Prisoner's Dilema with functional agents."""
import argparse
import shutil
import sys

import numpy as np
import torch
import tqdm

from metagames import functional_agents
from metagames import game
import metagames.utils.cli as cli_utils

AGENTS = {
    "dot": functional_agents.DotProductAgent,
    "sum": functional_agents.SumAgent,
    "selfish": functional_agents.SelfishAgent,
    "cooperate": functional_agents.CooperateAgent,
    "defect": functional_agents.DefectAgent,
    "linear": functional_agents.LinearAgent,
    "simple-linear": functional_agents.SimpleLinearAgent,
}
OPTIMIZERS = {"grad": torch.optim.SGD, "adam": torch.optim.Adam, "lbfgs": torch.optim.LBFGS}


def copy_loss_logit(p1_logit, p2_logit, utility):
    return torch.nn.functional.losses.mse_loss(p1_logit, p2_logit)


LOSSES = {
    "utility": lambda p1_logit, p2_logit, utility: -utility,
    "copy-logit": copy_loss_logit,
    "cooperate": lambda p1_logit, p2_logit, utility: -p1_logit,
    "defect": lambda p1_logit, p2_logit, utility: p1_logit,
}


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
    parser.add_argument("-k", "--num-parameters", default=10, type=int, help="Number of agent parameters.")
    parser.add_argument(
        "--agent", action=cli_utils.DictLookupAction, choices=AGENTS, default="dot", help="The agent type."
    )
    # Opponent types:
    #   self-aware: Same type as AGENT. Same parameters with gradients.
    #   self-unaware: Same type as AGENT. Same parameters without gradients.
    #   same-independent: Same type as AGENT. Independent parameters.
    #   <type>: An agent with the given type. Independent parameters.
    parser.add_argument(
        "--opponent",
        choices=("self_aware", "self_unaware", "independent"),
        default="independent",
        help=("Opponent parameters. self_aware: same with gradient; self_unaware: same without gradient."),
    )

    parser.add_argument("--num-steps", default=10000, type=int, help="Number of optimization steps.")
    parser.add_argument("--learning-rate", type=float, help="Agent learning rate.")
    parser.add_argument(
        "--optimizer", action=cli_utils.DictLookupAction, choices=OPTIMIZERS, default="grad", help="Optimizer."
    )
    parser.add_argument(
        "--objective",
        action=cli_utils.DictLookupAction,
        choices=LOSSES,
        default=["utility"],
        nargs="+",
        metavar=("AGENT", "OPPONENT"),
        help=f"Objective function. Choices: {set(LOSSES.keys())}",
    )

    parser.add_argument("--parameter-seed", type=int, default=1, help="Seed for parameter initialization.")
    parser.add_argument("--agent-seed", type=int, default=2, help="Seed for agent meta-parameter initialization.")
    return parser.parse_args(args)


def make_parameters(rand, size):
    return torch.nn.Parameter(torch.from_numpy(rand.normal(size=size, scale=1 / np.sqrt(size))))


def main(args=None):
    """Run script.

    Args:
        args: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(args)
    num_parameters = args.num_parameters

    rand = np.random.RandomState(args.parameter_seed)
    agent_rand = np.random.RandomState(args.agent_seed)

    agent = args.agent(num_parameters, rand=agent_rand)
    agent_parameters = make_parameters(rand, num_parameters)

    optimizer_kwargs = {}
    if args.learning_rate is not None:
        optimizer_kwargs["lr"] = args.learning_rate
    agent_optimizer = args.optimizer([agent_parameters], **optimizer_kwargs)

    loss_fn, = args.objective

    if args.opponent == "independent":
        opponent_parameters = make_parameters(rand, num_parameters)
    else:
        opponent_parameters = {
            "self_aware": lambda: agent_parameters,
            "self_unaware": lambda: agent_parameters.detach(),
        }[args.opponent]()

    # for _ in tqdm.tqdm(range(args.num_steps)):
    for i in range(args.num_steps):
        agent_logit = agent(agent_parameters, opponent_parameters)
        opponent_logit = agent(opponent_parameters, agent_parameters)
        agent_cooperate_prob = torch.sigmoid(agent_logit)
        opponent_cooperate_prob = torch.sigmoid(opponent_logit)

        agent_utility = game.prisoners_dilema(agent_cooperate_prob, opponent_cooperate_prob)
        loss = loss_fn(agent_logit, opponent_logit, agent_utility)
        print(f"i: {i}")
        print(f"agent params: {agent_parameters.detach().numpy()}")
        print(f"oppnt params: {opponent_parameters.detach().numpy()}")
        print(f"agent logit: {agent_logit.detach().numpy()}")
        print(f"oppnt logit: {opponent_logit.detach().numpy()}")
        print(f"agent cooperate prob: {agent_cooperate_prob.detach().numpy()}")
        print(f"oppnt cooperate prob: {opponent_cooperate_prob.detach().numpy()}")
        # Self-cooperation probability - not necessary for gradient
        agent_parameters_nograd = agent_parameters.detach()
        agent_self_cooperate_prob = (
            torch.sigmoid(agent(agent_parameters_nograd, agent_parameters_nograd)).detach().numpy()
        )
        print(f"agent self cooperate prob: {agent_self_cooperate_prob}")
        print(f"agent utility: {agent_utility.detach().numpy()}")
        print(f"loss: {loss.detach().numpy()}")
        print()

        agent_optimizer.zero_grad()
        loss.backward()
        agent_optimizer.step()


if __name__ == "__main__":
    try:
        _np = sys.modules["numpy"]
    except KeyError:
        pass
    else:
        _np.set_printoptions(linewidth=shutil.get_terminal_size().columns)
    main()
