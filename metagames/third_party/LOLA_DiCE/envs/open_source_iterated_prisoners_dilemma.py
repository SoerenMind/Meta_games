"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np
import torch

from gym.spaces import Discrete, Tuple

from .common import OneHot



class OpenSourceIteratedPrisonersDilemma(gym.Env):
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, gamma, max_steps=100, batch_size=1, device=None):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = np.array([[-2,0],[-3,-1]])
        self.states = np.array([[1,2],[3,4]])
        self.gamma = gamma
        self.device = device

        # self.action_space = Tuple([
        #     Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        # ])
        # self.observation_space = Tuple([
        #     OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)
        # ])
        # self.available_actions = [
        #     np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
        #     for _ in range(self.NUM_AGENTS)
        # ]
        #
        # self.step_count = None

    def phi(self, x1, x2):
        return [x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)]

    def true_objective(self, net1, net2):
        """Differentiable objective in torch"""
        p1 = torch.sigmoid(net1.forward(net2))
        # p2 = torch.sigmoid(net2[[0,1,3,2,4]])
        p2 = torch.sigmoid(net2.forward(net1)[[0, 1, 3, 2, 4]])
        # p2 = torch.sigmoid(net2.forward(net1))
        p0 = (p1[0], p2[0])
        p = (p1[1:], p2[1:])
        # create initial laws, transition matrix and rewards:
        P0 = torch.stack(self.phi(*p0), dim=0).view(1,-1)
        P = torch.stack(self.phi(*p), dim=1)
        R = torch.from_numpy(self.payout_mat).view(-1,1).float()
        # the true value to optimize:
        objective = (P0.mm(torch.inverse(torch.eye(4) - self.gamma*P))).mm(R)
        return -objective


    # def reset(self):
    #     self.step_count = 0
    #     init_state = np.zeros(self.batch_size)
    #     observation = [init_state, init_state]
    #     info = [{'available_actions': aa} for aa in self.available_actions]
    #     return observation, info


    # def step(self, action):
    #     ac0, ac1 = action
    #     self.step_count += 1
    #
    #     r0 = self.payout_mat[ac0, ac1]
    #     r1 = self.payout_mat[ac1, ac0]
    #     s0 = self.states[ac0, ac1]
    #     s1 = self.states[ac1, ac0]
    #     observation = [s0, s1]
    #     reward = [r0, r1]
    #     done = (self.step_count == self.max_steps)
    #     info = [{'available_actions': aa} for aa in self.available_actions]
    #     return observation, reward, done, info
