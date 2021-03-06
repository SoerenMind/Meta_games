"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np
import torch

from gym.spaces import Discrete, Tuple

from .common import OneHot

def phi(x1,x2):
    return [x1*x2, x1*(1-x2), (1-x1)*x2,(1-x1)*(1-x2)]

class PrisonersDilemma(gym.Env):
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 1

    def __init__(self, batch_size=1, payout_mat=[[-2,0],[-3,-1]], device=None):
        self.max_steps = 1  # or 0?
        self.batch_size = 1
        self.payout_mat = np.array(payout_mat)
        self.states = np.array([[0,0],[0,0]])
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


    # def reset(self):
    #     self.step_count = 0
    #     init_state = np.zeros(self.batch_size)
    #     observation = [init_state, init_state]
    #     info = [{'available_actions': aa} for aa in self.available_actions]
    #     return observation, info
    #
    # def step(self, action):
    #     """Actions should be defection probabilities"""
    #     ac0, ac1 = action
    #     self.step_count += 1
    #
    #     # outcome_probs = phi(ac0, ac1)
    #     outcome_probs = np.outer([1 - ac0, ac0], [1 - ac1, ac1])
    #     r0 = (self.payout_mat * outcome_probs).sum()
    #     r1 = (self.payout_mat.T * outcome_probs).sum()
    #
    #     # r0 = self.payout_mat[ac0, ac1]
    #     # r1 = self.payout_mat[ac1, ac0]
    #     s0 = self.states[ac0, ac1]
    #     s1 = self.states[ac1, ac0]
    #     observation = [s0, s1]
    #     # observation = None
    #     reward = [r0, r1]
    #     done = (self.step_count == self.max_steps)
    #     info = [{'available_actions': aa} for aa in self.available_actions]
    #     return observation, reward, done, info

    def true_objective(self, theta1, theta2):
        p1 = torch.sigmoid(theta1.forward())
        p2 = torch.sigmoid(theta2.forward())
        # create initial laws, transition matrix and rewards:
        # outcome_probs = torch.ger(torch.stack([p1, 1 - p1], dim=1).view(-1), torch.stack([p2, 1 - p2], dim=1).view(-1))
        outcome_probs = torch.ger(torch.stack([1 - p1, p1], dim=1).view(-1), torch.stack([1 -p2, p2], dim=1).view(-1))
        payout_mat = torch.from_numpy(self.payout_mat).float()
        objective = (payout_mat * outcome_probs).sum()
        return -objective
