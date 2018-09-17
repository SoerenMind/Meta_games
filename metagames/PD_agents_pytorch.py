# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torch.tensor as tt
import torch.nn.functional as F
from torch.distributions import Bernoulli
from copy import deepcopy
from collections import OrderedDict

from envs import IPD, PD, CG




class Hp():
    def __init__(self):
        self.lr_out = 0.2    # default: 0.2
        self.lr_in = 0.3     # default: 0.3
        self.optim_algo = torch.optim.Adam
        # self.optim_algo = torch.optim.SGD
        self.optim_rhythm = 'alt'
        # self.optim_rhythm = 'joint'
        self.gamma = 0.96
        self.n_update = 70
        self.n_lookahead = 4
        self.len_rollout = 10
        self.batch_size = 64
        self.seed = 42
        self.subs_dim = 10
        self.agent_type = 'NN'


hp = Hp()

def eval_func(input):
    if hp.agent_type == 'NN':
        return input.forward()
    else:
        return input


game, hp.num_states = IPD(hp.len_rollout, hp.gamma, eval_func, hp.batch_size), 5
# game, hp.num_states = PD(), 1
# game, hp.num_states = CG(hp.len_rollout, hp.batch_size, grid_size=2), 5


# class FcNet(nn.Module):
#     """A feed forward net."""
#     def __init__(self):
#         super(FcNet, self).__init__()
#         self.fc1 = nn.Linear(hp.subs_dim, 20)
#         self.fc2 = nn.Linear(20, hp.num_states)
#
#     def forward(self, x=torch.ones(hp.subs_dim)):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))     # This could lead to zero gradient since 1 dim!
#         return x



class FcNet(nn.Module):
    """A feed forward net."""
    def __init__(self):
        super(FcNet, self).__init__()
        self.five = nn.Parameter(torch.zeros(5, requires_grad=True))

    def forward(self, x=torch.ones(hp.subs_dim)):
        return self.five
        # x = F.relu(self.fc2(x))     # This could lead to zero gradient since 1 dim!




def act(batch_states, theta):

    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(eval_func(theta))[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions

def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    if type(theta) == FcNet:
        theta = theta.parameters()
    grad_objective = torch.autograd.grad(objective, theta, create_graph=True)
    return grad_objective

def step(theta1, theta2):
    # just to evaluate progress:
    (s1, s2), _ = game.reset()
    score1 = 0
    score2 = 0
    for t in range(hp.len_rollout):
        a1, lp1 = act(s1, theta1)
        a2, lp2 = act(s2, theta2)
        (s1, s2), (r1, r2),_,_ = game.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)
    return (score1, score2)


class tt(nn.Parameter):
    pass
    # def forward(self):
    #     return self,
    # def backward(self, x):
    #     return x
    # def parameters(self):
    #     return self



class Agent():
    def __init__(self, theta=None):
        # init theta and its optimizer
        # self.theta = tt(torch.zeros(hp.num_states, requires_grad=True))
        # self.theta = tt(np.zeros(hp.num_states)).float().requires_grad_()

        # self.theta = tt(torch.zeros(hp.num_states, requires_grad=True))   # Parameter unnecesary
        # self.theta_optimizer = hp.optim_algo((self.theta,),lr=hp.lr_out)
        self.theta = FcNet()
        self.theta_optimizer = hp.optim_algo(self.theta.parameters(),lr=hp.lr_out)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def in_lookahead(self, other_theta):
        other_objective = game.true_objective(other_theta, self.theta)
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta):
        objective = game.true_objective(self.theta, other_theta)
        self.theta_update(objective)




def play(agent1, agent2, n_lookaheads):
    print("start iterations with", n_lookaheads, "lookaheads:")
    joint_scores = []
    for update in range(hp.n_update):

        'Alt using nets'
        if hp.optim_rhythm == 'alt':
            # theta2_ = FcNet()
            # theta2_.load_state_dict(agent2.theta.state_dict())
            # theta2_.parameters() = [p.detach() for p in deepcopy(agent2.theta.parameters())]    # copy_net(agent2.theta)
            # theta2_ = ReturnParameter(deepcopy(agent2.theta))
            theta2_ = deepcopy(agent2.theta)

            # theta2_ = tt(torch.tensor(agent2.theta.detach(), requires_grad=True))     # tt Doesnt matter
            # theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)
            for k in range(n_lookaheads):
                grad2 = agent1.in_lookahead(theta2_)
                with torch.no_grad():
                    for i, layer in enumerate(theta2_.parameters()):
                        layer.sub_(hp.lr_in * grad2[i])
                        # layer.data = layer.data - hp.lr_in * grad2[i]
                        # layer = layer - hp.lr_in * grad2[i]     # indexing okay? No effect on parameters() to re-declare layer?

                        # # TODO: Check if this gradient is non-zero
                        # other_objective = game.true_objective(theta2_, agent1.theta)
                        # grad2 = get_gradient(other_objective, theta2_.parameters()[i])
                        # layer = layer - hp.lr_in * grad2

                'Only updating one layer'
                # other_objective = game.true_objective(theta2_, agent1.theta)
                # layer1 = list(theta2_.parameters())[-1]
                # gradlayer1 = get_gradient(other_objective, layer1)
                # # layer1 = layer1 - hp.lr_in * gradlayer1  # TODO: does this update anything? No. How about in the loop above?
                # layer1.data = layer1.data - hp.lr_in * gradlayer1   # Does update layer but no effect on graph
                # # layer1.sub_(hp.lr_in * gradlayer1)

                'Using list of tensors'
                # other_objective = game.true_objective(theta2_, agent1.theta)
                # theta2_list = list(theta2_.parameters())
                # theta2_list = [layer - hp.lr_in * get_gradient(other_objective, layer) for layer in theta2_list]
                # theta2_ = tt(theta2_ - hp.lr_in * grad2)  # Maybe parameter creation stops gradient flow!
                # theta2_ = theta2_ - hp.lr_in * grad2
                # theta2_.sub_(hp.lr_in * grad2)
                # Predicting own grad steps:
                # grad1_new = torch.autograd.grad(game.true_objective(agent1.theta, theta2_), (agent1.theta), create_graph=True)[0]
                # # objective_1 = game.true_objective(agent1.theta, theta2_)
                # # grad1_new = torch.autograd.grad(objective_1, (agent1.theta), create_graph=True)[0]
                # # grad1_old = torch.autograd.grad(game.true_objective(theta1_, theta2_), (agent1.theta), create_graph=True)[0]
                # # grad1 = agent2.in_lookahead(theta1_)
                # # agent1.theta -= hp.lr_in * grad1_old
                # agent1.theta = agent1.theta - hp.lr_in * grad1_new      # Set some gradients to zero?

            agent1.out_lookahead(theta2_)

            theta1_ = deepcopy(agent1.theta)

            # theta1_ = ReturnParameter(deepcopy(agent1.theta))
            # theta1_ = tt(torch.tensor(agent1.theta.detach(), requires_grad=True))     # tt Doesnt matter
            # theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
            for k in range(n_lookaheads):
                grad1 = agent2.in_lookahead(theta1_)
                with torch.no_grad():
                    for i, layer in enumerate(theta1_.parameters()):
                        layer.sub_(hp.lr_in * grad1[i])
                    # theta1_ = tt(theta1_ - hp.lr_in * grad1)
                    # theta1_ = theta1_ - hp.lr_in * grad1

            agent2.out_lookahead(theta1_)


        # if hp.optim_rhythm == 'alt':
        #     theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)
        #     for k in range(n_lookaheads):
        #         grad2 = agent1.in_lookahead(theta2_)
        #         theta2_ = theta2_ - hp.lr_in * grad2
        #         # Predicting own grad steps:
        #         # grad1_new = torch.autograd.grad(game.true_objective(agent1.theta, theta2_), (agent1.theta), create_graph=True)[0]
        #         # # objective_1 = game.true_objective(agent1.theta, theta2_)
        #         # # grad1_new = torch.autograd.grad(objective_1, (agent1.theta), create_graph=True)[0]
        #         # # grad1_old = torch.autograd.grad(game.true_objective(theta1_, theta2_), (agent1.theta), create_graph=True)[0]
        #         # # grad1 = agent2.in_lookahead(theta1_)
        #         # # agent1.theta -= hp.lr_in * grad1_old
        #         # agent1.theta = agent1.theta - hp.lr_in * grad1_new      # Set some gradients to zero?
        #
        #     agent1.out_lookahead(theta2_)
        #
        #     theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
        #     for k in range(n_lookaheads):
        #         grad1 = agent2.in_lookahead(theta1_)
        #         theta1_ = theta1_ - hp.lr_in * grad1
        #     agent2.out_lookahead(theta1_)

        if hp.optim_rhythm == 'joint':
            'Joint optimization'
            theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
            theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)

            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                grad2 = agent1.in_lookahead(theta2_)
                grad1 = agent2.in_lookahead(theta1_)
                # update other's theta
                theta2_ = theta2_ - hp.lr_in * grad2
                theta1_ = theta1_ - hp.lr_in * grad1

            # update own parameters from out_lookahead:
            # Is this doing joint or alternating updates?
            agent1.out_lookahead(theta2_)
            agent2.out_lookahead(theta1_)

        # evaluate:
        score = step(agent1.theta, agent2.theta)
        joint_scores.append(0.5*(score[0] + score[1]))

        # print
        if update%10==0:
            p1 = [p.item() for p in torch.sigmoid(eval_func(agent1.theta))]
            p2 = [p.item() for p in torch.sigmoid(eval_func(agent2.theta))]
            # print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]), 'policy (%.3f,%.3f)' % (p1[0], p2[0]))
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]), 'policy 1:', p1, 'policy 2:', p2)


    return joint_scores

# plot progress:
if __name__=="__main__":

    colors = ['b','c','m','r']

    for i in range(hp.n_lookahead):
        torch.manual_seed(hp.seed)
        scores = np.array(play(Agent(), Agent(), i))
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('grad steps')
    plt.ylabel('joint score')
    plt.show()

# copy other's parameters:

# set_requires_grad(agent1.theta, False)

# Using my own objective:
# grad2 = agent1.in_lookahead(theta2_)    # Gradient of agent 2's objective
# pl1_LOLA_objective = game.true_objective(agent1.theta, theta2_ - hp.lr_in * grad2)
# agent1.theta_update(pl1_LOLA_objective)
#
# theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
#
# grad1 = agent2.in_lookahead(theta1_)    # Gradient of agent 1's objective
# pl2_LOLA_objective = game.true_objective(agent2.theta, theta1_ - hp.lr_in * grad1)
# agent2.theta_update(pl2_LOLA_objective)
#
# def get_gradient(objective, theta):
#     # create differentiable gradient for 2nd orders:
#     grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)[0]
#     return grad_objective
#
# def theta_update(self, objective):
#     self.theta_optimizer.zero_grad()
#     objective.backward(retain_graph=True)
#     self.theta_optimizer.step()
#
# def in_lookahead(self, other_theta):
#     other_objective = game.true_objective(other_theta, self.theta)
#     grad = get_gradient(other_objective, other_theta)
#     return grad
#
# def out_lookahead(self, other_theta):
#     objective = game.true_objective(self.theta, other_theta)
#     self.theta_update(objective)