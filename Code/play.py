import torch
import torch.nn as nn
import torch.nn.functional as F
from envs import IPD, PD, OSPD
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.distributions import Bernoulli


class HyParams():
    def __init__(self):
        self.lr_out = 0.2      # default: 0.2
        self.lr_in = 0.3       # default: 0.3
        self.optim_algo = torch.optim.Adam
        # self.optim_algo = torch.optim.SGD
        self.joint_optim = False
        self.tensors = False    # False means using NNs
        self.game = 'OSPD'
        self.gamma = 0.96
        self.n_outer_opt = 70
        self.n_inner_opt = 4
        self.len_rollout = 10
        self.batch_size = 64
        self.seed = 42
        self.net_type = 'OppAwareNet'
        self.n_hidden = {1: 3, 2: 2}

hp = HyParams()

class SelfOutputNet(torch.nn.Module):
    def __init__(self):
        super(SelfOutputNet, self).__init__()
        self.theta = torch.nn.Parameter(torch.zeros(hp.num_states, requires_grad=True))
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    def forward(self, input):
        return self.theta



class OppAwareNet(torch.nn.Module):
    def __init__(self):
        super(OppAwareNet, self).__init__()
        # self.theta1 = torch.nn.Parameter(torch.zeros(hp.n_hidden[1], requires_grad=True))
        # self.theta2 = torch.nn.Parameter(torch.zeros(hp.n_hidden[2], requires_grad=True
        # self.optimizer = hp.optim_algo((self.theta2,), lr=hp.lr_out)
        self.fc1 = torch.nn.Linear(hp.n_hidden[1], hp.n_hidden[2], bias=True)
        self.fc2 = torch.nn.Linear(hp.n_hidden[2], 1, bias=True)
        self.optimizer = hp.optim_algo(self.fc2._parameters.values(), lr=hp.lr_out)

    # TODO: Make first layer shared or use random mask

    def forward(self, theta2):
        # out = torch.zeros(1, requires_grad=True)
        # Read parameters of layer [1:]
        params2 = torch.empty(0,requires_grad=True)
        for layer in list(theta2.parameters())[2:]:
            layer = layer.view(-1)
            params2 = torch.cat((params2, layer), dim=0)
        out = F.relu(self.fc1(params2))
        out = self.fc2(out)
        return out


# def test_bonus():
#     net1 = OppAwareNet()
#     net2 = OppAwareNet()
#     theta1_optimizer = torch.optim.SGD(net1.parameters(), lr=hp.lr_out)
#
#     def fake_objective_func(x1, x2):
#         """Prisoner's dilemma loss"""
#         out1, out2 = x1.forward(list(x2.parameters())[0]), x2.forward(list(x1.parameters())[0])
#         x1, x2 = torch.softmax(out1, 0), torch.softmax(out2, 0)
#         return - x1.view(1, -1).mm(A).view(-1).dot(x2)
#
#     net2_ = deepcopy(net2)  # No stop_gradient?
#
#     for k in range(gd_iter):
#         fake_objective2 = fake_objective_func(net2_, net1)
#         grad2 = torch.autograd.grad(fake_objective2, net2_.parameters(), create_graph=True)
#         for i, layer in enumerate(net2_.parameters()):
#             layer.data.sub_(lr * grad2[i])
#
#     fake_objective1 = fake_objective_func(net1, net2)
#     theta1_optimizer.zero_grad()
#     fake_objective1.backward()
#     theta1_optimizer.step()
#     return net1.theta


def play(n_inner_opt):
    print("start iterations with", n_inner_opt, "lookaheads:")
    joint_scores = []

    net1 = net_name_to_class(hp.net_type)()
    net2 = net_name_to_class(hp.net_type)()

    for update in range(hp.n_outer_opt):
        def LOLA_step(net1, net2_):

            # Inner optimization
            for k in range(n_inner_opt):
                true_objective2 = game_NN.true_objective(net2_, net1)
                grad2 = torch.autograd.grad(true_objective2, net2_.parameters(), create_graph=True)
                for i, (layer_name, layer) in enumerate(net2_._parameters.items()):
                    layer = layer - hp.lr_in * grad2[i]
                    net2_._parameters[layer_name] = layer

            # Outer optimization
            true_objective1 = game_NN.true_objective(net1, net2_)
            net1.optimizer.zero_grad()
            true_objective1.backward()
            net1.optimizer.step()

        net2_ = deepcopy(net2)
        if hp.joint_optim == True:  net1_ = deepcopy(net1)
        LOLA_step(net1, net2_)
        if hp.joint_optim == False: net1_ = deepcopy(net1)
        LOLA_step(net2, net1_)

        joint_scores = eval_and_print(joint_scores, update, net1, net2)

    return joint_scores








"""
========================================VARIOUS HELPER FUNCTIONS AND CLASSES ===========================================
"""
# def eval_func_NN(agent, agent2):
#     return agent.forward(agent2)
# def eval_func_tensor(agent, agent2):
#     return agent



def test1_tensors_working(n_inner_opt):
    print("start iterations with", n_inner_opt, "lookaheads:")
    joint_scores = []

    T1 = torch.zeros(hp.num_states, requires_grad=True)
    T2 = torch.zeros(hp.num_states, requires_grad=True)
    T1.optimizer = hp.optim_algo((T1,), lr=hp.lr_out)
    T2.optimizer = hp.optim_algo((T2,), lr=hp.lr_out)

    for update in range(hp.n_outer_opt):
        def LOLA_step(T1, T2_):

            # Inner optimization
            for k in range(n_inner_opt):
                true_objective2 = game_tensor.true_objective(T2_, T1)
                grad2 = torch.autograd.grad(true_objective2, (T2_,), create_graph=True)[0]
                T2_ = T2_ - hp.lr_in * grad2

            # Outer optimization
            true_objective1 = game_tensor.true_objective(T1, T2_)
            T1.optimizer.zero_grad()
            true_objective1.backward()
            T1.optimizer.step()

        T2_ = deepcopy(T2)
        T1_ = deepcopy(T1)
        LOLA_step(T1, T2_)
        LOLA_step(T2, T1_)

        joint_scores = eval_and_print(joint_scores, update, T1, T2)

    return joint_scores



def act(batch_states, theta, theta2):

    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta.forward(theta2))[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions


def eval_and_print(joint_scores, update, agent1, agent2):
    # evaluate:
    score = step(agent1, agent2)
    joint_scores.append(0.5 * (score[0] + score[1]))

    # print
    if update % 10 == 0:
        p1 = [p.item() for p in torch.sigmoid(agent1.forward(agent2))]
        p2 = [p.item() for p in torch.sigmoid(agent2.forward(agent1))]
        print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]), 'policy 1:', p1, 'policy 2:', p2)
    return joint_scores


def step(theta1, theta2):
    # just to evaluate progress:
    (s1, s2), _ = game.reset()
    score1 = 0
    score2 = 0
    for t in range(hp.len_rollout):
        a1, lp1 = act(s1, theta1, theta2)
        a2, lp2 = act(s2, theta2, theta2)
        (s1, s2), (r1, r2),_,_ = game.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)
    return (score1, score2)


def net_name_to_class(name):
    name_dict = {'OppAwareNet': OppAwareNet, 'SelfOutputNet': SelfOutputNet}
    return name_dict[name]

'Create game env'
if hp.tensors:
    game_tensor, hp.num_states = IPD(hp.len_rollout, hp.gamma, hp.batch_size), 5
    game = game_tensor
else:
    if hp.game == 'PD':
        game_NN, hp.num_states = PD(hp.batch_size), 1
    elif hp.game == 'OSPD':
        game_NN, hp.num_states = OSPD(hp.batch_size), 1
    elif hp.game == 'IPD':
        game_NN, hp.num_states = IPD(hp.len_rollout, hp.gamma, hp.batch_size), 5
    game = game_NN



# plot progress:
if __name__=="__main__":

    colors = ['b','c','m','r']

    for i in range(1, hp.n_inner_opt):
        torch.manual_seed(hp.seed)
        scores = np.array(test1_tensors_working(i)) if hp.tensors else np.array(play(i))
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('grad steps')
    plt.ylabel('joint score')
    plt.show(block=True)
