from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse

from metagames.third_party.LOLA_DiCE.envs import IPD, PD, OSPD, OSIPD
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# tensor = torch.tensor if device == "cpu" else torch.cuda.FloatTensor
FloatTensor = torch.FloatTensor

class HyParams():
    def __init__(self):
        # Optimization
        self.diff_through_inner_opt = True
        self.weight_grad_paths = False
        self.grad_weight_self = 0.
        self.lr_out = 0.1 * (1. + self.weight_grad_paths)
        self.lr_in = 0.01 * (1. + self.weight_grad_paths)
        # self.optim_algo = torch.optim.Adam
        self.optim_algo = torch.optim.SGD
        self.joint_optim = False    # joint or alternating GD
        self.n_outer_opt = 8000
        self.n_inner_opt_range = (0, 1 + 1)

        # Game
        # Games: PD (1 state), IPD (5 states), OSPD (1 state), OSIPD (5 states)
        self.game, self.num_states, self.net_type = 'OSPD', 1, 'OppAwareNetSubspace'    # 'OppAwareNetSubspace', 'NoInputFcNet'
        self.payout_mat = [[-2.9,0],[-3,-0.1]]
        # self.payout_mat = [[-2,0],[-3,-1]]  # Not implemented for IPD
        # self.gamma = 0.96

        # Neural nets
        self.layer_sizes = [10, 10, self.num_states]    #, 3, self.num_states]
        # self.biases = True
        self.init_std = 0.1
        self.seed = 2

        self.plot_progress = False
        self.plot_every_n = self.n_outer_opt // 5.


hp = HyParams()
exp_name = [(key, hp.__dict__[key]) for key in sorted(hp.__dict__)]
print("Hyperparams: \n", exp_name)




class OppAwareNet1stFixed(torch.nn.Module):
    """A feed-forward net with fixed parameters in the 1st layer that takes another net's parameters as input."""
    def __init__(self, diff_seed):
        super(OppAwareNet1stFixed, self).__init__()
        # layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(0,))
        layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(0,), bias=False)
        self.w1 = torch.zeros(layer_sizes[0], layer_sizes[1]).normal_(0, hp.init_std).to(device).requires_grad_()
        self.b1 = torch.zeros(layer_sizes[1]).to(device).requires_grad_()
        torch.manual_seed(diff_seed)    # Ensures only higher layers differ between nets
        self.w2 = torch.nn.Parameter(torch.ones(layer_sizes[1], layer_sizes[2]).normal_(0, hp.init_std))
        # self.b2 = torch.nn.Parameter(torch.zeros(layer_sizes[2]))
        self.b2 = torch.tensor(torch.zeros(layer_sizes[2])).to(device).requires_grad_()
        # self.w3 = torch.nn.Parameter(torch.zeros(layer_sizes[2], layer_sizes[3]).normal_(0, hp.init_std))
        # # self.b3 = torch.nn.Parameter(torch.zeros(layer_sizes[3]))True
        # self.b3 = torch.tensor(torch.zeros(layer_sizes[3])).to(device).requires_grad_()
        # self.w4 = torch.nn.Parameter(torch.zeros(layer_sizes[3], layer_sizes[4]).normal_(0, hp.init_std))
        # # self.b4 = torch.nn.Parameter(torch.zeros(layer_sizes[4]))
        # self.b4 = torch.tensor(torch.zeros(layer_sizes[4])).to(device).requires_grad_()
        # self.trainable_params = [self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]
        # self.mask_w1 = torch.FloatTensor(10, 10).uniform_() > 0.8     # bit mask
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    def forward(self, net2):
        out = torch.empty(0,requires_grad=True).to(device)
        # Concatenate parameters of layer [1:]
        for param2 in list(net2.parameters()):
            param2 = param2.view(-1)
            out = torch.cat((out, param2), dim=0)
        out = out.view(1, -1)
        out = F.leaky_relu(out.mm(self.w1) + self.b1, negative_slope=1/5.5)
        out = out.mm(self.w2) + self.b2
        # out = F.leaky_relu(out.mm(self.w3) + self.b3, negative_slope=1/5.5)
        # out =              out.mm(self.w4) + self.b4
        return out.view(-1)




class OppAwareNetSubspace(torch.nn.Module):
    """A feed-forward net that takes another net's parameters as input, with parameters trained in a subspace."""
    def __init__(self, diff_seed):
        super(OppAwareNetSubspace, self).__init__()
        LS = hp.layer_sizes
        n_free_params = LS[0]
        n_direct_params = sum([(m + hp.biases) * n for (m, n) in zip(hp.layer_sizes[:-1], hp.layer_sizes[1:])])
        exp_row_norm = np.sqrt(n_direct_params)

        self.subs_params = torch.nn.Parameter(torch.zeros(n_free_params))

        # Initial params remain unchanged; define offset of subspace
        torch.manual_seed(diff_seed)    # Ensures different init
        self.w1_init = FloatTensor(LS[0], LS[1]).normal_(0, hp.init_std)
        self.b1_init = FloatTensor(LS[1]).fill_(0)
        # self.b1_init = FloatTensor(LS[1]).normal_(0, hp.init_std)
        self.w2_init = FloatTensor(LS[1], LS[2]).normal_(0, hp.init_std)
        self.b2_init = FloatTensor(LS[2]).fill_(0)
        # self.b2_init = FloatTensor(LS[2]).normal_(0, hp.init_std)

        torch.manual_seed(hp.seed)      # Ensures same subspace
        # Sample a random ~orthonormal matrix consisting these submatrices
        # TODO: Test column norm
        self.w1_subspace = FloatTensor(LS[0], LS[0], LS[1]).normal_(0, 1 / exp_row_norm)
        self.b1_subspace = FloatTensor(LS[0], LS[1]).normal_(0, 1 / exp_row_norm)
        self.w2_subspace = FloatTensor(LS[0], LS[1], LS[2]).normal_(0, 1 / exp_row_norm)
        self.b2_subspace = FloatTensor(LS[0], LS[2]).normal_(0, 1 / exp_row_norm)

        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
        # self.mask_w1 = torch.FloatTensor(10, 10).uniform_() > 0.8     # bit mask
    def forward(self, net2):
        # params2_flat = torch.cat([param.view(-1) for param in net2.parameters()])
        assert len(list(net2.parameters())) == 1
        # Compute direct params = initial params + subs_params dot "random roughly orthonormal matrix"
        w1 = self.w1_init + torch.einsum("i,ijk->jk", (self.subs_params, self.w1_subspace))   # requires grad?
        b1 = self.b1_init + torch.einsum("i,ij->j", (self.subs_params, self.b1_subspace))
        w2 = self.w2_init + torch.einsum("i,ijk->jk", (self.subs_params, self.w2_subspace))
        b2 = self.b2_init + torch.einsum("i,ij->j", (self.subs_params, self.b2_subspace))

        # Note initial input is always zero
        subs_params2 = net2.subs_params
        out = subs_params2.view(1, -1)
        out = F.leaky_relu(out.mm(w1) + b1, negative_slope=1/5.5)
        out = out.mm(w2) + b2
        # out = F.leaky_relu(out.mm(w3) + self.b3, negative_slope=1/5.5)
        # out =              out.mm(w4) + self.b4
        return out.view(-1)


def play_LOLA(n_inner_opt):
    """Create two agents, play the game, return score        if update == 500:
            x = 1
            passs over time.
    :param n_inner_opt: number of steps opponent takes in inner loop for LOLA.
    """
    print("start iterations with", n_inner_opt, "lookaheads:")
    scores = []

    torch.manual_seed(hp.seed)
    net1 = name_dict[hp.net_type](diff_seed=1).to(device)
    torch.manual_seed(hp.seed)
    net2 = name_dict[hp.net_type](diff_seed=2).to(device)

    objective = game.make_weighted_grad_objective(hp.grad_weight_self) if hp.weight_grad_paths else game.true_objective

    def LOLA_step(net1, net2_):

        # Inner optimization
        for k in range(n_inner_opt):



            if hp.diff_through_inner_opt:
                objective2 = objective(net2_, net1)
            else:
                net1_ = deepcopy(net1)
                objective2 = objective(net2_, net1_)

            # Grad update for NN without modules like nn.Linear
            grad2 = torch.autograd.grad(objective2, net2_.parameters(), create_graph=True)
            assert len(list(net2_.parameters())) == len(net2_._parameters.items()) == len(grad2) # Ensure no params are missed
            for i, (param_name, param) in enumerate(net2_._parameters.items()):
                net2_._parameters[param_name] = param - hp.lr_in * grad2[i]

            # Grad update for NN with only modules
            # grad2 = {}
            # for i, (mod_name, mod) in enumerate(net2_._modules.items()):
            #     # list(mod._parameters.values())
            #     grad2[mod_name] = torch.autograd.grad(true_objective2, mod.parameters(), create_graph=True)
            # for i, (mod_name, mod) in enumerate(net2_._modules.items()):
            #     # Empty:
            #     name_param_list2 = [(name, param) for (name, param) in mod._parameters.items() if param is not None]
            #     for i, (param_name, param) in enumerate(name_param_list2):
            #         mod._parameters[param_name] = param - hp.lr_in * grad2[mod_name][i]



        # Outer optimization
        objective1 = objective(net1, net2_)
        net1.optimizer.zero_grad()
        objective1.backward()
        net1.optimizer.step()


    # SGD loop
    for update in range(hp.n_outer_opt):
        scores = eval_and_print(scores, update, net1, net2)
        if hp.plot_progress:
            plot_progress(scores, n_inner_opt, update)

        net2_ = deepcopy(net2).to(device)
        if hp.joint_optim == True:
            net1_ = deepcopy(net1).to(device)  #TODO(sorenmind): Turns parameters into tensors. Problem?
        LOLA_step(net1, net2_)
        if hp.joint_optim == False:
            net1_ = deepcopy(net1).to(device)
        LOLA_step(net2, net1_)

    return scores






"""
========================================================================================================================
========================================VARIOUS HELPER FUNCTIONS AND CLASSES ===========================================
========================================================================================================================
"""



def get_net_params_dict(net):
    params_dict = [param for mod in net._modules.values() for param in mod._parameters.items()] \
                  + list(net._parameters.items())  # Gets parameters and module parameters
    return dict((name, param) for (name, param) in params_dict if param is not None)

# class Plot():
#     def __init__(self):
#         self.colors = ['b','c','m','r','y','g']
#         plt.ion()
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(111)
#         # plt.legend()
#         plt.xlabel('grad steps')
#         plt.ylabel('player scores')
#         # plt.plot(joint_scores, colors[n_inner_opt_range], label=str(n_inner_opt_range) + " lookaheads")
#         plt.xlim([0,hp.n_outer_opt])
#         line, _ = self.ax.plot(joint_scores, colors[n_inner_opt_range], label=str(n_inner_opt_range) + " lookaheads")
#     def update(self, joint_scores, n_inner_opt_range, update):




def plot_progress(joint_scores, n_inner_opt, update, line=None):
    colors = ['b','c','m','r','y','g']
    if update == 0:
        plt.ion()
        plt.xlabel('grad steps')
        plt.ylabel('player scores')
        plt.xlim([0,hp.n_outer_opt])
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.legend()
        # ax.plot(joint_scores, colors[n_inner_opt_range], label=str(n_inner_opt_range) + " lookaheads")

    if update % hp.plot_every_n == 0:
        plt.plot(joint_scores, colors[n_inner_opt], label=str(n_inner_opt) + " lookaheads")
        # plt.show(block=False)
        plt.draw()
        plt.pause(1.0 / 6000.0)
        # plt.show(block=True)



def calc_input_dim(layer_sizes, fixed_layers=[0], bias=True):
    """Input: List of layer sizes with None for the input dimension to be calculated.
    E.g. [None, 20, 10, 1]. Replaces None."""
    n_trainable_params = 0
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if i in fixed_layers:
            continue
        n_trainable_params += (m + bias) * n
    layer_sizes[0] = n_trainable_params
    return layer_sizes


def grad_norm(grad):
    """Input: iterable of gradients for each layer. Out: gradient norm."""
    flat_grad = torch.cat([layer_grad.view(-1) for layer_grad in grad], dim=0)
    return torch.norm(flat_grad)


def eval_and_print(scores, update, agent1, agent2):
    # evaluate:
    score = (-game.true_objective(agent1, agent2), -game.true_objective(agent2, agent1))
    scores.append(score)

    # print
    if update % 10 == 0:
        [grad1, grad2] = [torch.autograd.grad(game.true_objective(agent1, agent2), agent1.parameters())
                          for agent1, agent2 in [[agent1, agent2], [agent2, agent1]]]
        p1 = [np.round(p.item(), 3) for p in torch.sigmoid(agent1.forward(agent2))]
        p2 = [np.round(p.item(), 3) for p in torch.sigmoid(agent2.forward(agent1))]
        print('update', update, 'score (%.5f,%.5f)' % (score[0], score[1]), 'policy 1:', p1, 'policy 2:', p2,
              # 'param 1: %.5f' % np.array(list(agent1.parameters()))[-1][0][:3].item(),
              'gradnorms x1000: %.4f, %.4f' %(1000 * grad_norm(grad1), 1000 * grad_norm(grad2))
              )
    return scores


class SelfOutputNet(torch.nn.Module):
    """Outputs its own parameters (for non-open source IPD)"""
    def __init__(self, diff_seed=None):
        super(SelfOutputNet, self).__init__()
        self.theta = torch.nn.Parameter(torch.zeros(hp.num_states, requires_grad=True))
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    def forward(self, input=None):
        return self.theta


class NoInputFcNet(torch.nn.Module):
    """A feed-forward net that doesn't see its opponent."""
    def __init__(self, diff_seed=None):
        super(NoInputFcNet, self).__init__()
        torch.manual_seed(diff_seed)
        self.fc1 = torch.nn.Linear(1, hp.layer_sizes[1], bias=True)
        self.fc2 = torch.nn.Linear(hp.layer_sizes[1], hp.layer_sizes[2], bias=True)
        self.fc3 = torch.nn.Linear(hp.layer_sizes[2], hp.layer_sizes[3], bias=True)
        self.fake_input = torch.ones(1).to(device)
        # list(self.fc1._parameters.values())[0].data = torch.zeros([5,1])
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    # def forward_one_layer(self, ignored_input=None):
    #     return torch.tensor([1.], requires_grad=True)
    def forward(self, ignored_input=None):
        # input = torch.ones(1).to(device)
        # out = F.relu(self.fc1(input))
        out = F.relu(self.fc1(self.fake_input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# def str_to_var(name):
name_dict = {'OppAwareNet1stFixed': OppAwareNet1stFixed,
             'OppAwareNetSubspace': OppAwareNetSubspace,
             'NoInputFcNet': NoInputFcNet,
             'SelfOutputNet': SelfOutputNet}
    # return name_dict[name]


'Create game env'
if hp.game == 'PD':
    game = PD(payout_mat=hp.payout_mat, device=device)
elif hp.game == 'OSPD':
    game = OSPD(payout_mat=hp.payout_mat, device=device)
elif hp.game == 'IPD':
    game = IPD(hp.gamma, device=device)
elif hp.game == 'OSIPD':
    game = OSIPD(hp.gamma, device=device)
else: raise ValueError('Unknown game')


# plot results:
if __name__=="__main__":

    colors = ['b','c','m','r','y','g']

    for i in range(*hp.n_inner_opt_range):
        torch.manual_seed(hp.seed)
        scores = np.array(play_LOLA(i))
        scores_copy = scores
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('grad steps')
    plt.ylabel('joint score')
    plt.show(block=True)





# class OppAwareNet(torch.nn.Module):
#     """A feed-forward net with fixed parameters in the 1st layer that takes another net's parameters as input."""
    # def __init__(self, diff_seed):
    #     super(OppAwareNet, self).__init__()
    #     layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(1,))
    #     self.layers, trainable_params = nn.ModuleList(), []
    #     # for i in range(len(layer_sizes[:-1])):
    #     #     layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=True)
    #     #     if i==0: torch.manual_seed(diff_seed)
    #     #     else: trainable_params += list(layer.parameters())
    #     #     self.layers.append(layer)
    #     # self.fc1 = torch.nn.Linear(layer_sizes[0], layer_sizes[1], bias=True)   # init stdv is 1./out_features
    #     self.optimizer = hp.optim_algo(trainable_params, lr=hp.lr_out)
    #
    # def forward(self, net2):
    #     # Read parameters of layer [1:]
    #     out = torch.empty(0,requires_grad=True).to(device)
    #     for layer in list(net2.parameters())[2:]:
    #         layer = layer.view(-1)
    #         out = torch.cat((out, layer), dim=0)
    #     # Feed parameters of pl2 through net
    #     for layer in self.layers:
    #         out = layer(out)
    #         if not layer == self.layers[-1]:
    #             out = F.relu(out)
    #     return out
