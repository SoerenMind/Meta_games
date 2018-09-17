from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from metagames.third_party.LOLA_DiCE.envs import IPD, PD, OSPD, OSIPD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class HyParams():
    def __init__(self):
        self.diff_through_inner_opt = False
        self.lr_out = 0.1          # default: 0.2
        self.lr_in = 0.1           # default: 0.3
        # self.optim_algo = torch.optim.Adam
        self.optim_algo = torch.optim.SGD
        self.joint_optim = False    # joint or alternating GD
        # Games: PD (1 state), IPD (5 states), OSPD (1 state), OSIPD (5 states)
        self.game, self.num_states, self.net_type = 'OSPD', 1, 'OppAwareNet'    # 'OppAwareNet', 'NoInputFcNet'
        self.layer_sizes = [None, 10, 3, 3, self.num_states]    # 50 free params
        self.gamma = 0.96
        self.n_outer_opt = 1000
        self.n_inner_opt = (0,1+1)
        self.seed = 0
        # self.payout_mat = [[-2,0],[-3,-1]]  # Not implemented for IPD
        self.payout_mat = [[-2.9,0],[-3,-0.1]]
        self.plot_progress = True
        self.plot_every_n = self.n_outer_opt // 5.

hp = HyParams()
exp_name = hp.__dict__
print(exp_name)


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



class OppAwareNet(torch.nn.Module):
    """A feed-forward net with fixed parameters in the 1st layer that takes another net's parameters as input."""
    def __init__(self, diff_seed):
        super(OppAwareNet, self).__init__()
        layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(0,))
        self.w1 = torch.zeros(layer_sizes[0], layer_sizes[1]).normal_(0, 0.1).to(device).requires_grad_()
        self.b1 = torch.zeros(layer_sizes[1]).normal_(0, 0.1).to(device).requires_grad_()
        torch.manual_seed(diff_seed)    # Ensures only higher layers differ between nets
        self.w2 = torch.nn.Parameter(torch.zeros(layer_sizes[1], layer_sizes[2]).normal_(0, 0.1))
        self.b2 = torch.nn.Parameter(torch.zeros(layer_sizes[2]).normal_(0, 0.1))
        self.w3 = torch.nn.Parameter(torch.zeros(layer_sizes[2], layer_sizes[3]).normal_(0, 0.1))
        self.b3 = torch.nn.Parameter(torch.zeros(layer_sizes[3]).normal_(0, 0.1))
        self.w4 = torch.nn.Parameter(torch.zeros(layer_sizes[3], layer_sizes[4]).normal_(0, 0.1))
        self.b4 = torch.nn.Parameter(torch.zeros(layer_sizes[4]).normal_(0, 0.1))
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
        out = F.relu(out.mm(self.w1) + self.b1)
        out = F.relu(out.mm(self.w2) + self.b2)
        out = F.relu(out.mm(self.w3) + self.b3)
        out =        out.mm(self.w4) + self.b4
        return out.view(-1)


def play_LOLA(n_inner_opt):
    """Create two agents, play the game, return scores over time.
    :param n_inner_opt: number of steps opponent takes in inner loop for LOLA.
    """
    print("start iterations with", n_inner_opt, "lookaheads:")
    scores = []

    torch.manual_seed(hp.seed)
    net1 = net_name_to_class(hp.net_type)(diff_seed=1).to(device)
    torch.manual_seed(hp.seed)
    net2 = net_name_to_class(hp.net_type)(diff_seed=2).to(device)

    def LOLA_step(net1, net2_):

        # Inner optimization
        for k in range(n_inner_opt):
            if hp.diff_through_inner_opt:
                true_objective2 = game_NN.true_objective(net2_, net1)
            else:
                net1_ = deepcopy(net1)
                true_objective2 = game_NN.true_objective(net2_, net1_)

            # Grad update for NN without modules like nn.Linear
            grad2 = torch.autograd.grad(true_objective2, net2_.parameters(), create_graph=True)
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
        true_objective1 = game_NN.true_objective(net1, net2_)
        net1.optimizer.zero_grad()
        true_objective1.backward()
        net1.optimizer.step()


    # SGD loop
    for update in range(hp.n_outer_opt):
        net2_ = deepcopy(net2).to(device)
        if hp.joint_optim == True:  net1_ = deepcopy(net1).to(device)  #TODO(sorenmind): Turns parameters into tensors. Problem?
        LOLA_step(net1, net2_)
        if hp.joint_optim == False: net1_ = deepcopy(net1).to(device)
        LOLA_step(net2, net1_)

        scores = eval_and_print(scores, update, net1, net2)
        if hp.plot_progress: plot_progress(scores, n_inner_opt, update)

    return scores






"""
========================================================================================================================
========================================VARIOUS HELPER FUNCTIONS AND CLASSES ===========================================
========================================================================================================================
"""



# def test1_tensors_working(n_inner_opt):
#     """A test "play" function with single tensors instead of neural nets"""
#     print("start iterations with", n_inner_opt, "lookaheads:")
#     joint_scores = []
#
#     T1 = torch.zeros(hp.num_states, requires_grad=True)
#     T2 = torch.zeros(hp.num_states, requires_grad=True)
#     T1.optimizer = hp.optim_algo((T1,), lr=hp.lr_out)
#     T2.optimizer = hp.optim_algo((T2,), lr=hp.lr_out)
#
#     for update in range(hp.n_outer_opt):
#         def LOLA_step(T1, T2_):
#
#             # Inner optimization
#             for k in range(n_inner_opt):
#                 true_objective2 = game_tensor.true_objective(T2_, T1)
#                 grad2 = torch.autograd.grad(true_objective2, (T2_,), create_graph=True)[0]
#                 T2_ = T2_ - hp.lr_in * grad2
#
#             # Outer optimization
#             true_objective1 = game_tensor.true_objective(T1, T2_)
#             T1.optimizer.zero_grad()
#             true_objective1.backward()
#             T1.optimizer.step()
#
#         T2_ = deepcopy(T2)
#         T1_ = deepcopy(T1)
#         LOLA_step(T1, T2_)
#         LOLA_step(T2, T1_)
#
#         joint_scores = eval_and_print(joint_scores, update, T1, T2)
#
#     return joint_scores

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
#         # plt.plot(joint_scores, colors[n_inner_opt], label=str(n_inner_opt) + " lookaheads")
#         plt.xlim([0,hp.n_outer_opt])
#         line, _ = self.ax.plot(joint_scores, colors[n_inner_opt], label=str(n_inner_opt) + " lookaheads")
#     def update(self, joint_scores, n_inner_opt, update):




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
        # ax.plot(joint_scores, colors[n_inner_opt], label=str(n_inner_opt) + " lookaheads")
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
        if i in fixed_layers: continue
        n_trainable_params += (m + bias) * n
    layer_sizes[0] = n_trainable_params
    return layer_sizes


def eval_and_print(scores, update, agent1, agent2):
    # evaluate:
    # score = step(agent1, agent2)
    score = (-game.true_objective(agent1, agent2), -game.true_objective(agent2, agent1))
    # scores.append(0.5 * (score[0] + score[1]))
    scores.append(score)

    # print
    if update % 10 == 0:
        p1 = [np.round(p.item(), 2) for p in torch.sigmoid(agent1.forward(agent2))]
        p2 = [np.round(p.item(), 2) for p in torch.sigmoid(agent2.forward(agent1))]
        print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]), 'policy 1:', p1, 'policy 2:', p2)
    return scores


class SelfOutputNet(torch.nn.Module):
    """Outputs its own parameters (for non-open source IPD)"""
    def __init__(self, diff_seed=None):
        super(SelfOutputNet, self).__init__()
        self.theta = torch.nn.Parameter(torch.zeros(hp.num_states, requires_grad=True))
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    def forward(self, input=None):
        return self.theta


def net_name_to_class(name):
    name_dict = {'OppAwareNet': OppAwareNet, 'NoInputFcNet': NoInputFcNet, 'SelfOutputNet': SelfOutputNet}
    return name_dict[name]

'Create game env'
if hp.game == 'PD':
    game_NN = PD(payout_mat=hp.payout_mat, device=device)
elif hp.game == 'OSPD':
    game_NN = OSPD(payout_mat=hp.payout_mat, device=device)
elif hp.game == 'IPD':
    game_NN = IPD(hp.gamma, device=device)
elif hp.game == 'OSIPD':
    game_NN = OSIPD(hp.gamma, device=device)
game = game_NN



# plot results:
if __name__=="__main__":

    colors = ['b','c','m','r','y','g']

    for i in range(*hp.n_inner_opt):
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
