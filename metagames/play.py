#!/usr/bin/env pythonr
"""Script that runs agents and their optimization in a variety of prisoner's dilemma
type games. In some of these games, agents take each other's parameters as input."""
from copy import deepcopy
import datetime
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse

from metagames.third_party.LOLA_DiCE.envs import IPD, PD, OSPD, OSIPD
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
if device == "cpu":
      from torch import FloatTensor
else: from torch.cuda import FloatTensor


def parse_args(args=None):
    """Parse command-line arguments. Some extra args are calculated at the end from the passed-in args.
    Args:
        args: A list of argument strings to use instead of sys.argv.
    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    par = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optimization
    par.add_argument('--dont-diff-through-inner-opt', action='store_true')
    par.add_argument('--weight-grad-paths', type=int, nargs=2, default=(False, None),
                     metavar='BOOL, 0<=FLOAT,=1',
                     help='1st arg: bool to give 0 < weight < 1 to gradient through self and 1-weight \ '
                          'through opponent. 2nd arg: the weight.')
    par.add_argument('--lr-out', type=float, default=0.01, help='Learning rate for regular optimization (outer loop)')
    par.add_argument('--lr-in', type=float, default=0.1, help='Learning rate for LOLA optimization (inner loop)')
    par.add_argument('--optim-algo', default='SGD', metavar='SGD or Adam')
    par.add_argument('--joint-optim', action='store_true', default=False, help='Joint descent of all agents')
    par.add_argument('--n-outer-opt', type=int, default=20000, help='N training steps')
    par.add_argument('--n-inner-opt-range', nargs=2, type=int, metavar=('LOW', 'HIGH'), default=(0, 2),
                     help='(Non-inclusive) range of LOLA inner opt steps. Each one will be plotted.')

    # Game env
    par.add_argument('--game', type=str, default='OSPD', help='Games: PD, IPD, OSPD, OSIPD. OS stands for open-source.')
    par.add_argument('--net-type', default='OppAwareNetSubspace', help='OppAwareNetSubspace or OppAwareNet1stFixed')
    par.add_argument('--DD-val', type=float, default=-2.9, help='Reward for mutual defection in PD. -2 is typical.')
    par.add_argument('--CC-val', type=float, default=-0.1, help='Reward for mutual cooperation in PD. -1 is typical.')
    par.add_argument('--gamma', type=float, default=0.96, help='discount factor for iterated games')

    # Neural nets
    par.add_argument('--layer-sizes', nargs='+', type=int, default=(10, 10), help='input and hidden layers')
    par.add_argument('--init-std', type=float, default=0.1, help='Initialization std for net weights')
    par.add_argument('--seed', type=int, default=0)
    par.add_argument('--biases', type=int, default=1, choices=[0, 1], help='Makes net with biases')
    par.add_argument('--biases-init', choices=['-1', '0', '1', 'normal'], default='normal', help='initialize biases to...')
    # TODO(sorenmind): remove
    par.add_argument('--layers-wo-bias', nargs='*', type=int, default=[], help='List of layer numbers without biases. Starts at 1!')

    # Other
    par.add_argument('--exp-group-name', default='no_name')
    par.add_argument('--foldername', default='None', help='If None, different time-based folder for each run')
    par.add_argument('--independent-vars', nargs='*', default=[], help='Names of independent hyperparams. Given in graph title.')
    par.add_argument('--plot-progress', action='store_true', help='Plot scores during training AND after')
    par.add_argument('--plot-every-n', type=int, help='If plotting progress, plot every N steps')

    args = par.parse_args(args)

    # Adjust learning rate
    if args.weight_grad_paths[0]:
        args.lr_out *= 2
        args.lr_in *= 2

    args.payout_mat = [[float(args.DD_val), 0.], [-3, float(args.CC_val)]]
    args.optim_algo = str_to_var(args.optim_algo)
    args.num_states = str_to_var((args.game, 'num_states'))
    args.layer_sizes = list(args.layer_sizes) + [args.num_states]
    args.start_time = str(datetime.datetime.now())[:-7]

    return args


def play_LOLA(n_inner_opt, hp):
    """Create two agents, play the game, return score        if update == 500:
            x = 1
            passs over time.
    :param n_inner_opt: number of steps opponent takes in inner loop for LOLA.
    """
    print("start iterations with", n_inner_opt, "lookaheads:")
    scores = []

    game = get_game(hp)

    net1 = str_to_var(hp.net_type)(hp, diff_seed=hp.seed + 1).to(device)
    net2 = str_to_var(hp.net_type)(hp, diff_seed=hp.seed + 2).to(device)

    if hp.weight_grad_paths[0]:
        objective = game.make_weighted_grad_objective(hp.grad_weight_self[1])
    else:
        objective = game.true_objective

    def LOLA_step(net1, net2_):
        # Inner optimization
        for k in range(n_inner_opt):
            if hp.dont_diff_through_inner_opt:
                net1_ = deepcopy(net1)
                objective2 = objective(net2_, net1_)
            else:
                objective2 = objective(net2_, net1)

            # Grad update for NN without modules like nn.Linear
            grad2 = torch.autograd.grad(objective2, net2_.parameters(), create_graph=True)
            assert len(list(net2_.parameters())) == len(net2_._parameters.items()) == len(grad2) # Ensure no params are missed
            for i, (param_name, param) in enumerate(net2_._parameters.items()):
                net2_._parameters[param_name] = param - hp.lr_in * grad2[i]

        # Outer optimization
        objective1 = objective(net1, net2_)
        net1.optimizer.zero_grad()
        objective1.backward()
        net1.optimizer.step()

    # SGD loop
    for update in range(hp.n_outer_opt):
        scores = eval_and_print(game, scores, update, net1, net2)
        if hp.plot_progress:
            plot_progress(scores, n_inner_opt, hp.n_outer_opt, hp.plot_every_n, update)
        net2_ = deepcopy(net2).to(device)
        if hp.joint_optim == True:
            net1_ = deepcopy(net1).to(device)
        LOLA_step(net1, net2_)
        if hp.joint_optim == False:
            net1_ = deepcopy(net1).to(device)
        LOLA_step(net2, net1_)
    return scores


class OppAwareNetSubspace(torch.nn.Module):
    """A feed-forward net that takes another net's parameters as input, with parameters trained in a subspace."""
    def __init__(self, hp, diff_seed):
        super(OppAwareNetSubspace, self).__init__()
        LS = hp.layer_sizes
        n_free_params = LS[0]
        n_direct_params = sum([(m + hp.biases) * n for (m, n) in zip(hp.layer_sizes[:-1], hp.layer_sizes[1:])])
        exp_row_norm = np.sqrt(n_direct_params)

        self.subs_params = torch.nn.Parameter(torch.zeros(n_free_params))

        # Initial params remain unchanged; define offset of subspace
        torch.manual_seed(diff_seed)    # Ensures different init for net
        self.w1_init = FloatTensor(LS[0], LS[1]).normal_(0, hp.init_std)
        self.b1_init = FloatTensor(LS[1])
        self.w2_init = FloatTensor(LS[1], LS[2]).normal_(0, hp.init_std)
        self.b2_init = FloatTensor(LS[2])
        init_biases([self.b1_init, self.b2_init], hp)

        torch.manual_seed(hp.seed)      # Ensures same subspace
        # Sample a random ~orthonormal matrix consisting of the following submatrices
        self.w1_subspace = FloatTensor(LS[0], LS[0], LS[1]).normal_(0, 1 / exp_row_norm)
        self.b1_subspace = FloatTensor(LS[0], LS[1])
        self.w2_subspace = FloatTensor(LS[0], LS[1], LS[2]).normal_(0, 1 / exp_row_norm)
        self.b2_subspace = FloatTensor(LS[0], LS[2])
        init_bias_subspaces([self.b1_subspace, self.b2_subspace], hp, exp_row_norm)

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


class OppAwareNet1stFixed(torch.nn.Module):
    """A feed-forward net with fixed parameters in the 1st layer that takes another net's parameters as input."""
    # TODO(sorenmind): Doesn't use hyperparams affecting biases
    def __init__(self, hp, diff_seed):
        super(OppAwareNet1stFixed, self).__init__()
        # layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(0,))
        layer_sizes = calc_input_dim(hp.layer_sizes, fixed_layers=(0,), bias=False)
        torch.manual_seed(hp.seed)
        # TODO(sorenmind): Make tensors parameters again
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




def plot_progress(joint_scores, n_inner_opt, n_outer_opt, plot_every_n, update):
    colors = ['b','c','m','r','y','g']
    if update == 0:
        plt.ion()
        plt.xlabel('grad steps')
        plt.ylabel('player scores')
        plt.xlim([0, n_outer_opt])
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.legend()
        # ax.plot(joint_scores, colors[n_inner_opt_range], label=str(n_inner_opt_range) + " lookaheads")

    if update % plot_every_n == 0:
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


def concat_params(net):
    return torch.cat([param.view(-1) for param in list(net.parameters())])


def eval_and_print(game, scores, update, agent1, agent2):
    # evaluate:
    score = (-game.true_objective(agent1, agent2), -game.true_objective(agent2, agent1))
    scores.append(score)

    # print
    if update % 25 == 0:
        [grad1, grad2] = [torch.autograd.grad(game.true_objective(agent1, agent2), agent1.parameters())
                          for agent1, agent2 in [[agent1, agent2], [agent2, agent1]]]
        p1 = [np.round(p.item(), 3) for p in torch.sigmoid(agent1.forward(agent2))]
        p2 = [np.round(p.item(), 3) for p in torch.sigmoid(agent2.forward(agent1))]
        print('update', update, 'score (%.5f,%.5f)' % (score[0], score[1]), 'policy 1:', p1, 'policy 2:', p2,
              # 'param 1: %.5f' % np.array(list(agent1.parameters()))[-1][0][:3].item(),
              'gradnorms x1000: %.4f, %.4f' %(1000 * grad_norm(grad1), 1000 * grad_norm(grad2)),
              'Max param 1, 2: %.2f, %.2f:' % (concat_params(agent1).abs().max(), concat_params(agent2).abs().max())
              )
    return scores


class SelfOutputNet(torch.nn.Module):
    """Outputs its own parameters (for non-open source IPD)"""
    def __init__(self, hp, diff_seed=None):
        super(SelfOutputNet, self).__init__()
        self.theta = torch.nn.Parameter(torch.zeros(hp.num_states, requires_grad=True))
        self.optimizer = hp.optim_algo(self.parameters(), lr=hp.lr_out)
    def forward(self, input=None):
        return self.theta


class NoInputFcNet(torch.nn.Module):
    """A feed-forward net that doesn't see its opponent."""
    def __init__(self, hp, diff_seed=None):
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


def str_to_var(name):
    """Maps some strings passed in as cmd line arguments to some variables."""
    name_dict = {'OppAwareNet1stFixed': OppAwareNet1stFixed,
                 'OppAwareNetSubspace': OppAwareNetSubspace,
                 'NoInputFcNet': NoInputFcNet,
                 'SelfOutputNet': SelfOutputNet,
                 'SGD': torch.optim.SGD,
                 'Adam': torch.optim.Adam,
                 ('PD', 'num_states'): 1,
                 ('OSPD', 'num_states'): 1,
                 ('IPD', 'num_states'): 5,
                 ('OSIPD', 'num_states'): 5}
    return name_dict[name]


def save_plot(hp, testing=False):
    """Warning: If there is an independent variable that isn't in the dict below, it's graph file
    will be overwritten and only one setting is saved.

    -testing: Saves empty plot and deletes it to see if it crashes."""
    hyperparams_in_filename = {'exp-group-name=': str(hp.exp_group_name),
               # 'diff-through-inner-opt=': str(hp.dont_diff_through_inner_opt),
               'lr-out=': str(hp.lr_out).replace('.',','),
               'lr-in=': str(hp.lr_in).replace('.',','),
               'joint-optim=': str(int(hp.joint_optim)),
               # 'n-inner-opt-range=': str(hp.n_inner_opt_range),
               'game=': str(hp.game),
               # 'net-type=': str(hp.net_type),
               'DD-val=': str(hp.DD_val).replace('.', ','),
               'CC-val=': str(hp.CC_val).replace('.', ','),
               'layer-sizes=': str(hp.layer_sizes),
               'seed=': str(hp.seed),
               'biases=': str(hp.biases),
               'biases_init=': str(hp.biases_init),
               'layers_wo_biases=': str(hp.layers_wo_bias)
               }
    filename = ','.join([key + str(val) for key, val in hyperparams_in_filename.items()])
    plt.title(make_plot_title(hp))

    if hp.foldername != 'None':
        foldername = hp.foldername
    else:
        foldername = hp.start_time + ': ' + hp.exp_group_name
    pathname = '../data/graphs/' + foldername + '/'
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    plt.savefig(pathname + filename)
    if testing:
        os.remove(pathname + filename + '.png')


def make_plot_title(hp):
    """Puts independent vars into title"""
    title = ''
    for var in hp.independent_vars:
        try:
            val = hp.__dict__[var]
            title += var + '=' + str(val) + ','
        except KeyError:
            raise KeyError('unknown independent variable')
    return title


def init_biases(ordered_biases_list, hp):
    """Biases must be ordered by layer. Initializes bias tensors as given by hyperparams. Layer number starts at 1."""
    for layer_num, bias in enumerate(ordered_biases_list):
        if hp.biases_init == 'normal':
            bias.normal_(0, hp.init_std)
        else:
            bias.fill_(float(hp.biases_init))
        if (not hp.biases) or layer_num + 1 in hp.layers_wo_bias:    # Layer num starts at 1
            bias.fill_(0)


def init_bias_subspaces(ordered_bias_subspaces_list, hp, exp_row_norm):
    """Bias subspaces must be ordered by layer.
    Sets bias subspace tensor to zero if the corresponding bias is specified to be disabled in hyperparams.
    Layer number starts at 1."""
    for layer_num, bias_subspace in enumerate(ordered_bias_subspaces_list):
        if hp.biases and layer_num + 1 not in hp.layers_wo_bias:
            bias_subspace.normal_(0, 1 / exp_row_norm)
        else:
            bias_subspace.fill_(0)    # Prevents changing the bias


def get_game(hp):
    # Create game env
    if hp.game == 'PD':
        game = PD(payout_mat=hp.payout_mat, device=device)
    elif hp.game == 'OSPD':
        game = OSPD(payout_mat=hp.payout_mat, device=device)
    elif hp.game == 'IPD':
        game = IPD(hp.gamma, device=device)
    elif hp.game == 'OSIPD':
        game = OSIPD(hp.gamma, device=device)
    else:
        raise ValueError('Unknown game')
    return game


def main(hp=None):
    """Run script.

    Args:
        hp: A list of argument strings to use instead of sys.argv.
    """

    hp = parse_args(hp)
    exp_name = [(key, hp.__dict__[key]) for key in sorted(hp.__dict__)]
    print("Hyperparams: \n", exp_name)

    colors = ['b','c','m','r','y','g']
    fig, ax = plt.subplots(1,1)
    save_plot(hp, testing=True)

    for i in range(*hp.n_inner_opt_range):
        torch.manual_seed(hp.seed)
        scores = np.array(play_LOLA(i, hp))
        ax.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('grad steps')
    plt.ylabel('score for each agent')
    # plt.show(block=True)
    save_plot(hp)
    plt.close()



# plot results:
if __name__== "__main__":
    main()







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
