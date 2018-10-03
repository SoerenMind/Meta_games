"""Script that plays an open source prisoner's dilemma using the autograd package instead of Pytorch."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.norm as normal
norm = np.linalg.norm
from autograd import grad, hessian, hessian_vector_product, vector_jacobian_product
from autograd.misc import flatten
import matplotlib
# matplotlib.use('MacOSX')
# import os
# print(os.environ['MPLBACKEND'])
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt

def assignprint(y, name=''):
    print(str(name), y)
    return y


# Global hyperparams:
show_gen_params = True
subspace_training = True
optimizer = assignprint('GD')
# CC_val, DD_val = assignprint((-1., -2.))
CC_val, DD_val = assignprint((-0.1, -2.9))
show_plot = False
identity_subspace = assignprint(True)  # Projection=identity, might make some params unseen
old_normalization = assignprint(True)   # Erroneous but initial CC happened here


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(normal.logpdf(x, mu, np.exp(log_std)), axis=-1)


def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] / 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std


def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean



### Define generator, discriminator, and objective ###
def relu(x):       return np.maximum(0, x)


def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)


def logsigmoid(x): return x - np.logaddexp(0, x)


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net. The are also used as initialization when subspace is used."""
    layer_sizes = [x for x in layer_sizes if x != None]
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def get_params_from_subspace(subs_params, subs_project, init_params):
    """Returns actual neural net weights. Adds subs_weights * Pw + 'fixed initial weights'. Same for biases.
    Returns input subs_params if no subs_project is given (ie subspace method isn't used).
    """
    if subs_project == None:
        return subs_params
    params = []
    for (Pw, Pb), (init_W, init_b) in zip(subs_project, init_params):
        init_W, unflatten_W = flatten(init_W)
        W = init_W + np.dot(subs_params, Pw)
        b = init_b + np.dot(subs_params, Pb)
        W = unflatten_W(W)
        params.append((W,b))
    return params

def sample_subs_projections(layer_sizes, subspace_dim, subspace_training):
    """Returns a list of tuples (Pw, Pb) - one for weights and biases of each layer. Pw and Pb are approximately
    orthonormal matrices (in high dimension) that project subspace weights into a subspace of the weights/biases space.
    The matrix P from Uber's work is a stacking of all Pw, Pb. We make P sparse to save space/time."""
    if not subspace_training:
        return None
    subs_project = []
    layer_sizes = [x for x in layer_sizes if x is not None]

    # Quantities needed for column norm normalization and sparsifying P
    num_params = np.sum([m * n for m,n in zip(layer_sizes[:-1], layer_sizes[1:])])
    if old_normalization:
        p = np.max([1 / np.sqrt(num_params), 0.1])
        exp_column_norm = np.sqrt(num_params * p)
    else: exp_column_norm = np.sqrt(num_params)

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        # Create random projections P
        Pw, Pb = npr.randn(subspace_dim, m * n), npr.randn(subspace_dim, n)
        if identity_subspace:
            Pw, Pb = np.eye(subspace_dim, m * n), np.eye(subspace_dim, n)
        # Sparsify
        # sparse_mask_w = np.random.choice(a=[0, 1], size=(subspace_dim, m * n), p=[1-p, p])
        # sparse_mask_b = np.random.choice(a=[0, 1], size=(subspace_dim, n), p=[1-p, p])
        # Pw = sparse_mask_w * Pw
        # Pb = sparse_mask_b * Pb

        # Normalize column norm of P to 1
        # norms_w, norms_b = norm(Pw, axis=1), norm(Pb, axis=1)
        # Pw, Pb = Pw / norms_w.reshape([-1,1]), Pb / norms_b.reshape([-1,1])
        Pw, Pb = Pw/exp_column_norm, Pb/exp_column_norm
        subs_project.append((Pw, Pb))

    return subs_project



def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs




### Define maximin version of Adam optimizer ###
def maximax_optimizer(pl1_grad, pl2_grad, pl1_objective, pl2_objective, pl1_all_params, pl2_all_params, assym_hess_A=None, callback=None, num_iters=100,
                      step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
    """Runs maximizing optimzer on both players. 3 optimizers available."""

    subspace_training = pl1_all_params[3]
    if subspace_training: trainable_param_idx = 0
    else: trainable_param_idx = 2
    reward_log, x_log = [], []

    x_max, unflatten_max = flatten(pl1_all_params[trainable_param_idx])  # Pick and flatten the trainable params
    x_min, unflatten_min = flatten(pl2_all_params[trainable_param_idx])

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))

    pl1_hess = hessian(pl1_objective, argnum=[0])
    pl2_hess = hessian(pl2_objective, argnum=[0])

    # # Alternating optimization per player
    params, objectives, grad_funcs, grad_eval, unflatten, step_sizes = [x_max, x_min], [pl1_objective, pl2_objective], \
               [pl1_grad, pl2_grad], [None, None],[unflatten_max, unflatten_min], [step_size_max, step_size_min]
    v, m, hess = [v_max, v_min], [m_max, m_min], [pl1_hess, pl2_hess]
    for i in range(num_iters):

        for id, opp_id in zip([0, 1], [1, 0]):
            g_uf = grad_funcs[id](unflatten[id](params[id]),
                                unflatten[opp_id](params[opp_id]), i)
            g, unflatten_g = flatten(g_uf)
            grad_eval[id] = g

            if optimizer == 'GD':
                params = params + step_sizes[id] * g

            elif optimizer == 'SGA':
                A = assym_hess_A(unflatten[id](params[id]), unflatten[opp_id](params[opp_id]), hess[id], None)
                update = flatten(g_uf + np.dot(A, g_uf))
                params[id] = params[id] + step_sizes[id] * update

            elif optimizer == 'Adam':
                m[id] = (1 - b1) * g + b1 * m[id]  # First  moment estimate.
                v[id] = (1 - b2) * (g ** 2) + b2 * v[id]  # Second moment estimate.
                mhat_max = m[id] / (1 - b1 ** (i + 1))  # Bias correction.
                vhat_max = v[id] / (1 - b2 ** (i + 1))
                params[id] = params[id] + step_sizes[id] * mhat_max / (np.sqrt(vhat_max) + eps)

        # Plot
        if i % 100 == 0:
            y2, y1 = (objectives[id](unflatten[id](params[id]),
                                     unflatten[opp_id](params[opp_id]), i),
                  objectives[opp_id](unflatten[opp_id](params[opp_id]),
                                     unflatten[id](params[id]), i))
            g_2_norm, g_1_norm = norm(g[id]), norm(g[opp_id])
            print(i, y1, y2, '   -   ', g_1_norm, g_2_norm)
            reward_log.append((y1, y2)), x_log.append(i)
        if callback: callback(reward_log, x_log, i)

    # # Joint optimization
    # for i in range(num_iters):
    #     # Get gradients
    #     g_max_uf = pl1_grad(unflatten_max(x_max),
    #                                    unflatten_min(x_min), i)
    #     g_min_uf = pl2_grad(unflatten_min(x_min),
    #                                    unflatten_max(x_max), i)
    #     g_max, unflatten_g_max = flatten(g_max_uf)
    #     g_min, unflatten_g_min = flatten(g_min_uf)
    #
    #     if optimizer == 'GD':
    #         x_max = x_max + step_size_max * g_max
    #         x_min = x_min + step_size_min * g_min
    #
    #     elif optimizer == 'SGA':
    #         # Apply symplectic adjusted gradient
    #         pl1_A = assym_hess_A(unflatten_max(x_max), unflatten_min(x_min), pl1_hess, None)
    #         pl2_A = assym_hess_A(unflatten_min(x_min), unflatten_max(x_max), pl2_hess, None)
    #         update_max = flatten(g_max_uf + np.dot(pl1_A, g_max_uf))
    #         update_min = flatten(g_min_uf + np.dot(pl2_A, g_min_uf))
    #         x_max = x_max + step_size_max * update_max
    #         x_min = x_min + step_size_min * update_min
    #
    #     elif optimizer == 'Adam':
    #         m_min = (1 - b1) * g_min + b1 * m_min  # First  moment estimate.
    #         v_min = (1 - b2) * (g_min ** 2) + b2 * v_min  # Second moment estimate.
    #         v_min = (1 - b2) * (g_min ** 2) + b2 * v_min  # Second moment estimate.
    #         mhat_min = m_min / (1 - b1 ** (i + 1))  # Bias correction.
    #         vhat_min = v_min / (1 - b2 ** (i + 1))
    #         x_min = x_min + step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)
    #
    #         m_max = (1 - b1) * g_max + b1 * m_max  # First  moment estimate.
    #         v_max = (1 - b2) * (g_max ** 2) + b2 * v_max  # Second moment estimate.
    #         mhat_max = m_max / (1 - b1 ** (i + 1))  # Bias correction.
    #         vhat_max = v_max / (1 - b2 ** (i + 1))
    #         x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)
    #
    #
    #     # Plot
    #     if i % 100 == 0:
    #         y1, y2 = (pl1_objective(unflatten_max(x_max),
    #                                        unflatten_min(x_min), i),
    #               pl2_objective(unflatten_min(x_min),
    #                                        unflatten_max(x_max), i))
    #         g_max_norm, g_min_norm = norm(g_max), norm(g_min)
    #         print(i, y1, y2, '   -   ', g_max_norm, g_min_norm)
    #         reward_log.append((y1, y2)), x_log.append(i)
    #     if callback: callback(reward_log, x_log, i)

    return unflatten_max(x_max), unflatten_min(x_min), reward_log, x_log

def make_objective_func(pl1_all_params, pl2_all_params):
    """Returns objective function of player 1 which takes player 2's parameters as input."""
    pl1_trainable_params, pl1_subs_project, init_pl1_params, _ = tuple(pl1_all_params)
    pl2_trainable_params, pl2_subs_project, init_pl2_params, _ = tuple(pl2_all_params)

    def objective(pl1_trainable_params, pl2_trainable_params, iter):
        """Returns expected reward for player 1 given parameters of both players. """
        if subspace_training:
            pl1_params = get_params_from_subspace(pl1_trainable_params, pl1_subs_project, init_pl1_params)
            pl2_params = get_params_from_subspace(pl2_trainable_params, pl2_subs_project, init_pl2_params)
        else:
            pl1_params, pl2_params = pl1_trainable_params, pl2_trainable_params

        prob_1 = sigmoid(neural_net_predict(pl1_params, pl2_trainable_params))
        prob_2 = sigmoid(neural_net_predict(pl2_params, pl1_trainable_params))
        M = np.array([[DD_val, 0], [-3, CC_val]])
        outcome_probs = np.outer([1 - prob_1, prob_1], [1 - prob_2, prob_2])
        reward_1 = (M * outcome_probs).sum()
        # reward_2 = (M.T * outcome_probs).sum()
        return reward_1

    return objective



if __name__ == '__main__':

    # Model hyper-parameters
    latent_dim = 1
    data_dim = 1
    gen_subspace_dim, dsc_subspace_dim = 50, 50
    gen_units_1, gen_units_2, gen_units_3, dsc_units_1, dsc_units_2, dsc_units_3 = assignprint((200, 40, 40, 200, 40, 40))
    # gen_units_1, gen_units_2, gen_units_3, dsc_units_1, dsc_units_2, dsc_units_3 = assignprint((50, 10, None, 50, 10, None))
    # gen_units_1, gen_units_2, gen_units_3, dsc_units_1, dsc_units_2, dsc_units_3 = assignprint((50, None, None, 50, None, None))
    pl1_subs_params, pl2_subs_params = np.zeros(gen_subspace_dim), np.zeros(dsc_subspace_dim)
    # seed = npr.RandomState(assignprint(0, 'seed:'))
    npr.seed(assignprint(1, 'seed:'))

    # Training parameters
    param_scale = assignprint(.1)
    # batch_size = 77
    num_epochs = 50000
    lrate_adjust = 10. if optimizer == 'GD' else 1.
    _, lrate = assignprint(("lrate:", 0.001))
    step_size_max = lrate * lrate_adjust
    step_size_min = lrate * lrate_adjust
    step_size_max_LOLA = lrate * lrate_adjust
    step_size_min_LOLA = lrate * lrate_adjust

    # Initialize gen & dsc params
    gen_layer_sizes = [dsc_subspace_dim, gen_units_1, gen_units_2, gen_units_3, 1]
    dsc_layer_sizes = [gen_subspace_dim, dsc_units_1, dsc_units_2, dsc_units_3, 1]

    init_pl1_params = init_random_params(param_scale, gen_layer_sizes)
    num_trainable_gen_params = gen_subspace_dim if subspace_training else np.size(flatten(init_pl1_params)[0])
    num_direct_gen_params = np.size(flatten(init_pl1_params)[0])
    print("num trainable and direct gen params: " + str(num_trainable_gen_params) + ', ' + str(num_direct_gen_params))
    # if show_gen_params:
    #     dsc_input_size = data_dim + num_trainable_gen_params
    # else: dsc_input_size = data_dim
    init_pl2_params = init_random_params(param_scale, dsc_layer_sizes)
    num_trainable_dsc_params = dsc_subspace_dim if subspace_training else np.size(flatten(init_pl2_params)[0])
    num_direct_dsc_params = np.size(flatten(init_pl2_params)[0])
    print("num trainable and direct dsc params: " + str(num_trainable_dsc_params) + ', ' + str(num_direct_dsc_params))

    # Draw random subspace matricesA
    pl1_subs_project = sample_subs_projections(gen_layer_sizes, gen_subspace_dim, subspace_training)
    pl2_subs_project = sample_subs_projections(dsc_layer_sizes, dsc_subspace_dim, subspace_training)

    pl1_all_params = [pl1_subs_params, pl1_subs_project, init_pl1_params, subspace_training]
    pl2_all_params = [pl2_subs_params, pl2_subs_project, init_pl2_params, subspace_training]



    # Make objectives
    pl1_objective = make_objective_func(pl1_all_params, pl2_all_params)
    pl2_objective = make_objective_func(pl2_all_params, pl1_all_params)

    d_R2_d_t2 = grad(pl2_objective, argnum=[0])
    d_R1_d_t2 = grad(pl1_objective, argnum=[1])
    d_R2_d_t1 = grad(pl2_objective, argnum=[1])
    d_R1_d_t1 = grad(pl1_objective, argnum=[0])

    def pl1_LOLA_objective_taylor(pl1_subs_params, pl2_subs_params, iter):
        """Note that this predicts a GD step, not an Adam step."""
        return pl1_objective(pl1_subs_params, pl2_subs_params, iter) \
            + step_size_min  \
              * np.dot(d_R2_d_t2(pl2_subs_params, pl1_subs_params, iter)[0], d_R1_d_t2(pl1_subs_params, pl2_subs_params, iter)[0])

    def pl2_LOLA_objective_taylor(pl2_subs_params, pl1_subs_params, iter):
        return pl2_objective(pl2_subs_params, pl1_subs_params, iter) \
            + step_size_max \
              * np.dot(d_R1_d_t1(pl1_subs_params, pl2_subs_params, iter)[0], d_R2_d_t1(pl2_subs_params, pl1_subs_params, iter)[0])

    pl1_LOLA_objective = lambda pl1_subs_params, pl2_subs_params, iter: pl1_objective(pl1_subs_params, pl2_subs_params
                                       + d_R2_d_t2(pl2_subs_params, pl1_subs_params, iter)[0] * step_size_min_LOLA, iter)
    pl2_LOLA_objective = lambda pl2_subs_params, pl1_subs_params, iter: pl2_objective(pl2_subs_params, pl1_subs_params
                                       + d_R1_d_t1(pl1_subs_params, pl2_subs_params, iter)[0] * step_size_max_LOLA, iter)

    # pl1_hess = hessian(pl1_objective, argnum=[0])
    # pl2_hess = hessian(pl2_objective, argnum=[0])
    #
    # def assym_hess_A(pl1_subs_params, pl2_subs_params, hess, iter):
    #     """For gradient adjustment from https://arxiv.org/abs/1802.05642."""
    #     hess_eval = hess(pl1_subs_params, pl2_subs_params, iter)
    #     A = (hess_eval[0] - hess_eval[0].T) / 2.
    #     return A

    # Test forward pass
    if subspace_training:
        pl1_objective(pl1_subs_params, pl2_subs_params, None)     # With subspace
        pl2_objective(pl2_subs_params, pl1_subs_params, None)

        pl1_LOLA_objective(pl1_subs_params, pl2_subs_params, None)
        pl2_LOLA_objective(pl2_subs_params, pl1_subs_params, None)

        # assym_hess_A(pl1_subs_params, pl2_subs_params, pl1_hess, None)
        # assym_hess_A(pl2_subs_params, pl1_subs_params, pl2_hess, None)
    else: pl1_objective(init_pl1_params, init_pl2_params, None)     # W/o  subspace

    # Get gradient function of objective using autograd.
    # pl1_grad = grad(pl1_objective, argnum=[0])
    # pl2_grad = grad(pl2_objective, argnum=[0])
    # print('Not using LOLA')
    pl1_grad = grad(pl1_LOLA_objective, argnum=[0])
    pl2_grad = grad(pl2_LOLA_objective, argnum=[0])
    print('Using LOLA')

    # Set up figure.
    fig = plt.figure(figsize=(3, 4), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    ax.set_title("Reward of players vs grad steps")
    print("Epoch|Objective 1|Objective 2      |      grad norm 1   |  grad norm 2")


    def print_log(log, x_log, iter):
        if iter % 50 == 0:
            if show_plot:
                ax.plot(x_log, log)
                ax.set_ylim([-5, 0])
                plt.draw()
                plt.pause(1.0 / 6000.0)
        return




    _, _, reward_log, x_log = maximax_optimizer(pl1_grad, pl2_grad, pl1_objective, pl2_objective,
                                         pl1_all_params, pl2_all_params, assym_hess_A=None, b1=0,
                                         step_size_max=step_size_max, step_size_min=step_size_min,
                                         num_iters=num_epochs, callback=print_log)
    plt.show(block=True)

