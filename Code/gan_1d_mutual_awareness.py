from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.norm as norm
from copy import deepcopy

from autograd import grad
from autograd.misc import flatten

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Global hyperparams:
show_gen_params = True
subspace_training = True


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)


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




# def init_random_params(scale, layer_sizes, subspace_weights, rs=npr.RandomState(0)):
#     """Build a list of (weights, biases) tuples,
#        one for each layer in the net."""
#
#     params = []
#     subspace_params = [subspace_weights, []]
#     subspace_dim = len(subspace_weights)
##
#     for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
#
#         # Create random projections P
#         Pw, Pb = npr.randn(subspace_dim, m * n), npr.randn(subspace_dim, n)
#         norms_w, norms_b = np.linalg.norm(Pw, axis=1), np.linalg.norm(Pb, axis=1)
#         Pw, Pb = Pw / norms_w.reshape([-1,1]), Pb / norms_b.reshape([-1,1])
#
#         # Initial params
#         init_params = (scale * rs.randn(m * n),  # weight matrix
#                         scale * rs.randn(n))    # bias vector
#
#         # Initial params + subspace
#         layer_weights = init_params[0] + np.dot(subspace_weights, Pw)
#         layer_biases = init_params[1] + np.dot(subspace_weights, Pb)
#         layer_weights = layer_weights.reshape([m,n])
#
#         params.append((layer_weights, layer_biases))
#         subspace_params[1].append((Pw, Pb))
#
#     return params

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net. The are also used as initialization when subspace is used."""
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

def sample_subs_projections(layer_sizes, subspace_dim, subspace_training, rs=npr.RandomState(0)):
    """Returns a list of tuples (Pw, Pb) - one for weights and biases of each layer. Pw and Pb are approximately
    orthonormal matrices (in high dimension) that project subspace weights into a subspace of the weights/biases space.
    The matrix P from Uber's work is a stacking of all Pw, Pb. We make P sparse to save space/time."""
    if not subspace_training:
        return None
    subs_project = []

    # Quantities needed for column norm normalization and sparsifying P
    num_params = np.sum([m * n for m,n in zip(layer_sizes[:-1], layer_sizes[1:])])
    p = np.max([1 / np.sqrt(num_params), 0.1])
    exp_column_norm = np.sqrt(num_params * p)

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        # Create random projections P
        Pw, Pb = npr.randn(subspace_dim, m * n), npr.randn(subspace_dim, n)

        # Sparsify
        # sparse_mask_w = np.random.choice(a=[0, 1], size=(subspace_dim, m * n), p=[1-p, p])
        # sparse_mask_b = np.random.choice(a=[0, 1], size=(subspace_dim, n), p=[1-p, p])
        # Pw = sparse_mask_w * Pw
        # Pb = sparse_mask_b * Pb

        # Normalize column norm of P to 1
        # norms_w, norms_b = np.linalg.norm(Pw, axis=1), np.linalg.norm(Pb, axis=1)
        # Pw, Pb = Pw / norms_w.reshape([-1,1]), Pb / norms_b.reshape([-1,1])
        Pw, Pb = Pw/exp_column_norm, Pb/exp_column_norm
        subs_project.append((Pw, Pb))

    # lst = [x for x, y in subs_project]+[y for x, y in subs_project]
    # P = np.hstack(lst)
    return subs_project



def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)  # missing sigmoid + logits?
    return outputs




def generate_from_noise(gen_params, num_samples, noise_dim, rs, dsc_trainable_params=None):
    inputs = rs.rand(num_samples, noise_dim)    # Noise input
    # Dsc parameters input
    if dsc_trainable_params is not None:
        inputs = np.concatenate([inputs, np.repeat(dsc_trainable_params.reshape([1, -1]), num_samples, axis=0)], axis=1)
    return neural_net_predict(gen_params, inputs)


def disc(dsc_params, gen_params, data):
    """Uncomment/comment to determine if the discriminator sees the generator's params."""
    if not show_gen_params:
        return neural_net_predict(dsc_params, data)

    N, D = data.shape
    flat_params, _ = flatten(gen_params)
    data_and_params = np.hstack([data, np.tile(flat_params, (N, 1))])
    return neural_net_predict(dsc_params, data_and_params)

def gan_objective(gen_params, dsc_params, gen_trainable_params, dsc_trainable_params, real_data, num_samples, noise_dim, rs):
    fake_data = generate_from_noise(gen_params, num_samples, noise_dim, rs, dsc_trainable_params)
    input_params = gen_trainable_params if subspace_training else gen_params
    logprobs_fake = disc(dsc_params, input_params, fake_data)
    # logprobs_real = disc(dsc_params, gen_params, real_data)
    logprobs_real = np.exp(disc(dsc_params, input_params, real_data))
    # TODO(sorenmind): Figure out why np.exp helps here
    return np.mean(logprobs_real) - np.mean(logprobs_fake)


### Define maximin version of Adam optimizer ###

def adam_maximin(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
                 step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
    """Adam modified to do minimiax optimization, for instance to help with
    training generative adversarial networks."""

    subspace_training = init_params_max[3]
    if subspace_training: trainable_param_idx = 0
    else: trainable_param_idx = 2

    x_max, unflatten_max = flatten(init_params_max[trainable_param_idx])  # Pick and flatten the trainable params
    x_min, unflatten_min = flatten(init_params_min[trainable_param_idx])

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))

    for i in range(num_iters):

        g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                       unflatten_min(x_min), i)
        g_max, _ = flatten(g_max_uf)
        g_min, _ = flatten(g_min_uf)

        if callback: callback(unflatten_max(x_max), unflatten_min(x_min), init_params_max, init_params_min, i)

        m_min = (1 - b1) * g_min + b1 * m_min  # First  moment estimate.
        v_min = (1 - b2) * (g_min ** 2) + b2 * v_min  # Second moment estimate.
        mhat_min = m_min / (1 - b1 ** (i + 1))  # Bias correction.
        vhat_min = v_min / (1 - b2 ** (i + 1))
        x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

        m_max = (1 - b1) * g_max + b1 * m_max  # First  moment estimate.
        v_max = (1 - b2) * (g_max ** 2) + b2 * v_max  # Second moment estimate.
        mhat_max = m_max / (1 - b1 ** (i + 1))  # Bias correction.
        vhat_max = v_max / (1 - b2 ** (i + 1))
        x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

    return unflatten_max(x_max), unflatten_min(x_min)


# Define true data distribution
mu1 = 0.3
s1 = 0.5


def true_data_dist_logprob(x):
    return diag_gaussian_log_density(x, mu1, np.log(s1))


def sample_true_data_dist(N, rs):
    return rs.randn(N, 1) * s1 + mu1


if __name__ == '__main__':

    # Model hyper-parameters
    latent_dim = 1
    data_dim = 1
    gen_subspace_dim, dsc_subspace_dim= 100, 1000
    gen_units_1, gen_units_2, dsc_units_1, dsc_units_2 = 20, 20, 30, 20
    gen_subs_weights, dsc_subs_weights = np.zeros(gen_subspace_dim), np.zeros(dsc_subspace_dim)
    seed = npr.RandomState(0)

    # Training parameters
    param_scale = 0.1
    batch_size = 77
    num_epochs = 5000
    step_size_max = 0.001
    step_size_min = 0.001

    # Initialize gen & dsc params
    gen_layer_sizes = [latent_dim + dsc_subspace_dim, gen_units_1, gen_units_2, data_dim]
    init_gen_params = init_random_params(param_scale, gen_layer_sizes)
    num_trainable_gen_params = gen_subspace_dim if subspace_training else np.size(flatten(init_gen_params)[0])
    num_direct_gen_params = np.size(flatten(init_gen_params)[0])
    print("num trainable and direct gen params: " + str(num_trainable_gen_params) + ', ' + str(num_direct_gen_params))
    if show_gen_params:
        dsc_input_size = data_dim + num_trainable_gen_params
    else: dsc_input_size = data_dim
    dsc_layer_sizes = [dsc_input_size, dsc_units_1, dsc_units_2, latent_dim]    # TODO(sorenmind): Why latent_dim?
    init_dsc_params = init_random_params(param_scale, dsc_layer_sizes)
    num_trainable_dsc_params = dsc_subspace_dim if subspace_training else np.size(flatten(init_dsc_params)[0])
    num_direct_dsc_params = np.size(flatten(init_dsc_params)[0])
    print("num trainable and direct dsc params: " + str(num_trainable_dsc_params) + ', ' + str(num_direct_dsc_params))

    # Draw random subspace matrices
    gen_subs_project = sample_subs_projections(gen_layer_sizes, gen_subspace_dim, subspace_training, rs=seed)
    dsc_subs_project = sample_subs_projections(dsc_layer_sizes, dsc_subspace_dim, subspace_training, rs=seed)

    gen_all_params = [gen_subs_weights, gen_subs_project, init_gen_params, subspace_training]
    dsc_all_params = [dsc_subs_weights, dsc_subs_project, init_dsc_params, subspace_training]

    # Define training objective
    def objective(gen_trainable_params, dsc_trainable_params, iter):
        if subspace_training:
            gen_params = get_params_from_subspace(gen_trainable_params, gen_subs_project, init_gen_params)
            dsc_params = get_params_from_subspace(dsc_trainable_params, dsc_subs_project, init_dsc_params)
        else: gen_params, dsc_params = gen_trainable_params, dsc_trainable_params
        real_data = sample_true_data_dist(batch_size, seed)

        # For testing
        # dsc_trainable_params = deepcopy(dsc_trainable_params)
        # dsc_trainable_params = np.zeros(len(dsc_trainable_params))

        return gan_objective(gen_params, dsc_params, gen_trainable_params, dsc_trainable_params, real_data,
                             batch_size, latent_dim, seed)

    # Test forward pass
    if subspace_training:
          objective(gen_subs_weights, dsc_subs_weights, None)     # With subspace
    else: objective(init_gen_params, init_dsc_params, None)     # W/o  subspace

    # Get gradient function of objective using autograd.
    both_objective_grad = grad(objective, argnum=[0, 1])

    # Set up figure.
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)
    ax.set_title("True weights")

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")


    def print_perf(gen_trainable_params, dsc_trainable_params, init_params_max, init_params_min, iter):
        if iter % 10 == 0:

            ability = np.mean(objective(gen_trainable_params, dsc_trainable_params, iter))

            subspace_training = init_params_max[3]
            gen_nn_params = get_params_from_subspace(gen_trainable_params, init_params_max[1], init_params_max[2])
            dsc_nn_params = get_params_from_subspace(dsc_trainable_params, init_params_min[1], init_params_min[2])
            input_params = gen_trainable_params if subspace_training else gen_nn_params

            # TODO(sorenmind): REMOVE!
            # dsc_trainable_params = np.zeros(len(dsc_trainable_params))

            fake_data = generate_from_noise(gen_nn_params, 1000, latent_dim, seed, dsc_trainable_params)
            real_data = sample_true_data_dist(100, seed)
            probs_fake = np.mean(np.exp(disc(dsc_nn_params, input_params, fake_data)))
            probs_real = np.mean(np.exp(disc(dsc_nn_params, input_params, real_data)))
            print("{:15}|{:20}|{:20}|{:20}".format(iter, ability, probs_fake, probs_real))

            # Plot data and functions.
            figrange = (-1, 3)
            plot_inputs = np.expand_dims(np.linspace(*figrange, num=400), 1)
            outputs = np.exp(true_data_dist_logprob(plot_inputs))
            discvals = sigmoid(disc(dsc_nn_params, input_params, plot_inputs))

            h, b = np.histogram(fake_data, bins=100, range=figrange, density=True)

            plt.cla()
            ax.plot(plot_inputs, outputs, 'g-')
            ax.plot(plot_inputs, discvals, 'r-')
            ax.plot(b[:-1], h, 'b-')
            ax.set_ylim([0, 3])
            plt.draw()
            plt.pause(1.0 / 60.0)



    optimized_params = adam_maximin(both_objective_grad,
                                    gen_all_params, dsc_all_params, b1=0,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs, callback=print_perf)