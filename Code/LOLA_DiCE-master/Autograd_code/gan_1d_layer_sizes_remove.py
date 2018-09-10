from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import numpy as nnp
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc import flatten

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import ortho_group  # Requires version 0.18 of scipy
# m = ortho_group.rvs(dim=3)


# Global hyperparams
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

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_random_params_subspace(scale, layer_sizes, subspace_dim, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    num_params = sum([(m * n + n)
                  for m, n in zip(layer_sizes[:-1], layer_sizes[1:])])

    P = npr.randn(subspace_dim,num_params)
    norms = np.linalg.norm(P,axis=0)
    P = P/norms
    subspace_params = np.zeros(subspace_dim)


    init_params = [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


    params = init_params # + np.dot(P,subspace_params)

    return

def neural_net_predict(params, layer_sizes, subspace_params, P, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""

    if not subspace_params:
        return neural_net_predict_regular(params, inputs)

    flat_params = np.dot(P, subspace_params)

    idx = 0
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        params_W = flat_params[idx:idx + m*n]
        b = flat_params[idx + m*n: idx + m*n+n]
        W = np.reshape(params_W, [m, n])
        idx = idx + (m*n + n)

        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)

    return outputs

def neural_net_predict_regular(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs

def generate_from_noise(gen_params, gen_layer_sizes, subspace_params, num_samples, noise_dim, rs):
    noise = rs.rand(num_samples, noise_dim)
    return neural_net_predict(gen_params, gen_layer_sizes, subspace_params, noise)

def disc(dsc_params, dsc_layer_sizes, gen_params, data):
    """Uncomment/comment to determine if the discriminator sees the generator's params."""
    # return neural_net_predict(dsc_params, data)

    N, D = data.shape
    flat_params, _ = flatten(gen_params)
    data_and_params = np.hstack([data, np.tile(flat_params, (N, 1))])
    return neural_net_predict(dsc_params, dsc_layer_sizes, data_and_params)

def gan_objective(gen_params, dsc_params, real_data, num_samples, noise_dim, rs):
    fake_data = generate_from_noise(gen_params, gen_layer_sizes, num_samples, noise_dim, rs)
    logprobs_fake = disc(dsc_params, dsc_layer_sizes, gen_params, fake_data)
    logprobs_real = np.exp(disc(dsc_params, dsc_layer_sizes, gen_params, real_data))
    return np.mean(logprobs_real) - np.mean(logprobs_fake)

### Define maximin version of Adam optimizer ###

def adam_maximin(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam modified to do minimiax optimization, for instance to help with
    training generative adversarial networks."""

    x_max, unflatten_max = flatten(init_params_max)
    x_min, unflatten_min = flatten(init_params_min)

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))

    for i in range(num_iters):

        g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                       unflatten_min(x_min), i)
        g_max, _ = flatten(g_max_uf)
        g_min, _ = flatten(g_min_uf)

        if callback: callback(unflatten_max(x_max), unflatten_min(x_min), i)

        m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
        v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
        mhat_min = m_min / (1 - b1**(i + 1))        # Bias correction.
        vhat_min = v_min / (1 - b2**(i + 1))
        x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

        m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
        v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
        mhat_max = m_max / (1 - b1**(i + 1))        # Bias correction.
        vhat_max = v_max / (1 - b2**(i + 1))
        x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

    return unflatten_max(x_max), unflatten_min(x_min)

# Define true data distribution
mu1 = 0.3
s1 = 0.5
def true_data_dist_logprob(x):
    return diag_gaussian_log_density(x, mu1, np.log(s1))
def sample_true_data_dist(N, rs):
    return rs.randn(N,1) * s1 + mu1

if __name__ == '__main__':


    # Model hyper-parameters
    latent_dim = 1
    data_dim = 1
    gen_layer_sizes = [latent_dim, 20, 20, data_dim]


    # Training parameters
    param_scale = 0.1
    batch_size = 100
    num_epochs = 5000
    step_size_max = 0.001
    step_size_min = 0.001

    init_gen_params = init_random_params(param_scale, gen_layer_sizes)

    num_gen_params = np.size(flatten(init_gen_params)[0])
    print("num gen params: " + str(num_gen_params))

    # dsc_layer_sizes = [data_dim, 30, 20, latent_dim]                          # Don't show generator params
    dsc_layer_sizes = [data_dim + num_gen_params, 30, 20, latent_dim]           # Show generator params
    init_dsc_params = init_random_params(param_scale, dsc_layer_sizes)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(gen_params, gen_layer_sizes, dsc_params, dsc_layer_sizes, iter):
        real_data = sample_true_data_dist(batch_size, seed)
        return gan_objective(gen_params, gen_layer_sizes, dsc_params, dsc_layer_sizes, real_data,
                             batch_size, latent_dim, seed)

    # Get gradients of objective using autograd.
    objective_eval = objective(init_gen_params,gen_layer_sizes,init_dsc_params,dsc_layer_sizes,num_epochs)
    both_objective_grad = grad(objective, argnum=[0, 2])    # TODO: check this is right!

    # Set up figure.
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)
    ax.set_title("True weights")

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(gen_params, gen_layer_sizes, subspace_params, dsc_params, iter):
        if iter % 10 == 0:
            ability = np.mean(objective(gen_params, gen_layer_sizes, dsc_params, dsc_layer_sizes, iter))
            fake_data = generate_from_noise(gen_params, gen_layer_sizes, subspace_params, 1000, latent_dim, seed)
            real_data = sample_true_data_dist(100, seed)
            probs_fake = np.mean(np.exp(disc(dsc_params, gen_params, fake_data)))
            probs_real = np.mean(np.exp(disc(dsc_params, gen_params, real_data)))
            print("{:15}|{:20}|{:20}|{:20}".format(iter, ability, probs_fake, probs_real))

            # Plot data and functions.
            figrange = (-1, 3)
            plot_inputs = np.expand_dims(np.linspace(*figrange, num=400), 1)
            outputs = np.exp(true_data_dist_logprob(plot_inputs))
            discvals = sigmoid(disc(dsc_params, gen_params, plot_inputs))

            h, b = np.histogram(fake_data, bins=100, range=figrange, density=True)

            plt.cla()
            ax.plot(plot_inputs, outputs, 'g-')
            ax.plot(plot_inputs, discvals, 'r-')
            ax.plot(b[:-1], h, 'b-')
            ax.set_ylim([0, 3])
            plt.draw()
            plt.pause(1.0/60.0)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam_maximin(both_objective_grad,
                                    init_gen_params, init_dsc_params, b1=0,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs, callback=print_perf)
