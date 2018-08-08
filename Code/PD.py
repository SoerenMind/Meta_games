from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc import flatten


class PD():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.M = np.array([[-2,0],[-3,-1]])

    def R1(self, actions1, actions2):
        rewards1 = self.M[actions1, actions2]
        return rewards1

    def R2(self, actions1, actions2):
        rewards2 = self.M[actions2, actions1]
        return rewards2

game = PD()
objective_1 = game.R1
objective_2 = game.R2



