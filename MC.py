import numpy as np
import scipy as sc
import tensorflow as tf
from scipy import stats


class class_loguni():
    def __init__(self, loguni_param, x_lower_limit, x_upper_limit, scaling=1):
        self.a = loguni_param
        self.b = 1 + loguni_param
        self.loguni_param = loguni_param
        self.x_lower_limit = x_lower_limit
        self.x_upper_limit = x_upper_limit
        self.scaling = scaling

    def __call__(self, x):
        #retransform x
        x = (x - self.x_lower_limit)/(self.x_upper_limit - self.x_lower_limit) + self.loguni_param
        x = self.scaling * stats.loguniform.pdf(x, self.a, self.b)
        return x

class gaussian():
    def __init__(self, mu = 0, sigma=1, scaling=1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling

    def __call__(self, x):
        y = self.scaling * 1/(np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-(tf.math.abs(x) - self.mu)**2/(2*self.sigma**2))
        return y

class erf():
    def __init__(self, mu=0, sigma=1, scaling=1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling
    def __call__(self, x):
        y = self.scaling * 1/2 * (1+ sc.special.erf((x-self.mu)/(np.sqrt(2*(self.sigma**2)))))
        return y

def pt_cut(features):
    pt = np.sqrt(features[:,0] * features[:,1] * (1-np.tanh(features[:,2] + 1/2 * np.log(features[:,1]/features[:,0]))**2)) * 6500
    cut = [pt < 20]
    return cut

def pt_cut_vale(features):
    pt = (2*features[:,0] * features[:,1] * np.sqrt(1-np.tanh(features[:,2])**2))/(features[:,0] + features[:,1] - (features[:,0] - features[:,1])*np.tanh(features[:,2])) * 6500
    cut = [pt < 20]
    return cut


class calc_diff_WQ():
    def __init__(self, PDF, quarks, x_1=None, x_2=None, eta=None, E=6500):
        self.PDF = PDF
        self.quarks = quarks
        self.x_1 = x_1
        self.x_2 = x_2
        self.eta = eta
        self.E = E

    def __call__(self, x_1 = None, x_2=None, eta=None):
        if self.x_1 is not None:
            x_1 = self.x_1
        if self.x_2 is not None:
            x_2 = self.x_2
        if self.eta is not None:
            eta= self.eta

        for i, q in enumerate(self.quarks["quark"]):
            if i==0:
                diff_WQ = (((self.quarks["charge"][q - 1]) ** 4) / (192 * np.pi * x_1 * x_2 * self.E ** 2)) * \
                       ((np.maximum(np.array(self.PDF.xfxQ2(q, x_1, 2 * x_1 * x_2 * (self.E ** 2))) * np.array(self.PDF.xfxQ2(-q, x_2, 2 * x_1 * x_2 * (self.E ** 2))), 0) + np.maximum(
                           np.array(self.PDF.xfxQ2(-q, x_1, 2 * x_1 * x_2 * (self.E ** 2))) * np.array(self.PDF.xfxQ2(q, x_2, 2 * x_1 * x_2 * (self.E ** 2))), 0)) / (x_1 * x_2)) * \
                       (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)
            else:
                diff_WQ += (((self.quarks["charge"][q - 1]) ** 4) / (192 * np.pi * x_1 * x_2 * self.E ** 2)) * \
                       ((np.maximum(np.array(self.PDF.xfxQ2(q, x_1, 2 * x_1 * x_2 * (self.E ** 2))) * np.array(self.PDF.xfxQ2(-q, x_2, 2 * x_1 * x_2 * (self.E ** 2))), 0) + np.maximum(
                           np.array(self.PDF.xfxQ2(-q, x_1, 2 * x_1 * x_2 * (self.E ** 2))) * np.array(self.PDF.xfxQ2(q, x_2, 2 * x_1 * x_2 * (self.E ** 2))), 0)) / (x_1 * x_2)) * \
                       (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)

        if diff_WQ.size == 1:
            diff_WQ = float(diff_WQ)
        return diff_WQ