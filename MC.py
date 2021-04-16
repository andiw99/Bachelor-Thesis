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
