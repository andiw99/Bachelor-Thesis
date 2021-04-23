import numpy as np
import scipy as sc
import tensorflow as tf
from scipy import stats
from scipy import constants

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

def pt_cut(features, E=6500, cut_energy=40, return_cut = False):
    pt = np.sqrt(features[:,0] * features[:,1] * (1-np.tanh(features[:,2] + 1/2 * np.log(features[:,1]/features[:,0]))**2)) * E
    cut = pt > cut_energy
    features = features[cut]
    if return_cut:
        return features, cut
    else:
        return features

def calc_pt(eta, x_1, x_2, E):
    pt = np.sqrt(x_1 * x_2 * (1-np.tanh(eta + 1/2 * np.log(x_2/x_1))**2)) * E
    return pt

def eta_cut(features, cut_eta=2.37, crack_interval=(1.37, 1.52), return_cut = False):
    other_eta = features[:,2] + 1/2 * np.log((features[:,1])**2/(features[:,0]**2))
    #print(crack_interval[0] < np.abs(features[:,2]) < crack_interval[1])
    cut = (np.abs(other_eta) < cut_eta) & (~((np.abs(features[:,2]) > crack_interval[0]) & (np.abs(features[:,2]) < crack_interval[1]))) \
           & (~((np.abs(other_eta) > crack_interval[0]) & (np.abs(other_eta) < crack_interval[1])))
    features = features[cut]
    if return_cut:
        return features, cut
    else:
        return features

def cut(features, E=6500, cut_energy=40, cut_eta=2.37, crack_interval=(1.37, 1.52), return_cut=False):
    pt = calc_pt(eta=features[:,2], x_1=features[:,0], x_2=features[:,1], E=E)
    other_eta = features[:,2] + 1/2 * np.log((features[:,1])**2/(features[:,0]**2))
    cut = (np.abs(other_eta) < cut_eta) & (~((np.abs(features[:,2]) > crack_interval[0]) & (np.abs(features[:,2]) < crack_interval[1]))) \
           & (~((np.abs(other_eta) > crack_interval[0]) & (np.abs(other_eta) < crack_interval[1]))) & (pt > cut_energy) & (np.abs(features[:,2]) < cut_eta)
    features = features[cut]
    if return_cut:
        return features, cut
    else:
        return features

def crack_cut(eta_values, crack_interval=(1.37, 1.52), return_cut = False):
    cut = (~((np.abs(eta_values) > crack_interval[0]) & (np.abs(eta_values) < crack_interval[1])))
    eta_values = eta_values[cut]
    if return_cut:
        return eta_values, cut
    else:
        return eta_values

def calc_other_eta(eta, x_1, x_2):
    other_eta = eta + 1/2 * np.log((x_2)**2/(x_1**2))
    return other_eta

def pt_cut_vale(features):
    pt = (2*features[:,0] * features[:,1] * np.sqrt(1-np.tanh(features[:,2])**2))/(features[:,0] + features[:,1] - (features[:,0] - features[:,1])*np.tanh(features[:,2])) * 6500
    cut = [pt < 20]
    return cut

class diff_WQ_theta():
    def __init__(self, s=1000, q=2/3):
        self.s = s
        self.q = q
        self.e = 0.30282212

    def __call__(self, theta):
        diff_WQ = (self.q**4 * self.e**4)/(24 * np.pi * self.s) * (1 + np.cos(theta)**2)/(np.sin(theta))
        return diff_WQ


class diff_WQ_omega():
    def __init__(self, s=1000, q=2/3):
        self.s = s
        self.q = q
        self.e = 0.30282212

    def __call__(self, theta):
        diff_WQ = (self.q ** 4 * self.e**4) / (48 * np.pi**2 * self.s) * (1 + np.cos(theta) ** 2) / (np.sin(theta)**2)
        return diff_WQ

class diff_WQ_eta():
    def __init__(self, s=40000, q=1/3):
        self.s = s
        self.q = q
        self.e = 0.30282212

    def __call__(self, eta):
        diff_WQ = ((self.q)**4 * self.e**4)/(48 * np.pi * self.s) * (np.tanh(eta)**2 + 1)
        return diff_WQ


class diff_WQ_omega_vale():
    def __init__(self, E=200, q=1/3):
        self.E = E
        self.q = q

    def __call__(self, theta):
        diff_WQ = (self.q ** 4 * (constants.fine_structure)**2) / (6 * self.E**2) * (1 + np.cos(theta) ** 2) / (np.sin(theta)**2)
        return diff_WQ

class diff_WQ_eta_vale():
    def __init__(self, E=200, q=1/3):
        self.E = E
        self.q = q

    def __call__(self, eta):
        diff_WQ = 2 * np.pi * (constants.fine_structure**2 * (self.q)**4)/(6 * self.E**2) * (np.tanh(eta)**2 + 1)
        return diff_WQ


class inverse_cdf():
    def __init__(self, xp, fp):
        self.xp = xp
        self.fp = fp

    def __call__(self, x):
        y = np.interp(x, xp=self.xp, fp=self.fp)
        return y

def gev_to_pb(x):
    return x * 0.389379e9

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

class x_power_dist():
    def __init__(self, power, offset, mean, scale, epsilon):
        self.offset = offset
        self.mean = mean
        self.scale = scale
        self.power = power
        self.epsilon = epsilon

    def __call__(self, x):
        y = (self.offset + (x-self.mean) ** self.power)/self.scale
        return y

    def cdf(self, x):
        y = (self.offset * (x-self.epsilon) + 1/(self.power + 1) * \
            ((x - self.mean) ** (self.power + 1) - (self.epsilon - self.mean) ** (self.power + 1)))/self.scale
        return y