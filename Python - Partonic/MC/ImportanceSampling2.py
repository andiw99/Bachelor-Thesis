import numpy as np
from numpy import random
from scipy import stats
from scipy import integrate
from scipy import constants
from matplotlib import pyplot as plt
from sympy.solvers import solve
from sympy.solvers import solveset
from sympy import Symbol
import MC


class omega_integrand():
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x) * np.sin(x)

def main():
    uniform_dist = stats.uniform.rvs(loc = 0, scale = 1, size=200)
    plt.hist(uniform_dist, bins=20)
    plt.show()
    epsilon = 0.163
    offset = 0.2
    """
    custom_dist = MC.x_power_dist(power=4, offset=offset, mean=np.pi/2, scale=1, epsilon=epsilon)
    scale = custom_dist.cdf(np.pi - epsilon) - custom_dist.cdf(epsilon)
    custom_dist = MC.x_power_dist(power=4, offset=offset, mean=np.pi/2, scale=scale, epsilon=epsilon)
    """
    custom_dist = MC.x_power_dist(power=4, offset=offset, a=epsilon, b=np.pi-epsilon, normed=True)

    x = np.linspace(0+epsilon, np.pi-epsilon, num=500)
    y = custom_dist.cdf(x)
    plt.plot(y, x)
    plt.show()

    custom_samples = custom_dist.rvs(size=50000, interpol_nr=50000)
    plt.hist(custom_samples, bins=20)
    plt.show()

    #Plot WQ von theta
    diff_WQ_omega_vale = MC.diff_WQ_omega_vale(E=200, q=1/3)
    plt.plot(np.linspace(start=epsilon, stop=np.pi - epsilon, num=200), MC.gev_to_pb(diff_WQ_omega_vale(np.linspace(start=epsilon, stop=np.pi-epsilon, num=200))))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\frac{d\sigma}{d\Omega}$")
    plt.show()

    #totalen WQ berechnen, integration über theta und über phi (*2pi)
    sigma_total = 2 * np.pi * integrate.quad(diff_WQ_omega_vale, a=epsilon, b=np.pi-epsilon)[0]
    sigma_total_mc = 2 * np.pi * np.mean(diff_WQ_omega_vale(custom_samples)/custom_dist(custom_samples))
    print("sigma total in GeV:", sigma_total, "sigma total in pb:", MC.gev_to_pb(sigma_total))
    print("mit MC:", sigma_total_mc, "in pb:", MC.gev_to_pb(sigma_total_mc))

    #totalen WQ berechnen, integration über theta und über phi (*2pi)
    diff_WQ_theta = MC.diff_WQ_theta(s=40000, q=1/3)
    sigma_total = integrate.quad(diff_WQ_theta, a=epsilon, b=np.pi-epsilon)[0]
    sigma_total_mc =  np.mean(diff_WQ_theta(custom_samples)/custom_dist(custom_samples))
    print("sigma total in GeV:", sigma_total, "sigma total in pb:", 1/2 * MC.gev_to_pb(sigma_total))
    print("mit MC:", sigma_total_mc, "in pb:", 1/2 * MC.gev_to_pb(sigma_total_mc))

    diff_WQ_omega = MC.diff_WQ_omega(s=40000, q=1/3)
    omega_integrand_func = omega_integrand(diff_WQ_omega)
    sigma_total = 2 * np.pi * integrate.quad(omega_integrand_func, a=epsilon, b=np.pi-epsilon)[0]
    sigma_total_mc = 2 * np.pi * np.mean(omega_integrand_func(custom_samples)/custom_dist(custom_samples))
    print("sigma total in GeV:", sigma_total, "sigma total in pb:", MC.gev_to_pb(sigma_total))
    print("mit MC:", sigma_total_mc, "in pb:", MC.gev_to_pb(sigma_total_mc))

    #totalen WQ berechnen mit integration über eta
    diff_WQ_eta_vale = MC.diff_WQ_eta_vale(E=200, q=1/3)
    sigma_total_eta = integrate.quad(diff_WQ_eta_vale, a=-2.5, b=2.5)[0]
    eta_integration_dist = stats.uniform.rvs(loc=-2.5 , scale=5, size=5000)
    sigma_total_eta_mc =  np.mean(diff_WQ_eta_vale(eta_integration_dist)/(stats.uniform.pdf(x=eta_integration_dist, loc=-2.5, scale=5)))
    print("sigma total in GeV mit eta:", sigma_total_eta, "in pb", MC.gev_to_pb(sigma_total_eta))
    print("sigma total mit mc:", sigma_total_eta_mc, "in pb", MC.gev_to_pb(sigma_total_eta_mc))

    #totalen WQ berechnen mit integration über eta
    diff_WQ_eta = MC.diff_WQ_eta(s=40000, q=1/3)
    sigma_total_eta = integrate.quad(diff_WQ_eta, a=-2.5, b=2.5)[0]
    eta_integration_dist = stats.uniform.rvs(loc=-2.5 , scale=5, size=5000)
    sigma_total_eta_mc =  np.mean(diff_WQ_eta(eta_integration_dist)/(stats.uniform.pdf(x=eta_integration_dist, loc=-2.5, scale=5)))
    print("sigma total in GeV mit eta:", sigma_total_eta, "in pb", MC.gev_to_pb(sigma_total_eta))
    print("sigma total mit mc:", sigma_total_eta_mc, "in pb", MC.gev_to_pb(sigma_total_eta_mc))

if __name__ == "__main__":
    main()

