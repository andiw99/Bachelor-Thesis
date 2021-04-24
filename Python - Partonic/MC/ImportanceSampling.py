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


uniform_dist = stats.uniform.rvs(loc = 0, scale = 1, size=200)
plt.hist(uniform_dist, bins=20)
plt.show()
epsilon = 0.163
offset = 0.2

custom_dist = MC.x_power_dist(power=4, offset=offset, mean=np.pi/2, scale=1, epsilon=epsilon)
scale = custom_dist.cdf(np.pi - epsilon)

custom_dist = MC.x_power_dist(power=4, offset=offset, mean=np.pi/2, scale=scale, epsilon=epsilon)
print(custom_dist.cdf(np.pi-epsilon))


plt.plot(np.linspace(0, np.pi, num=500), custom_dist(np.linspace(0, np.pi, num=500)))
plt.show()

plt.plot(np.linspace(0, np.pi, num=500), custom_dist.cdf(np.linspace(0, np.pi, num=500)))
plt.show()

x = np.linspace(0+epsilon, np.pi -epsilon, num=500)
order = np.argsort(x)
x = x[order]
y = custom_dist.cdf(x)
plt.plot(y, x)
plt.show()

def inverse_cdf(z):
    w = np.interp(z, xp=y, fp=x)
    return w

custom_samples = inverse_cdf(stats.uniform.rvs(loc=0, scale=1, size=5000))
plt.hist(custom_samples, bins=20)
plt.show()

#Schwerpunktsenergie 4*p_1*p_2=s mit p_1 approx 0.1 * E mit E=6500GeV, up quark
diff_WQ_omega = MC.diff_WQ_omega(s=200**2, q=1/3)
diff_WQ_omega_vale = MC.diff_WQ_omega_vale(E=200, q=1/3)
analytic_integral = np.mean(diff_WQ_omega(custom_samples)/custom_dist(custom_samples))
quad_integral = integrate.quad(diff_WQ_omega, a=epsilon, b=np.pi-epsilon)
analytic_integral_vale = np.mean(diff_WQ_omega_vale(custom_samples)/custom_dist(custom_samples))
quad_integral_vale = integrate.quad(diff_WQ_omega_vale, a=epsilon, b=np.pi-epsilon)
print("Ergebnis der MC-Integration für den WQ des partonsichen Prozesses:",analytic_integral, "1/GeV²")
print("Vergleich, Ergebnis mit quad:", quad_integral, "1/GeV²")
print("Ergebnis der MC-Integration für den WQ des partonsichen Prozesses:",analytic_integral_vale, "1/GeV²")
print("Vergleich, Ergebnis mit quad:", quad_integral_vale, "1/GeV²")
#1/GeV² approx 1/(10¹⁸eV²) approx 1e-18 * 1e-14 * 1,97m²
SI_analytic_integral = analytic_integral * 1e-32 * 1.97**2
barn_analytic_integral = SI_analytic_integral * 1e+28 * 1e+12
print("In SI Einheiten:", SI_analytic_integral, "m²", "In barn:", barn_analytic_integral, "pb" )
SI_analytic_integral_vale = analytic_integral_vale * 1e-32 * 1.97**2
barn_analytic_integral_vale = SI_analytic_integral_vale * 1e+28 * 1e+12
print("In SI Einheiten:", SI_analytic_integral_vale, "m²", "In barn:", barn_analytic_integral_vale, "pb" )

print(np.arccos(np.tanh(2.5)))
print(np.arccos(np.tanh(-2.5)))

analytic_expression= (np.pi * (constants.fine_structure)**2 * (1 /3)**4)/(3 * 200.0**2) * (np.tanh(-2.5) - np.tanh(2.5) - 2 * (-2.5 - 2.5))
theta_1, theta_2 = 0.16, np.pi-0.16
analytic_expression_theta = (np.pi * (constants.fine_structure)**2 * (1 /3)**4)/(3 * 200.0**2) * (np.cos(theta_2)- np.cos(theta_1) + 2*(1/np.tanh(np.cos(theta_1)) - 1/np.tanh(np.cos(theta_2))))
SI_analytic_expression = analytic_expression * 1e-32 * (1.97327)**2
gev_to_pb = 0.389379e9
barn_analytic_expression_theta = analytic_expression_theta * gev_to_pb
print("der analytische Wert:", analytic_expression, "1/GeV²")
barn_analytic_expression = analytic_expression * gev_to_pb
print("Eta: In SI Einheiten:", SI_analytic_expression, "m²", "In barn:", barn_analytic_expression, "pb" )
print("Theta: In GeV Einheiten:", analytic_expression_theta, "m²", "In barn:", barn_analytic_expression_theta, "pb" )

theta = np.linspace(np.pi/2-0.5, np.pi/2+0.5, 200)
dsigma_domega = diff_WQ_omega_vale(theta) * gev_to_pb
plt.plot(theta, dsigma_domega)
plt.show()

quad_integral_vale = integrate.quad(diff_WQ_omega_vale, a=epsilon, b=np.pi-epsilon)
barn_analytic_integral_vale = quad_integral_vale[0] * gev_to_pb
print("Vale in GeV:", quad_integral_vale[0])
print("Vale in barn", barn_analytic_integral_vale)

diff_WQ_eta_vale = MC.diff_WQ_eta_vale(E=200, q=1/3)
quad_integral_eta_vale = integrate.quad(diff_WQ_eta_vale, a=-2.5, b=2.5)
print("Vale, eta in GeV:", quad_integral_eta_vale)
barn_integral_eta_vale = quad_integral_eta_vale[0] * gev_to_pb
print("Vale, eta in pb:", barn_integral_eta_vale)
eta = np.linspace(-2.5, 2.5, 200)
dsigma_deta = diff_WQ_eta_vale(eta) * gev_to_pb
plt.plot(eta, dsigma_deta)
plt.show()

diff_WQ_eta = MC.diff_WQ_eta(s=40000, q=1/3)
quad_integral_eta = integrate.quad(diff_WQ_eta, a=-2.5, b=2.5)
print("eta in GeV:", quad_integral_eta)
barn_integral_eta = quad_integral_eta[0] * gev_to_pb
print("eta in pb:", barn_integral_eta)
eta = np.linspace(-2.5, 2.5, 200)
dsigma_deta = diff_WQ_eta(eta) * gev_to_pb
plt.plot(eta, dsigma_deta)
plt.show()


