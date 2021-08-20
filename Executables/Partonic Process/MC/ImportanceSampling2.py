import numpy as np
import pandas as pd
from scipy import stats
from scipy import constants
from matplotlib import pyplot as plt
from tensorflow import keras
import MC
import ml


class omega_integrand():
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x) * np.sin(x)

def main():
    #modelle laden
    eta_model, eta_transformer = ml.load_model_and_transformer(model_path="//Files/Partonic/Models/PartonicEta/best_model")
    theta_model, theta_transformer = ml.load_model_and_transformer(model_path="//Files/Partonic/Models/PartonicTheta/theta_model_important_range")
    save_path = "//Results/"
    name = "partonic_mc_int"
    uniform_dist = stats.uniform.rvs(loc = 0, scale = 1, size=200)
    plt.hist(uniform_dist, bins=20)
    plt.show()
    epsilon = 0.1638
    offset = 0.2

    n_theta = 100
    theta_size = 1000
    n_eta = 100
    eta_size = 1000

    custom_dist = MC.x_power_dist(power=4, offset=offset, a=epsilon, b=np.pi-epsilon, normed=True)
    # analytische theta werte
    diff_WQ_theta = MC.diff_WQ_theta(s=40000, q=1 / 3)
    theta_WQ = np.zeros(shape=n_theta)
    theta_WQ_no_IS = np.zeros(shape=n_theta)
    theta_WQ_mean_std = np.zeros(shape=n_theta)
    for i in range(n_theta):
        custom_samples = custom_dist.rvs(size=theta_size)
        uniform_samples = stats.uniform.rvs(loc=epsilon, scale=np.pi - 2 * epsilon, size=theta_size)
        #totalen WQ berechnen, integration über theta und über phi (*2pi)
        sigma_total_mc = MC.gev_to_pb(np.mean(1/2 * diff_WQ_theta(custom_samples)/custom_dist(custom_samples)))
        sigma_total_mc_mean_std = np.sqrt(1/(len(custom_samples) - 1) * (np.mean(np.square(1/2 * MC.gev_to_pb(diff_WQ_theta(custom_samples))/custom_dist(custom_samples))) - sigma_total_mc ** 2))
        sigma_total_mc_no_IS = MC.gev_to_pb(np.mean(1/2 * diff_WQ_theta(uniform_samples)/
                                                    stats.uniform.pdf(x=uniform_samples, loc=epsilon, scale=np.pi - 2 * epsilon)))
        theta_WQ[i] = sigma_total_mc
        theta_WQ_no_IS[i] = sigma_total_mc_no_IS
        theta_WQ_mean_std[i] = sigma_total_mc_mean_std
    plt.hist(theta_WQ, bins=30)
    plt.show()
    sigma_total_mc = np.mean(theta_WQ)
    sigma_total_mc_error = 1/np.sqrt(len(theta_WQ)- 1) * np.std(theta_WQ, ddof=1)
    sigma_total_mc_std = 1/len(theta_WQ_mean_std) * np.sqrt(np.sum(np.square(theta_WQ_mean_std)))
    sigma_total_mc_no_IS = np.mean(theta_WQ_no_IS)
    sigma_total_mc_error_no_IS = 1/np.sqrt(len(theta_WQ_no_IS)- 1) * np.std(theta_WQ_no_IS, ddof=1)
    print(theta_WQ_mean_std)
    print("theta sigma total mit IS in pb", sigma_total_mc, "+-", sigma_total_mc_error, "(+-", sigma_total_mc_std, ")")
    print("theta sigma total ohne IS in pb", sigma_total_mc_no_IS, "+-",
          sigma_total_mc_error_no_IS)

    # ml theta werte
    theta_WQ = np.zeros(shape=n_theta)
    theta_WQ_no_IS = np.zeros(shape=n_theta)
    theta_WQ_mean_std = np.zeros(shape=n_theta)
    for i in range(n_theta):
        custom_samples = custom_dist.rvs(size=theta_size)
        uniform_samples = stats.uniform.rvs(loc=epsilon, scale=np.pi - 2 * epsilon, size=theta_size)
        #totalen WQ berechnen, integration über theta und über phi (*2pi)
        sigma_total_mc_ml = MC.gev_to_pb(np.mean(1/2 * theta_transformer.retransform(theta_model.predict(custom_samples))[:,0]/custom_dist(custom_samples)))
        sigma_total_mc_mean_std = np.sqrt(1/(len(custom_samples) - 1) * (np.mean(np.square(1/2 * MC.gev_to_pb(theta_transformer.retransform(theta_model.predict(custom_samples))[:,0])/custom_dist(custom_samples))) - sigma_total_mc ** 2))
        sigma_total_mc_no_IS_ml = MC.gev_to_pb(np.mean(1/2 * theta_transformer.retransform(theta_model.predict(uniform_samples))[:,0]/
                                                    stats.uniform.pdf(x=uniform_samples, loc=epsilon, scale=np.pi - 2 * epsilon)))
        theta_WQ[i] = sigma_total_mc_ml
        theta_WQ_no_IS[i] = sigma_total_mc_no_IS_ml
        theta_WQ_mean_std[i] = sigma_total_mc_mean_std
    plt.hist(theta_WQ, bins=30)
    plt.show()
    sigma_total_mc_ml = np.mean(theta_WQ)
    sigma_total_mc_ml_error = 1/np.sqrt(len(theta_WQ)- 1) * np.std(theta_WQ, ddof=1)
    sigma_total_mc_ml_std = 1/len(theta_WQ_mean_std) * np.sqrt(np.sum(np.square(theta_WQ_mean_std)))
    sigma_total_mc_ml_no_IS = np.mean(theta_WQ_no_IS)
    sigma_total_mc_ml_error_no_IS = 1/np.sqrt(len(theta_WQ_no_IS)- 1) * np.std(theta_WQ_no_IS, ddof=1)
    print(theta_WQ_mean_std)
    print("theta sigma total mit IS in pb", sigma_total_mc_ml, "+-", sigma_total_mc_ml_error, "(+-", sigma_total_mc_ml_std, ")")
    print("theta sigma total ohne IS in pb", sigma_total_mc_ml_no_IS, "+-",
          sigma_total_mc_ml_error_no_IS)
    #totalen WQ berechnen mit integration über eta
    diff_WQ_eta = MC.diff_WQ_eta(s=40000, q=1/3)
    eta_WQ = np.zeros(shape=n_eta)
    eta_WQ_ml = np.zeros(shape=n_eta)
    for i in range(n_eta):
        eta_integration_dist = stats.uniform.rvs(loc=-2.5 , scale=5, size=eta_size)
        sigma_total_eta_mc =  MC.gev_to_pb(np.mean(diff_WQ_eta(eta_integration_dist)/(stats.uniform.pdf(x=eta_integration_dist, loc=-2.5, scale=5))))
        eta_WQ[i] = sigma_total_eta_mc
        sigma_total_eta_mc_ml = np.mean(eta_transformer.retransform(eta_model.predict(eta_integration_dist))/(stats.uniform.pdf(x=eta_integration_dist, loc=-2.5, scale=5)))
        eta_WQ_ml[i] = sigma_total_eta_mc_ml

    sigma_total_eta_mc = np.mean(eta_WQ)
    sigma_total_eta_mc_error = 1/np.sqrt(len(eta_WQ)- 1) * np.std(eta_WQ, ddof=1)
    sigma_total_eta_mc_ml = np.mean(eta_WQ_ml)
    sigma_total_eta_mc_ml_error = 1/np.sqrt(len(eta_WQ_ml) - 1) * np.std(eta_WQ_ml, ddof=1)
    print("sigma total mit mc in pb", sigma_total_eta_mc, "+-", sigma_total_eta_mc_error)
    print("sigma total mit mc und ml in pb", sigma_total_eta_mc_ml, "+-", sigma_total_eta_mc_ml_error)


    #analytischer Wert
    analytic_expression = MC.gev_to_pb((np.pi * (constants.fine_structure) ** 2 * (
                1 / 3) ** 4) / (3 * 200.0 ** 2) * (
                                      np.tanh(-2.5) - np.tanh(2.5) - 2 * (
                                          -2.5 - 2.5)))
    print("analytischer Wert", analytic_expression)

    results = pd.DataFrame(
        {
            "analytischer Wert": analytic_expression,
            "sigma theta": sigma_total_mc,
            "sigma theta error": sigma_total_mc_error,
            "sigma theta std": sigma_total_mc_std,
            "sigma theta no IS": sigma_total_mc_no_IS,
            "sigma theta no IS error": sigma_total_mc_error_no_IS,
            "sigma theta ml": sigma_total_mc_ml,
            "sigma theta ml error": sigma_total_mc_ml_error,
            "sigma theta ml std": sigma_total_mc_ml_std,
            "sigma eta": sigma_total_eta_mc,
            "sigma eta error": sigma_total_eta_mc_error,
            "sigma eta ml": sigma_total_eta_mc_ml,
            "sigma eta ml error": sigma_total_eta_mc_ml_error
        },
        index = [0]
    )
    results = results.transpose()
    results.to_csv(save_path+name)


if __name__ == "__main__":
    main()

