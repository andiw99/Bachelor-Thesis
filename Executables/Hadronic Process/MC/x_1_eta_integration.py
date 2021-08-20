
import numpy as np
import scipy as sc
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from scipy import stats
from matplotlib import pyplot as plt
from scipy import integrate
import ml
import ast
import time
import MC
import lhapdf as pdf


def main():
    #Random Samples einlesen, und zwar alle und dann in bins einteilen:
    dataset_paths = list()
    dataset_paths.append("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/MC50M_I/")
    #dataset_paths.append("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/MC50M_II/")
    label_name = "WQ"
    with_ml = True

    #Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_paths[0] + "config")
    #model einlesen
    model_path = "//Files/Hadronic/Models/best_model"
    save_path = "//Plots/finished/"
    (model, transformer) = ml.import_model_transformer(model_path=model_path)

    #Variablen initialisieren (der Verteilungen):
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])

    # Wahrscheinlichkeitsverteilungen initialisieren
    scaling_loguni = 1 / (
            variables["x_upper_limit"] - variables["x_lower_limit"])
    loguni = MC.class_loguni(loguni_param=variables["loguni_param"],
                             x_lower_limit=variables["x_lower_limit"],
                             x_upper_limit=variables["x_upper_limit"],
                             scaling=scaling_loguni)

    # Anfangsdaten für x_2_interval
    features, labels = MC.data_handling(data_path=dataset_paths[0] + "all",
                                        label_name=label_name, return_pd=False)

    # Aus den x_2-Werten Bins machen
    # bins festlegen, pro bin ca. 50000 Punkte
    nr_bins = int(variables["total_data"]/500000)
    x_2_interval = np.exp(np.linspace(start=np.log(np.min(features[:,1])), stop=0, num=nr_bins+1))
    print("loguni cdf x_2_interval", loguni.cdf(x_2_interval))
    print("nr_bins", nr_bins)
    print("x_2_interval", x_2_interval)
    # Listen für die integrationen
    analytic_integrals = list()
    analytic_stddevs = list()
    ml_integrals = list()
    ml_stddevs = list()
    for i,dataset_path in enumerate(dataset_paths):
        #verschiedene eta Werte isolieren
        if i != 0:
            features, labels = MC.data_handling(data_path=dataset_path + "all", label_name=label_name, return_pd=False)
        config = pd.read_csv(dataset_path + "config")
        # Variablen initialisieren (der Verteilungen):
        variables = dict()
        for key in config:
            variables[key] = float(config[key][0])
        features_x_2_constant = dict()
        labels_x_2_constant = dict()
        predictions = dict()
        for i in range(nr_bins):
            # Die features in die bins aufteilen
            features_x_2_constant["{:.5f}".format(x_2_interval[i])] = features[(features[:,1] < x_2_interval[i+1]) & (features[:,1] > x_2_interval[i])]
            labels_x_2_constant["{:.5f}".format(x_2_interval[i])] = labels[(features[:,1] < x_2_interval[i+1]) & (features[:,1] > x_2_interval[i])]
        del features
        del labels

        if with_ml:
            for i in range(nr_bins):
                try:
                    predictions["{:.5f}".format(x_2_interval[i])] = transformer.retransform(model.predict(features_x_2_constant["{:.5f}".format(x_2_interval[i])]))
                except ValueError:
                    predictions["{:.5f}".format(x_2_interval[i])] = np.array([[0]])
                if i % 2 == 0:
                    print("predictions", predictions["{:.5f}".format(x_2_interval[i])])
                    print("labels", labels_x_2_constant["{:.5f}".format(x_2_interval[i])])
                    print("mean", np.mean(labels_x_2_constant["{:.5f}".format(x_2_interval[i])]))


        # Benötige ich die Gauß verteilung?
        er_fc = MC.erf(mu=0, sigma=variables["stddev"])
        scaling_gauss = 1 / (
                er_fc(variables["eta_limit"]) - er_fc(-variables["eta_limit"]))
        gauss = MC.gaussian(mu=0, sigma=variables["stddev"], scaling=scaling_gauss)

        I = integrate.quad(gauss, a=-variables["eta_limit"],
                           b=variables["eta_limit"])
        print("ist gauß normiert?", I)
        I2 = integrate.quad(loguni, a=variables["x_lower_limit"], b=variables["x_upper_limit"])
        print("Normierung loguni", I2)

        analytic_integral = np.zeros(shape=(nr_bins))
        ml_integral = np.zeros(shape=(nr_bins))
        quadratic_analytic_integral = np.zeros(shape=(nr_bins))
        quadratic_ml_integral = np.zeros(shape=(nr_bins))
        analytic_stddev = np.zeros(shape=(nr_bins))
        ml_stddev = np.zeros(shape=nr_bins)
        x_2 = np.zeros(shape=(nr_bins))
        print("wir sind vor der schleife die die MC-Integration macht")
        # ratio mit einbeziehen
        ratio = variables["x_total"]/variables["total_data"]
        print(features_x_2_constant.keys())
        for i,x_2_value in enumerate(features_x_2_constant.keys()):
            scaling = 1/(loguni.cdf(x_2_interval[i+1]) - loguni.cdf(x_2_interval[i])) * (x_2_interval[i+1] - x_2_interval[i])
            print("scaling", scaling, "interval", x_2_interval[i+1]-x_2_interval[i])
            print("ratio", ratio)
            analytic_integral[i] = tf.math.reduce_mean(labels_x_2_constant[x_2_value][:,0] /
                                                       (ratio * scaling * loguni(features_x_2_constant[x_2_value][:,0]) * loguni(features_x_2_constant[x_2_value][:,1])
                                                        * gauss(features_x_2_constant[x_2_value][:,2])))
            quadratic_analytic_integral[i] = tf.math.reduce_mean(tf.math.square(labels_x_2_constant[x_2_value][:,0] /
                                                                                (ratio * scaling * loguni(features_x_2_constant[x_2_value][:,0]) * loguni(features_x_2_constant[x_2_value][:,1])
                                                                                 * gauss(features_x_2_constant[x_2_value][:,2]))))
            analytic_stddev[i] = np.sqrt((quadratic_analytic_integral[i] - analytic_integral[i]**2) * 1/(len(labels_x_2_constant[x_2_value][:,0])-1))

            if with_ml:
                ml_integral[i] = tf.math.reduce_mean((predictions[x_2_value][:,0])/
                                                     (ratio * scaling * loguni(features_x_2_constant[x_2_value][:,0]) * loguni(features_x_2_constant[x_2_value][:,1]) *
                                                      gauss(features_x_2_constant[x_2_value][:,2])))
                quadratic_ml_integral[i] = (tf.math.reduce_mean(tf.math.square(predictions[x_2_value][:,0] / ( ratio * scaling * loguni(features_x_2_constant[x_2_value][:, 0])
                                                                                                               * gauss(features_x_2_constant[x_2_value][:,2]) * loguni(features_x_2_constant[x_2_value][:,1])))))

                ml_stddev[i] = np.sqrt((quadratic_ml_integral[i] - ml_integral[i]**2) * 1/(len(predictions[x_2_value][:,0])-1))
            x_2[i] = (x_2_interval[i+1] - x_2_interval[i])/2 + x_2_interval[i]
            print(x_2_value)

            if (i + 1) % 2 == 0:
                print("lables", labels_x_2_constant[x_2_value][:,0])
                if with_ml:
                    print("predictions", predictions[x_2_value][:,0])
                    print("ml_int", ml_integral[i])
                print("anl int", analytic_integral[i])
                print(i+1, "/", (nr_bins))

        #FUnktionswerte an listen anhängen
        analytic_integrals.append(analytic_integral)
        analytic_stddevs.append(analytic_stddev)
        ml_integrals.append(ml_integral)
        ml_stddevs.append(ml_stddev)

        # Speicher wieder freigeben
        del features_x_2_constant
        del labels_x_2_constant
        del predictions

    print("x_2", x_2)
    print("quadratic_analytic_int:", quadratic_analytic_integral)
    print("stddev:", analytic_stddev)
    print("stddev in pb:", MC.gev_to_pb(analytic_stddev))
    order = np.argsort(x_2)
    x_2 = np.array(x_2)[order]
    analytic_integral = np.mean(analytic_integrals, axis=0)
    analytic_stddev = np.mean(analytic_stddevs, axis=0)
    analytic_stddev_stat = 1/np.sqrt(len(analytic_integrals)) * np.std(analytic_integrals, axis=0, ddof=1)
    ml_integral = np.mean(ml_integrals, axis=0)
    ml_stddev = np.mean(ml_stddevs, axis=0)
    ml_stddev_stat =  1/np.sqrt(len(ml_integrals)) * np.std(ml_integrals, axis=0, ddof=1)

    print("x_2", x_2)
    print("analytic integrals", MC.gev_to_pb(np.array(analytic_integrals)))
    print("quadratic_analytic_int:", quadratic_analytic_integral)
    print("stddev:", analytic_stddev)
    print("stddev in pb:", MC.gev_to_pb(analytic_stddev))
    print("analytic stddev stat in pb", MC.gev_to_pb(analytic_stddev_stat))

    analytic_integral = np.array(analytic_integral)[order]
    ml_integral = np.array(ml_integral)[order]
    if not with_ml:
        ml_integral  = analytic_integral
    ml.make_MC_plot(x=x_2, analytic_integral=MC.gev_to_pb(analytic_integral), ml_integral=MC.gev_to_pb(ml_integral), xlabel=r"$x_2$",
                    analytic_errors=MC.gev_to_pb(analytic_stddev), ml_errors=MC.gev_to_pb(ml_stddev),
                    ylabel=r"$\frac{d\sigma}{dx_2}[\mathrm{pb}]$", save_path=save_path, name="x_1_etaIntegrationwobistdu", scale="log", xscale="log")
    plt.show()

    ml.make_MC_plot(x=x_2, analytic_integral=MC.gev_to_pb(analytic_integral),
                    ml_integral=MC.gev_to_pb(ml_integral), analytic_errors=MC.gev_to_pb(analytic_stddev_stat), ml_errors=MC.gev_to_pb(ml_stddev_stat),
                    xlabel=r"$x_2$",
                    ylabel=r"$\frac{d\sigma}{dx_2}[\mathrm{pb}]$", save_path=save_path,
                    name="x_1_etaIntegration", scale="log", xscale="log")
    plt.show()

    # TODO funktioniert das hier schon?

    print("ML_integral", ml_integral, "in pb", MC.gev_to_pb(ml_integral))
    print("analytic_integral in GeV", analytic_integral, "analytic integral in pb:", MC.gev_to_pb(analytic_integral))

    analytic_integrand = [analytic_integral[0]] + [(analytic_integral[i] + analytic_integral[i+1])/2 for i in range(len(analytic_integral) - 1)] + [analytic_integral[-1]]
    analytic_integrand = np.array(analytic_integrand)
    print(analytic_integrand)
    sigma_total = integrate.trapezoid(MC.gev_to_pb(analytic_integral), x_2)
    print("sigma_total in pb", sigma_total)
    sigma_total = integrate.simpson(MC.gev_to_pb(analytic_integral), x_2)
    print("simpson", sigma_total)
    sigma_total = 0
    for i in range(len(analytic_integral)):
        sigma_total += (x_2_interval[i+1] - x_2_interval[i]) * MC.gev_to_pb(analytic_integral[i])
    print("andere variante", sigma_total)
    sigma_total_2 = integrate.trapezoid(MC.gev_to_pb(analytic_integrand), x_2_interval)
    print("sigma total2 in pb", sigma_total_2)
if __name__ == "__main__":
    main()

