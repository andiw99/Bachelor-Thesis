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
    dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC100M_newgauss/"
    label_name = "WQ"
    features, labels = MC.data_handling(data_path=dataset_path + "all", label_name=label_name, return_pd=False)

    #Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_path + "config")
    #model einlesen
    model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/LastRandomSearch/best_model"
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
    (model, transformer) = ml.import_model_transformer(model_path=model_path)

    #Variablen initialisieren (der Verteilungen):
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])

    #verschiedene eta Werte isolieren
    features_x_2_constant = dict()
    labels_x_2_constant = dict()
    predictions = dict()

    # Aus den x_2-Werten Bins machen
    # bins festlegen, pro bin ca. 50000 Punkte
    nr_bins = int(variables["total_data"]/750000)
    x_2_interval = np.exp(np.linspace(start=np.log(np.min(features[:,1])), stop=0, num=nr_bins+1))
    print("nr_bins", nr_bins)
    print("x_2_interval", x_2_interval)
    for i in range(nr_bins):
        # Die features in die bins aufteilen
        features_x_2_constant["{:.5f}".format(x_2_interval[i])] = features[(features[:,1] < x_2_interval[i+1]) & (features[:,1] > x_2_interval[i])]
        labels_x_2_constant["{:.5f}".format(x_2_interval[i])] = labels[(features[:,1] < x_2_interval[i+1]) & (features[:,1] > x_2_interval[i])]
        try:
            predictions["{:.5f}".format(x_2_interval[i])] = transformer.retransform(model.predict(features_x_2_constant["{:.5f}".format(x_2_interval[i])]))
        except ValueError:
            predictions["{:.5f}".format(x_2_interval[i])] = np.array([[0]])
        print(predictions["{:.5f}".format(x_2_interval[i])])
        print(labels_x_2_constant["{:.5f}".format(x_2_interval[i])])


    # Wahrscheinlichkeitsverteilungen initialisieren
    scaling_loguni = 1/(variables["x_upper_limit"]-variables["x_lower_limit"])
    print("scaling_loguni:", scaling_loguni)
    loguni = MC.class_loguni(loguni_param=variables["loguni_param"], x_lower_limit=variables["x_lower_limit"], x_upper_limit=variables["x_upper_limit"],scaling=scaling_loguni)

    # Benötige ich die Gauß verteilung?
    er_fc = MC.erf(mu=0, sigma=variables["stddev"])
    scaling_gauss = 1 / (
                er_fc(variables["eta_limit"]) - er_fc(-variables["eta_limit"]))
    gauss = MC.gaussian(mu=0, sigma=variables["stddev"], scaling=scaling_gauss)


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
        analytic_integral[i] = tf.math.reduce_mean(labels_x_2_constant[x_2_value][:,0] /
                                                     (ratio * loguni(features_x_2_constant[x_2_value][:,0])
                                                      * gauss(features_x_2_constant[x_2_value][:,2])))
        quadratic_analytic_integral[i] = tf.math.reduce_mean(tf.math.square(labels_x_2_constant[x_2_value][:,0] /
                                                     (ratio * loguni(features_x_2_constant[x_2_value][:,0])
                                                      * gauss(features_x_2_constant[x_2_value][:,2]))))
        analytic_stddev[i] = np.sqrt((quadratic_analytic_integral[i] - analytic_integral[i]**2) * 1/(len(labels_x_2_constant[x_2_value][:,0])-1))

        ml_integral[i] = tf.math.reduce_mean((predictions[x_2_value][:,0])/
                                              (ratio * loguni(features_x_2_constant[x_2_value][:,0]) *
                                                 gauss(features_x_2_constant[x_2_value][:,2])))
        quadratic_ml_integral[i] = (tf.math.reduce_mean(tf.math.square(predictions[x_2_value][:,0] / ( ratio * loguni(features_x_2_constant[x_2_value][:, 0])
                                                       * gauss(features_x_2_constant[x_2_value][:,2])))))

        ml_stddev[i] = np.sqrt((quadratic_ml_integral[i] - ml_integral[i]**2) * 1/(len(predictions[x_2_value][:,0])-1))
        x_2[i] = (x_2_interval[i+1] - x_2_interval[i])/2 + x_2_interval[i]
        print(x_2_value)

        if (i + 1) % 5 == 0:
            print("lables", labels_x_2_constant[x_2_value][:,0])
            print("predictions", predictions[x_2_value][:,0])
            print("anl int", analytic_integral[i])
            print("ml_int", ml_integral[i])
            print(i+1, "/", (nr_bins))

    print("x_2", x_2)
    print("quadratic_analytic_int:", quadratic_analytic_integral)
    print("stddev:", analytic_stddev)
    print("stddev in pb:", MC.gev_to_pb(analytic_stddev))
    analytic_stddev_mean = MC.gev_to_pb(analytic_stddev)
    ml_stddev_mean = MC.gev_to_pb(ml_stddev)
    order = np.argsort(x_2)
    x_2 = np.array(x_2)[order]
    analytic_integral = np.array(analytic_integral)[order]
    ml_integral = np.array(ml_integral)[order]
    ml.make_MC_plot(x=x_2, analytic_integral=MC.gev_to_pb(analytic_integral), ml_integral=MC.gev_to_pb(ml_integral), xlabel=r"$x_2$",
                    ylabel=r"$\frac{d\sigma}{dx_2}[pb]$", save_path=save_path, name="x_1_etaIntegration", scale="log")
    plt.show()

    # TODO funktioniert das hier schon?

    print("ML_integral", ml_integral, "in pb", MC.gev_to_pb(ml_integral))
    print("analytic_integral in GeV", analytic_integral, "analytic integral in pb:", MC.gev_to_pb(analytic_integral))


if __name__ == "__main__":
    main()

