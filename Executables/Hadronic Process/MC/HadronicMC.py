import numpy as np
import scipy as sc
import pandas as pd
import tensorflow as tf
from scipy import stats
from matplotlib import pyplot as plt
from scipy import integrate
import ml
import MC
import os


def main():
    #Random Samples einlesen:
    #Random Samples einlesen, und zwar alle und dann in bins einteilen:
    # eventuell mehrere datensets
    dataset_paths = list()
    directory = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/MC10M/"
    datasets = os.listdir(directory)
    for dataset in datasets:
        if dataset != "conifg":
            dataset_paths.append(directory + dataset + "/")

    max_iter = 20

    analytic_integrals = []
    analytic_stddevs = []
    ml_integrals = []
    ml_stddevs = []
    for i,path in enumerate(dataset_paths):
        if i == max_iter:
            break
        # Config der RandomSample generierung einlesen
        config = pd.read_csv(path + "config")
        # model einlesen
        model_path = "//Files/Hadronic/Models/best_model"
        model, transformer = ml.import_model_transformer(model_path=model_path)

        # Variablen initialisieren:
        variables = dict()
        for key in config:
            variables[key] = float(config[key][0])

        (features, labels) = MC.data_handling(data_path=path + "all", label_name="WQ")
        # TODO lables zu arrays wegen item assignment

        #Wahrscheinlichkeitsverteilung initialisieren und kalibriren
        er_fc = MC.erf(mu=variables["eta_limit"], sigma=variables["stddev"])
        #Verteilung herstellen aus der gezogen wurde
        scaling_gauss = 1/(2*(er_fc(variables["eta_limit"]) - er_fc(0)))
        gauss = MC.gaussian(mu = variables["eta_limit"], sigma=variables["stddev"], scaling=scaling_gauss)
        if variables["eta_gauss"]:
            er_fc = MC.erf(mu=0, sigma=variables["stddev"])
            scaling_gauss = 1/(er_fc(variables["eta_limit"]) - er_fc(-variables["eta_limit"]))
            gauss = MC.gaussian(mu=0, sigma=variables["stddev"], scaling=scaling_gauss)
        scaling_loguni = 1/(variables["x_upper_limit"]-variables["x_lower_limit"])
        loguni = MC.class_loguni(loguni_param=variables["loguni_param"], x_lower_limit=variables["x_lower_limit"], x_upper_limit=variables["x_upper_limit"],scaling=scaling_loguni)


        I = 2 * integrate.quad(gauss, a=0, b=variables["eta_limit"])
        I2 = integrate.quad(loguni, a=variables["x_lower_limit"], b=variables["x_upper_limit"])
        print("gauss norm:", I)
        print("loguni norm:", I2)

        ratio = (variables["x_total"]/variables["total_data"])
        print("ratio:", ratio)

        analytic_integral = tf.math.reduce_mean((labels[:,0])/ \
                                                (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(np.abs(features[:,2]))))
        quadratic_analytic_integral = tf.math.reduce_mean(tf.math.square((labels[:,0])/ \
                                                (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(np.abs(features[:,2])))))
        analytic_std = np.sqrt(quadratic_analytic_integral - analytic_integral ** 2) * 1/np.sqrt(len(labels[:,0]))
        del labels

        print("analytic_integral", float(analytic_integral))

        predictions = transformer.retransform(model.predict(features))
        print("predictions:",predictions)
        print(np.mean(predictions))
        predictions = np.array(predictions)
        print("predictions:", predictions)
        print(np.mean(predictions))
        ML_integral = tf.math.reduce_mean((predictions[:,0])/ \
                                             (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(features[:,2])))
        quadratic_analytic_integral = tf.math.reduce_mean(tf.math.square((predictions[:,0])/ \
                                                (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(np.abs(features[:,2])))))
        ml_std = np.sqrt(quadratic_analytic_integral - ML_integral ** 2) * 1/np.sqrt(len(predictions[:,0]))

        #Werte Appenden
        analytic_integrals.append(analytic_integral)
        analytic_stddevs.append(analytic_std)
        ml_integrals.append(ML_integral)
        ml_stddevs.append(ml_std)


    print("analytic_integrals", analytic_integrals)
    print("ml_integrals", ml_integrals)
    # mitteln und fehler berechnen
    analytic_integral = np.mean(analytic_integrals)
    ml_integral = np.mean(ml_integrals)
    analytic_std = 1/len(analytic_stddevs) * np.sqrt(np.sum(np.square(analytic_stddevs)))
    ml_std = 1/len(ml_stddevs) * np.sqrt(np.sum(np.square(ml_stddevs)))
    analytic_stat_std = np.std(analytic_integrals, ddof=1) * 1/np.sqrt(len(analytic_integrals))
    ml_stat_std = np.std(ml_integrals, ddof=1) * 1/np.sqrt(len(ml_integrals))



    print("ML_integral in GeV", float(ml_integral), "in pb:", MC.gev_to_pb(float(ml_integral)))
    print("analytic_integral in GeV", float(analytic_integral), "in pb:", MC.gev_to_pb(float(analytic_integral)))
    print("stddev analytic integral in pb", MC.gev_to_pb(analytic_std),
          "stat std", MC.gev_to_pb(analytic_stat_std))
    print("stddev ml integral in pb", MC.gev_to_pb(ml_std), "stat std", MC.gev_to_pb(ml_stat_std))

if __name__ == "__main__":
    main()
