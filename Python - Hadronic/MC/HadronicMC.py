import numpy as np
import scipy as sc
import pandas as pd
import tensorflow as tf
from scipy import stats
from matplotlib import pyplot as plt
from scipy import integrate
import ml
import MC


def main():
    #Random Samples einlesen:
    dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC50M_I/"
    #Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_path + "config")
    #model einlesen
    model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_model"
    model, transformer = ml.import_model_transformer(model_path=model_path)

    #Variablen initialisieren:
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])

    (features, labels) = MC.data_handling(data_path= dataset_path + "all", label_name="WQ")
    # TODO lables zu arrays wegen item assignment
    labels = labels.numpy()

    print("labels:", labels)
    print("features:", features)
    print(np.mean(labels))
    print(np.max(labels))
    print(np.min(labels))

    # Phasenraumcuts durchziehen
    print("labels nach cut:", labels),
    print("mean nach cut", np.mean(labels))
    # Hier duerfte sich nix geandert haben, wenn der cut schon durchgefuehrt ist

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
    analytic_var = quadratic_analytic_integral - analytic_integral ** 2
    del labels

    #print("WKen", loguni(features[:,0]), loguni(features[:,1]), gauss(features[:,2]))
    print("analytic_integral", float(analytic_integral))

    predictions = transformer.retransform(model.predict(features))
    #predictions = np.zeros(shape=len(labels))
    print("predictions:",predictions)
    print(np.mean(predictions))
    #aus den predictions die schlechten events rauscutten
    predictions = np.array(predictions)
    print("predictions:", predictions)
    print(np.mean(predictions))
    ML_integral = tf.math.reduce_mean((predictions[:,0])/ \
                                         (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(features[:,2])))
    quadratic_analytic_integral = tf.math.reduce_mean(tf.math.square((predictions[:,0])/ \
                                            (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(np.abs(features[:,2])))))
    ml_std = np.sqrt(quadratic_analytic_integral - ML_integral ** 2) * 1/np.sqrt(len(predictions[:,0]))

    print("ML_integral in GeV", float(ML_integral), "in pb:", MC.gev_to_pb(float(ML_integral)))
    print("analytic_integral in GeV", float(analytic_integral), "in pb:", MC.gev_to_pb(float(analytic_integral)))
    print("stddev analytic integral in GeV²", np.sqrt(float(analytic_var))/np.sqrt(len(predictions[:,0])), "in pb:", MC.gev_to_pb(np.sqrt(float(analytic_var)))/np.sqrt(len(predictions[:,0])))
    print("stddev ml in pb:", MC.gev_to_pb(ml_std))

if __name__ == "__main__":
    main()
