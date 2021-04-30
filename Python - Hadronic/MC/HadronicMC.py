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
    dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC2M_newgauss_tinyloguni/"
    #Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_path + "config")
    #model einlesen
    model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_guess_4M"
    model, transformer = ml.import_model_transformer(model_path=model_path)

    #Variablen initialisieren:
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])

    (_, features, labels, _, _, features_pd, labels_pd, _) = ml.data_handling(data_path= dataset_path + "all", label_name="WQ", return_pd=True, return_as_tensor=True)
    # TODO lables zu arrays wegen item assignment
    labels = labels.numpy()

    print("labels:", labels)
    print("features:", features)
    print(np.mean(labels))
    print(np.max(labels))
    print(np.min(labels))

    # Phasenraumcuts durchziehen
    (_,pt_cut) = MC.pt_cut(features, return_cut=True)
    (_, eta_cut) = MC.eta_cut(features, return_cut=True)
    labels[~pt_cut] = 0.0
    labels[~eta_cut] = 0.0
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


    #Wk-Verteilungen veranschaulichen
    probabilities_x = loguni(features_pd["x_1"])
    probabilities_eta = gauss(features_pd["eta"])

    order = np.argsort(loguni(features_pd["x_1"]))
    x_1 = np.array(features_pd["x_1"])[order]
    probabilities_x = np.array(probabilities_x)[order]

    order = np.argsort(features_pd["eta"])
    eta = np.array(features_pd["eta"])[order]
    probabilities_eta = np.array(probabilities_eta)[order]

    plt.plot(x_1, probabilities_x)
    plt.yscale("log")
    plt.show()

    plt.plot(eta, probabilities_eta)
    plt.show()

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
    #print("WKen", loguni(features[:,0]), loguni(features[:,1]), gauss(features[:,2]))
    print("analytic_integral", float(analytic_integral))


    features_batched = tf.data.Dataset.from_tensor_slices((features))
    features_batched = features_batched.batch(batch_size=8096)
    labels_batched = tf.data.Dataset.from_tensor_slices((labels))
    labels_batched = labels_batched.batch(batch_size=8096)


    predictions = transformer.retransform(model.predict(features))
    #predictions = np.zeros(shape=len(labels))
    print("predictions:",predictions)
    print(np.mean(predictions))
    #aus den predictions die schlechten events rauscutten
    predictions = np.array(predictions)
    predictions[~pt_cut] = 0.0
    predictions[~eta_cut] = 0.0
    print("predictions:", predictions)
    print(np.mean(predictions))
    ML_integral = tf.math.reduce_mean((predictions[:,0])/ \
                                         (ratio * loguni(features[:,0]) * loguni(features[:,1]) * gauss(features[:,2])))

    print("ML_integral in GeV", float(ML_integral), "in pb:", MC.gev_to_pb(float(ML_integral)))
    print("analytic_integral in GeV", float(analytic_integral), "in pb:", MC.gev_to_pb(float(analytic_integral)))
    print("stddev analytic integral in GeV²", np.sqrt(float(analytic_var))/np.sqrt(2000000), "in pb:", MC.gev_to_pb(np.sqrt(float(analytic_var)))/np.sqrt(2000000))


if __name__ == "__main__":
    main()
