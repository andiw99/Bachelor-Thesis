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
    #Random Samples einlesen:
    dataset_path = "//Files/Hadronic/Data/MCx500k_tiny_loguni/"
    label_name = "WQ"
    (data, features, labels, _, _, features_pd, labels_pd, _) = ml.data_handling(data_path=dataset_path + "all", label_name=label_name, return_pd=True)
    #Grid einelsen zur Trapez-Integration
    #testdata_path = "/Files/Hadronic/Data/MC_xIntegrationTrapez/"
    #(_, test_features, test_labels, _, _, test_features_pd, test_labels_pd, _) = ml.data_handling(data_path=testdata_path + "all", label_name=label_name, return_pd=True)

    #Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_path + "config")
    #model einlesen
    model_path = "//Files/Hadronic/Models/best_guess_4M"
    (model, transformer) = ml.import_model_transformer(model_path=model_path)

    #Variablen initialisieren (der Verteilungen):
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])

    #verschiedene eta Werte isolieren
    eta_values = list(set(features_pd["eta"]))
    features_eta_constant = dict()
    labels_eta_constant = dict()
    predictions = dict()
    #gesamte Messdaten in einzelne mit konstantem eta einteilen
    for eta_value in eta_values:

        features_eta_constant["{:.3f}".format(eta_value)] = features[features[:,2] == eta_value]
        labels_eta_constant["{:.3f}".format(eta_value)] = labels[features[:,2] == eta_value]


    #WQ für Ereignisse mit pt < 20 GeV  auf 0 setzen und solche die die eta cuts nicht erfüllen
    for eta_value in features_eta_constant:
        #TODO: kurz in np.array umwandeln denn tf.constant unterstützt kein item assignment
        (_, pt_cut) = MC.pt_cut(features_eta_constant[eta_value], return_cut=True)
        (_, eta_cut) = MC.eta_cut(features_eta_constant[eta_value], return_cut=True)
        labels_eta_constant[eta_value] = labels_eta_constant[eta_value].numpy()     #cut liefert bool an jeder stelle ob pt<20 GeV ist und setzt dann WQ auf 0
        labels_eta_constant[eta_value][~pt_cut] = 0.0
        labels_eta_constant[eta_value][~eta_cut] = 0.0
        labels_eta_constant[eta_value] = tf.constant(labels_eta_constant[eta_value])
        predictions[eta_value] = transformer.retransform(model.predict(features_eta_constant[eta_value]))
        predictions[eta_value] = predictions[eta_value]     #cut liefert bool an jeder stelle ob pt<20 GeV ist und setzt dann WQ auf 0
        predictions[eta_value][~pt_cut] = 0.0
        predictions[eta_value][~eta_cut] = 0.0
        predictions[eta_value] = tf.constant(predictions[eta_value])

    scaling_loguni = 1/(variables["x_upper_limit"]-variables["x_lower_limit"])
    print("scaling_loguni:", scaling_loguni)

    loguni = MC.class_loguni(loguni_param=variables["loguni_param"], x_lower_limit=variables["x_lower_limit"], x_upper_limit=variables["x_upper_limit"],scaling=scaling_loguni)

    probabilities_x = loguni(features_pd["x_1"])

    order = np.argsort(loguni(features_pd["x_1"]))
    x_1 = np.array(features_pd["x_1"])[order]
    probabilities_x = np.array(probabilities_x)[order]

    plt.plot(x_1, probabilities_x)
    plt.show()

    I2 = integrate.quad(loguni, a=variables["x_lower_limit"], b=variables["x_upper_limit"])
    print("Normierung loguni", I2)

    analytic_integral = np.zeros(shape=len(eta_values))
    ml_integral = np.zeros(shape=len(eta_values))
    quadratic_analytic_integral = np.zeros(shape=len(eta_values))
    quadratic_ml_integral = np.zeros(shape=len(eta_values))
    analytic_stddev = np.zeros(shape=len(eta_values))
    eta = np.zeros(shape=len(eta_values))
    print("wir sind vor der schleife die die MC-Integration macht")
    for i,eta_value in enumerate(features_eta_constant.keys()):
        analytic_integral[i] = (tf.math.reduce_mean(labels_eta_constant[eta_value][:,0] /
                                                     (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1]))))
        quadratic_analytic_integral[i] = tf.math.reduce_mean(tf.math.square(labels_eta_constant[eta_value][:,0] /
                                                     (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1]))))
        analytic_stddev[i] = np.sqrt(quadratic_analytic_integral[i] - analytic_integral[i]**2)

        ml_integral[i] = tf.math.reduce_mean((predictions[eta_value][:,0])/
                                              (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1])))
        quadratic_ml_integral[i] = (tf.math.reduce_mean(tf.math.square(predictions[eta_value][:,0] /
            (loguni(features_eta_constant[eta_value][:, 0]) * loguni(
                features_eta_constant[eta_value][:, 1])))))

        eta[i] = eta_value

        if (i + 1) % 2 == 0:
            print(i+1, "/", len(eta_values))

    print("quadratic_analytic_int:", quadratic_analytic_integral)
    print("stddev:", analytic_stddev)
    print("stddev in pb:", MC.gev_to_pb(analytic_stddev))
    analytic_stddev = 0
    order = np.argsort(eta)
    eta = np.array(eta)[order]
    analytic_integral = np.array(analytic_integral)[order]
    ml_integral = np.array(ml_integral)[order]
    fig, ax = plt.subplots()
    ax.errorbar(x=eta, y=MC.gev_to_pb(analytic_integral), yerr= MC.gev_to_pb(analytic_stddev),
                linewidth=0, elinewidth=0.8, ecolor="blue", marker=".",
                label="analytic")
    ax.errorbar(x=eta, y=MC.gev_to_pb(ml_integral), linewidth=0, marker="x",
                label="ML")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\frac{d\sigma}{d\eta}$")
    ax.legend()
    plt.show()





    ML_integral = 0

    ML_integral = (1/variables["total_data"]) * ML_integral
    print("ML_integral", ML_integral)
    print("analytic_integral in GeV", analytic_integral, "analytic integral in pb:", MC.gev_to_pb(analytic_integral))


if __name__ == "__main__":
    main()

