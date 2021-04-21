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

quarks = {"quark": [1, 2, 3, 4],
          "charge": [-1/3,2/3, -1/3, 2/3]}
PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF = pdf.mkPDF("CT14nnlo", 0)

#Random Samples einlesen:
dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MCx20k/"
label_name = "WQ"
(data, features, labels, _, _, features_pd, labels_pd, _) = ml.data_handling(data_path=dataset_path + "all", label_name=label_name, return_pd=True)
#Grid einelsen zur Trapez-Integration
#testdata_path = "/Files/Hadronic/Data/MC_xIntegrationTrapez/"
#(_, test_features, test_labels, _, _, test_features_pd, test_labels_pd, _) = ml.data_handling(data_path=testdata_path + "all", label_name=label_name, return_pd=True)

#Config der RandomSample generierung einlesen
config = pd.read_csv(dataset_path + "config")
#model einlesen
model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/best_model"
(model, transformer) = ml.import_model_transformer(model_path=model_path)

#Variablen initialisieren (der Verteilungen):
variables = dict()
for key in config:
    variables[key] = float(config[key][0])

#verschiedene eta Werte isolieren
eta_values = list(set(features_pd["eta"]))
#x_2_values = list(set(test_features_pd["x_2"]))
#x_order = np.argsort(x_2_values)
#x_2_values = np.array(x_2_values)[x_order]
features_eta_constant = dict()
labels_eta_constant = dict()
test_features_eta_constant = dict()
test_labels_eta_constant = dict()
x_1_integration = np.array([])
x_1_integration_WQ = np.array([])
for eta_value in eta_values:
    print(features)
    print(features[:,2] == eta_value)
    exit()
    features_eta_constant["{:.2f}".format(eta_value)] = features[features[:,2] == eta_value]
    labels_eta_constant["{:.2f}".format(eta_value)] = labels[features[:,2] == eta_value]
    #test_features_eta_constant["{:.2f}".format(eta_value)] = test_features[test_features[:,2] == eta_value]
    #test_labels_eta_constant["{:.2f}".format(eta_value)] = test_labels[test_features[:,2] == eta_value]
    #Trapez integration vorbereiten zur Überprüfung von MC-Integration
    """
    for i,x_2_value in enumerate(x_2_values):
        if i == 0:
            x_1_integration = np.array(test_features_eta_constant["{:.2f}".format(eta_value)][:,0][test_features_eta_constant["{:.2f}".format(eta_value)][:,1] == x_2_value])
            x_1_integration_WQ = np.array(test_labels_eta_constant["{:.2f}".format(eta_value)][:,0][test_features_eta_constant["{:.2f}".format(eta_value)][:,1] == x_2_value])
        else:
            x_1_integration = np.stack((x_1_integration,
                                   test_features_eta_constant["{:.2f}".format(eta_value)][:,0][test_features_eta_constant["{:.2f}".format(eta_value)][:,1] == x_2_value]))
            x_1_integration_WQ = np.stack((x_1_integration_WQ, np.array(test_labels_eta_constant["{:.2f}".format(eta_value)][:, 0][
                                              test_features_eta_constant["{:.2f}".format(eta_value)][:,
                                              1] == x_2_value])))
    """

#TODO testing...
I = sc.integrate.trapezoid(y = x_1_integration_WQ, x = x_1_integration)
print(I)

#WQ für Ereignisse mit pt < 20 GeV auf 0 setzen
for eta_value in features_eta_constant:
    cut = MC.pt_cut(features_eta_constant[eta_value]) #cut liefert bool an jeder stelle ob pt<20 GeV ist und setzt dann WQ auf 0
    print(cut)
    exit()
    test_cut = MC.pt_cut(test_features_eta_constant[eta_value])
    #TODO: kurz in np.array umwandeln denn tf.constant unterstützt kein item assignment
    labels_eta_constant[eta_value] = np.array(labels_eta_constant[eta_value])
    test_labels_eta_constant[eta_value] = np.array(test_labels_eta_constant[eta_value])
    test_labels_eta_constant[eta_value][test_cut] = 0
    labels_eta_constant[eta_value][cut] = 0
    test_labels_eta_constant[eta_value] = tf.constant(test_labels_eta_constant[eta_value])
    labels_eta_constant[eta_value] = tf.constant(labels_eta_constant[eta_value])

er_fc = MC.erf(mu=3, sigma=variables["stddev"])

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
print("I2", I2)

analytic_integral = list()
dblquad_integral = list()
trapezoid_integral = list()
eta = list()
time1 = time.time()
for i,eta_value in enumerate(features_eta_constant.keys()):
    analytic_integral.append(tf.math.reduce_mean(labels_eta_constant[eta_value][:,0] /
                                                 (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1]))))

    #calc_WQ = MC.calc_diff_WQ(PDF=PDF, quarks=quarks, eta=float(eta_value))
    #dblquad_integral.append(integrate.nquad(calc_WQ, ranges=[[0,1],[0,1]]))
    print(integrate.trapezoid(test_labels_eta_constant[eta_value][:,0], test_features_eta_constant[eta_value][:,1]))
    trapezoid_integral.append(integrate.trapezoid(
        integrate.trapezoid(test_labels_eta_constant[eta_value][:,0], test_features_eta_constant[eta_value][:,1]),
        test_features_eta_constant[eta_value][:,0]))
    eta.append(float(eta_value))
    if i % 5 == 0:
        print(i, "/", len(eta_values))


order = np.argsort(eta)
eta = np.array(eta)[order]
analytic_integral = np.array(analytic_integral)[order]
plt.plot(eta, analytic_integral)
plt.plot(eta, trapezoid_integral)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\frac{d\sigma}{d\eta}$")
plt.tight_layout()
plt.show()
exit()

analytic_integral = tf.math.reduce_mean((labels[:,0])/ \
                                        (loguni(features[:,0]) * loguni(features[:,1])))
print("analytic_integral", analytic_integral)

features_batched = tf.data.Dataset.from_tensor_slices((features))
features_batched = features_batched.batch(batch_size=8096)
labels_batched = tf.data.Dataset.from_tensor_slices((labels))
labels_batched = labels_batched.batch(batch_size=8096)

ML_integral = 0
for batch in features_batched:
    """
    print(batch)
    print(loguni(batch[:,0]) * loguni(batch[:,1]) * gauss(batch[:,2]))
    print(transformer.retransform(model(batch))[:,0])
    for i,batch2 in enumerate(labels_batched):
        print(batch2)
        if i==0:
            break
    """
    ML_integral += tf.math.reduce_sum((transformer.retransform(model(batch))[:,0])/ \
                                     (loguni(batch[:,0]) * loguni(batch[:,1])))

ML_integral = (1/variables["total_data"]) * ML_integral
print("ML_integral", ML_integral)
print("analytic_integral", analytic_integral)


