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


#Random Samples einlesen:
dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/MC_xIntegration/"
label_name = "WQ"
(data, features, labels, _, _, features_pd, labels_pd, _) = ml.data_handling(data_path=dataset_path + "all", label_name=label_name, return_pd=True)
#Config der RandomSample generierung einlesen
config = pd.read_csv(dataset_path + "config")
#model einlesen
model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/best_model"
(model, transformer) = ml.import_model_transformer(model_path=model_path)

#Variablen initialisieren(der Verteilungen):
variables = dict()
for key in config:
    variables[key] = float(config[key][0])

#verschiedene eta Werte isolieren
eta_values = list(set(features_pd["eta"]))
features_eta_constant = dict()
labels_eta_constant = dict()
for eta_value in eta_values:
    features_eta_constant["{:.2f}".format(eta_value)] = features[features[:,2] == eta_value]
    labels_eta_constant["{:.2f}".format(eta_value)] = labels[features[:,2] == eta_value]
    print(features_eta_constant["{:.2f}".format(eta_value)])
    print(labels_eta_constant["{:.2f}".format(eta_value)])
    exit()

er_fc = MC.erf(mu=3, sigma=variables["stddev"])

scaling_loguni = 1/((stats.loguniform.cdf(x=variables["x_upper_limit"], a = variables["loguni_param"], b =1+ variables["loguni_param"]) - \
                       stats.loguniform.cdf(x=variables["x_lower_limit"], a = variables["loguni_param"], b =1+ variables["loguni_param"])))

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
eta = list()
time1 = time.time()
for i,eta_value in enumerate(features_eta_constant.keys()):
    """
    print(labels_eta_constant[eta_value][:,0])
    print((loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1])))
    print(loguni(features_eta_constant[eta_value][:,0]).shape)
    time2 = time.time()
    time = time2 - time1
    a = labels_eta_constant[eta_value][:,0] / (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1]))
    print("a", a, "shape a", a.shape)
    print(time)
    exit()
    """
    analytic_integral.append(tf.math.reduce_mean(labels_eta_constant[eta_value][:,0] /
                                                 (loguni(features_eta_constant[eta_value][:,0]) * loguni(features_eta_constant[eta_value][:,1]))))
    eta.append(float(eta_value))
    if i % 25 == 0:
        print(i, "/", len(eta_values))


order = np.argsort(eta)
eta = np.array(eta)[order]
analytic_integral = np.array(analytic_integral)[order]
plt.plot(eta, analytic_integral)
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


