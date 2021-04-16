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


#Random Samples einlesen:
dataset_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/MC/"
data = pd.read_csv(dataset_path + "all")
#Config der RandomSample generierung einlesen
config = pd.read_csv(dataset_path + "config")
#model einlesen
model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/best_model"
model = keras.models.load_model(filepath=model_path)
model_config = pd.read_csv(model_path + "/config")
model_config = model_config.transpose()
transformer_config = ast.literal_eval(model_config["transformer_config"][0])
transformer = ml.label_transformation(config=transformer_config)

#Variablen initialisieren:
variables = dict()
for key in config:
    variables[key] = float(config[key][0])

features_pd = data.copy()
labels_pd = features_pd.pop("WQ")

#Labels unf features zu tf tensoren machen
for i, feature in enumerate(features_pd):
    if i == 0:
        features = tf.constant([features_pd[feature]], dtype="float32")
    else:
        more_features = tf.constant([features_pd[feature]], dtype="float32")
        features = tf.experimental.numpy.append(features, more_features, axis=0)
labels = tf.constant([labels_pd])

features = tf.transpose(features)
labels = tf.transpose(labels)

#Wahrscheinlichkeitsverteilung initialisieren
class class_loguni():
    def __init__(self, loguni_param, x_lower_limit, x_upper_limit, scaling=1):
        self.a = loguni_param
        self.b = 1 + loguni_param
        self.loguni_param = loguni_param
        self.x_lower_limit = x_lower_limit
        self.x_upper_limit = x_upper_limit
        self.scaling = scaling

    def __call__(self, x):
        #retransform x
        x = (x - self.x_lower_limit)/(self.x_upper_limit - self.x_lower_limit) + self.loguni_param
        x = self.scaling * stats.loguniform.pdf(x, self.a, self.b)
        return x

class gaussian():
    def __init__(self, mu = 0, sigma=1, scaling=1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling

    def __call__(self, x):
        y = self.scaling * 1/(np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-(tf.math.abs(x) - self.mu)**2/(2*self.sigma**2))
        return y

class erf():
    def __init__(self, mu=0, sigma=1, scaling=1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling
    def __call__(self, x):
        y = self.scaling * 1/2 * (1+ sc.special.erf((x-self.mu)/(np.sqrt(2*(self.sigma**2)))))
        return y

er_fc = erf(mu=3, sigma=variables["stddev"])

scaling_gauss = 1/(2*(er_fc(variables["eta_limit"]) - er_fc(0)))
scaling_loguni = 1/((stats.loguniform.cdf(x=variables["x_upper_limit"], a = variables["loguni_param"], b =1+ variables["loguni_param"]) - \
                       stats.loguniform.cdf(x=variables["x_lower_limit"], a = variables["loguni_param"], b =1+ variables["loguni_param"])))

loguni = class_loguni(loguni_param=variables["loguni_param"], x_lower_limit=variables["x_lower_limit"], x_upper_limit=variables["x_upper_limit"],scaling=scaling_loguni)
gauss = gaussian(mu = variables["eta_limit"], sigma=variables["stddev"], scaling=scaling_gauss)
probabilities_x = loguni(features_pd["x_1"])
probabilities_eta = gauss(features_pd["eta"])

order = np.argsort(loguni(features_pd["x_1"]))
x_1 = np.array(features_pd["x_1"])[order]
probabilities_x = np.array(probabilities_x)[order]

order = np.argsort(features_pd["eta"])
eta = np.array(features_pd["eta"])[order]
probabilities_eta = np.array(probabilities_eta)[order]

plt.plot(x_1, probabilities_x)
plt.show()

plt.plot(eta, probabilities_eta)
plt.show()

I = integrate.quad(gauss, a=-variables["eta_limit"], b=variables["eta_limit"])
I2 = integrate.quad(loguni, a=variables["x_lower_limit"], b=variables["x_upper_limit"])

print(features[:,0])
print(labels[:,0])


analytic_integral = tf.math.reduce_mean((labels[:,0])/ \
                                        (loguni(features[:,0]) * loguni(features[:,1]) * gauss(features[:,2])))
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
                                     (loguni(batch[:,0]) * loguni(batch[:,1]) * gauss(batch[:,2])))

ML_integral = (1/variables["total_data"]) * ML_integral
print("ML_integral", ML_integral)
print("analytic_integral", analytic_integral)


