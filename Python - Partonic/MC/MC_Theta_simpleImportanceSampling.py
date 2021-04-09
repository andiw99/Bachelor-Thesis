import numpy as np
import scipy as sc
from scipy import integrate
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import ml
import ast
#Zuerst naiv mit uniformer distribution

def func(x):
    return (1 + np.cos(x) ** 2) / (np.sin(x) ** 2)

def Func(x):
    return -x - 2 * (1/np.tan(x))
#Model laden
model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/PartonicTheta/best_model"
model = keras.models.load_model(filepath=model_path)
config = pd.read_csv(model_path + "/config")
config = config.transpose()
transformer_config = ast.literal_eval(config[8][1])
transformer = ml.LabelTransformation(config=transformer_config)

#Variablen
N = 1000 #punkte
n = 100 #iterationen
lower_limit = 0.1
upper_limit = np.pi - 0.1
sigma = 0.2

class gaussian():
    def __init__(self, mu = 0, sigma=1, scaling = 1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling

    def __call__(self, x):
        y = self.scaling * 1/(np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-(x - self.mu)**2/(2*self.sigma**2))
        return y

class erf():
    def __init__(self, mu=0, sigma=1, scaling=1):
        self.mu = mu
        self.sigma = sigma
        self.scaling = scaling
    def __call__(self, x):
        y = self.scaling * 1/2 * (1+ sc.special.erf((x-self.mu)/(np.sqrt(2*(self.sigma**2)))))
        return y

ML_var_list=[]
analytic_var_list=[]
sigma_list=[]

for sigma in np.linspace(0.10,0.60, 20):
    #sigma printen
    print("sigma:", sigma)


    er_fc = erf(mu=0, sigma=sigma)

    scaling = 1/(2*(er_fc(np.pi/2) - er_fc(lower_limit)))

    lower_gauss = gaussian(mu=0, sigma=sigma, scaling=scaling)
    upper_gauss = gaussian(mu=np.pi, sigma=sigma, scaling=scaling)
    """
    lower_gauss_2 = gaussian(mu=0, sigma=sigma)
    I3 = integrate.quad(lower_gauss_2, a=lower_limit, b = np.pi/2)
    I4 = []
    for b in np.linspace(-5,5,20):
        I4.append(integrate.quad(lower_gauss_2, a=-np.inf, b=b)[0])
    print(I4)
    plt.plot(np.linspace(-5,5,50), er_fc(np.linspace(-5,5,50)))
    plt.plot(np.linspace(-5,5,20), I4 )
    plt.show()
    print(I3)
    exit()
    """
    def g(x):
        if x <= np.pi/2:
            return lower_gauss(x)
        else:
            return upper_gauss(x)

    I = integrate.quad(lower_gauss, a=lower_limit, b=np.pi/2)
    I2 = integrate.quad(upper_gauss, a = np.pi/2, b=upper_limit)

    a = tf.constant([0.2, 1, 1.5])
    b = a/lower_gauss(a)

    analytic_integral_lower = 0
    analytic_integral_upper = 0
    ML_integral_lower = 0
    ML_integral_upper = 0
    ML_integral_quad_lower = 0
    ML_integral_quad_upper = 0
    analytic_integral_quad_lower = 0
    analytic_integral_quad_upper = 0
    for i in range(n):
        # Random numpy arrays erstellen
        xRand_lower = np.array([])
        xRand_upper = np.array([])

        while xRand_lower.size < int(N / 2) or xRand_upper.size < int(N / 2):
            xRand_lower = np.append(xRand_lower,
                                    abs(np.random.normal(loc=0, scale=sigma, size=int(N / 2 - xRand_lower.size))))
            xRand_upper = np.append(xRand_upper, -abs(
                np.random.normal(loc=0, scale=sigma, size=int(N / 2 - xRand_upper.size))) + np.pi)
            # Werte unter lower_limit und über upper_limit herauswerfen, Schichten abgrenzen
            xRand_lower = xRand_lower[xRand_lower <= np.pi / 2]
            xRand_lower = xRand_lower[xRand_lower >= lower_limit]
            xRand_upper = xRand_upper[xRand_upper > np.pi / 2]
            xRand_upper = xRand_upper[xRand_upper <= upper_limit]

        # numpy arrays zu tf.tensoren machen
        xRand_lower = tf.constant(xRand_lower, shape=(1, int(N / 2)), dtype="float32")
        xRand_upper = tf.constant(xRand_upper, shape=(1, int(N / 2)), dtype="float32")
        xRand_lower = tf.transpose(xRand_lower)
        xRand_upper = tf.transpose(xRand_upper)

        if i == 1:
            xRand = np.append(xRand_lower, xRand_upper)
            plt.hist(xRand, bins=20, rwidth=0.8)
            plt.title(r"$\sigma:$ {0:3.2f}".format(sigma))
            plt.show()

        #integral im unteren bereich berechnen und durch anzahl der segmente teilen!!!!
        ML_integral_lower += 1/2 * tf.math.reduce_mean(transformer.retransform(model(xRand_lower))/lower_gauss(xRand_lower))
        analytic_integral_lower += 1/2 *  tf.math.reduce_mean(func(xRand_lower)/lower_gauss(xRand_lower))
        #integral des quadrats im unterren bereich berechnen
        ML_integral_quad_lower += 1/2 * tf.math.reduce_mean(tf.math.square(transformer.retransform(model(xRand_lower))/lower_gauss(xRand_lower)))
        analytic_integral_quad_lower += 1/2 * tf.math.reduce_mean(tf.math.square(func(xRand_lower)/lower_gauss(xRand_lower)))

        #integral im oberen bereich berechnen
        ML_integral_upper += 1/2 * tf.math.reduce_mean(transformer.retransform(model(xRand_upper))/upper_gauss(xRand_upper))
        analytic_integral_upper += 1/2 * tf.math.reduce_mean(func(xRand_upper)/upper_gauss(xRand_upper))
        #integral des quadrats im oberen bereich berechnen
        ML_integral_quad_upper += 1/2 * tf.math.reduce_mean(tf.math.square(transformer.retransform(model(xRand_upper))/upper_gauss(xRand_upper)))
        analytic_integral_quad_upper += 1/2 * tf.math.reduce_mean(tf.math.square(func(xRand_upper)/upper_gauss(xRand_upper)))

    ML_integral = 1/n * (ML_integral_lower + ML_integral_upper)
    ML_integral_quad = 1/n * (ML_integral_quad_lower + ML_integral_quad_upper)
    #print("ML_integral_quad", ML_integral_quad, "ML_integral_quad_lower", ML_integral_quad_lower, "ML_integral_quad_upper", ML_integral_quad_upper)
    analytic_integral = 1/n * (analytic_integral_lower + analytic_integral_upper)
    analytic_integral_quad = 1/n * (analytic_integral_quad_lower + analytic_integral_quad_upper)
    ML_Var = ML_integral_quad - ML_integral**2
    analytic_Var = analytic_integral_quad - analytic_integral**2

    sigma_list.append(sigma)
    ML_var_list.append(ML_Var)
    analytic_var_list.append(analytic_Var)

    integral = Func(upper_limit) - Func(lower_limit)

    print("Monte Carlo integration mit analytischen Funktionswerten liefert:", "{:.3f}".format(analytic_integral))
    print("Monte Carlo integration mit ML-Funktionswerten liefert:", "{:.3f}".format(ML_integral))
    print("analytisch exakter wert zum vergleich:", "{:.3f}".format(integral))
    print("ML_Varianz:", float(ML_Var))
    print("analytiv_varianz", float(analytic_Var))

plt.plot(sigma_list, ML_var_list)
plt.plot(sigma_list, analytic_var_list)
plt.yscale("log")
plt.show()