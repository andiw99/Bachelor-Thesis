import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
#Zuerst naiv mit uniformer distribution

def func(x):
    return 1+(np.tanh(x))**2

def Func(x):
    return 2*x - np.tanh(x)

#Model laden
model = keras.models.load_model(filepath="/Files/Partonic/Models/PartonicEta/best_model")

#Variablen
N = 100000 #punkte
n = 100 #iterationen
lower_limit = -3
upper_limit = 3

analytic_integral = 0
ML_integral = 0
steps = []
for i in range(n):
    # Random numbers festlegen
    xRand = np.random.uniform(low=lower_limit, high=upper_limit, size=N)
    xRand = tf.constant(xRand, shape=(1, N), dtype="float32")
    xRand = tf.transpose(xRand)
    step = (upper_limit - lower_limit) * tf.math.reduce_mean(model(xRand))
    ML_integral += step
    steps.append(step)
    analytic_integral += (upper_limit - lower_limit) * tf.math.reduce_mean(func(xRand))

print(steps)

analytic_integral = 1/n * analytic_integral
ML_integral = 1/n * ML_integral
integral = Func(upper_limit) - Func(lower_limit)

print("Monte Carlo integration mit analytischen Funktionswerten liefert:", "{:.3f}".format(analytic_integral))
print("Monte Carlo integration mit ML-Funktionswerten liefert:", "{:.3f}".format(ML_integral))
print("analytisch exakter wert zum vergleich:", "{:.3f}".format(integral))

plt.hist(steps, bins=10, rwidth=0.8)
plt.show()