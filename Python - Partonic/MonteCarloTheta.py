import numpy as np
import tensorflow as tf
from tensorflow import keras
#Zuerst naiv mit uniformer distribution

def func(x):
    return (1 + np.cos(x) ** 2) / (np.sin(x) ** 2)

def Func(x):
    return -x - 2 * (1/np.tan(x))
#Model laden
model = keras.models.load_model(filepath="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/PartonicTheta/best_model")

#Variablen
N = 100000 #punkte
n = 100 #iterationen
lower_limit = 0.1
upper_limit = np.pi - 0.1

#Random numbers festlegen
xRand = np.random.uniform(low=lower_limit, high=upper_limit, size=N)
analytic_integral = 0
for i in range(n):
    analytic_integral += (upper_limit - lower_limit) * np.mean(func(xRand))
analytic_integral = 1/n * analytic_integral

xRand = tf.constant(xRand, shape=(1,N), dtype="float32")
xRand = tf.transpose(xRand)
ML_integral = 0

for i in range(n):
    ML_integral += (upper_limit - lower_limit) * tf.math.reduce_mean(model(xRand))
ML_integral = 1/n * ML_integral

integral = Func(upper_limit) - Func(lower_limit)
print("Monte Carlo integration mit analytischen Funktionswerten liefert:", "{:.3f}".format(analytic_integral))
print("Monte Carlo integration mit ML-Funktionswerten liefert:", "{:.3f}".format(ML_integral))
print("analytisch exakter wert zum vergleich:", "{:.3f}".format(integral))
