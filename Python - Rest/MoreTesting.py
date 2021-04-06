import tensorflow as tf
#from tensorflow import keras
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import numpy as np
import math
"""
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)

print(dataset)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()
print(loss_fn.name)
print(optimizer._name)

train_frac = 0.8
nr_hidden_layers = 3
batch_size = 5

parameters = pd.DataFrame(
    {
        "train_frac": [train_frac],
        "nr_hidden_layers": [nr_hidden_layers],
        "batch_size": [batch_size],
    }
)

name = str()
for elem in parameters:
    name += str(parameters[elem][0]) + "__"
    print(name)

print(name)

y_true = [[1.], [2.], [3.]]
y_pred = [[0.], [0.], [2.]]
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

print("mae:", mae(y_true, y_pred).numpy())
print("mse:", mse(y_true, y_pred).numpy())

print(tf.nn.relu.__name__)

print(keras.initializers.RandomNormal.get_config())
"""
xRand = np.abs(random.normal(loc=0, scale=0.15, size=3600))
xRand = xRand/(2*np.max(xRand))
xRand2 = np.abs(random.normal(loc=0, scale=0.2, size=400))
xRand2 = xRand2/(2*np.max(xRand2))+0.5
plt.hist(xRand, bins=10, rwidth=0.8)
plt.show()
plt.hist(xRand2, bins=10, rwidth=0.8)
plt.show()
xRand = np.append(xRand, xRand2)
print(xRand)

plt.hist(xRand, bins=20, rwidth=0.8)
plt.show()

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3]])
print(a,b)

print(np.append(a,b, axis=0))

a = tf.constant([[1,2,3]], dtype="float32")
b= tf.constant([[1,2,3]], dtype="float32")
print(a)
print(b)
print(tf.experimental.numpy.append(a,b, axis=0))

dict = {
    "hallo": 2,
    "hello": 1
}
print(list(dict.keys()))
print(list(dict.values()))


learning_rate = 5

config = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/PartonicTheta/Logarithm+MAE" + "/config")
config = config.transpose()
print("config[2][2]:", type(config[2][2]))
i=0
print(float("nan"), type(float("nan")))

while pd.notna(config[2][i]):
    i += 1
print(i)
exit()

config[2][2] = learning_rate
print(config[2])
exit()
config = config.transpose()
print(config)