import tensorflow as tf
from tensorflow import keras
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import numpy as np
import math
import scipy as sc
from scipy import integrate
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


config = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/Scaling+logarithm+MSE+leaky_relu/config")
print(config)
#print(config["learning_rate"])
config = config.transpose()
print(config)
exit()
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


def func():
    return (1,2,3)

_, b, _ = func()
print(b)

loss_fn = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.NONE)
loss_fn2 = keras.losses.MeanAbsoluteError()

y_true = tf.transpose(tf.constant([[1.0, 2.0, 3.0]]))
y_pred = tf.transpose(tf.constant([[1.0, 1.0, 4.0]]))

loss = loss_fn(y_true=y_true, y_pred=y_pred)
loss2 = loss_fn2(y_true=y_true, y_pred=y_pred)

print(loss, loss2)


#read_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/SourceModel6/config"
read_path = "config"
learning_rate = 5
config = pd.read_csv(read_path, index_col="property")

print(config)
config = config.transpose()
print(config.keys())
print(config["transformer_config"][0])
config = config.transpose()

config.to_csv("config", index=True, header=True, index_label="property")

a = tf.constant([[1,-2], [2,4], [5,4]], dtype="float32")
print(a)
print(a[a[:,1] == 4])
exit()

print(a)
print(np.min(a[:,0]), np.max(a[:,1]))
print(tf.reduce_min(a))
print(a.shape)

b = np.array([[1,-2], [3,4], [5,6]], dtype="float32")
print(b)
for i in range(b.shape[1]):
    b[:, i] = b[:, i] - np.min(b[:, i])
    b[:, i] = b[:, i] / np.max(b[:, i])
b = tf.constant(b, dtype="float32")
print(b)

a[0][0] = a[0][0] - 1.0
print(a)



def func(x,y):
    z = x * y
    return z
from scipy.integrate import simps
import numpy as np
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 30)
z = np.cos(x[:,None])**4 + np.sin(y)**2
print(x)
print(x[:,None])
print(np.sin(y)**2)
print(z)
print(x)
print(simps(z,y))
print(simps(simps(z, y), x))


import math

gev_to_pb = 0.389379e9
alpha = 0.0072973525693
eta_min = -2.5
eta_max = 2.5
e_cms = 200.
z=1/3

print(
    math.pi/3*alpha**2*z**4*e_cms**-2*
    (math.tanh(eta_min)-math.tanh(eta_max)+2*(eta_max-eta_min))*gev_to_pb
)


a = "hallo"
bool = True
print(float(bool))
b = a * False
print(b)

for i in range(0):
    print("hallo?")


a = (1,2)
print(type(a), *a)
if type(a) == tuple:
    print("hello")

a = [1, 2, 3]
b = np.array([1, 2, 3])

a = a[a != 1]
print(a)

b= b[b != 1]
print(b)
c = None

print(c, type(c))
print(str(c), type(str(c)))
"""

a = pd.DataFrame()
a["hallo"] = ["test"]
a["mello"] = ["wello"]
print(a)

a = a.transpose()
print(a)