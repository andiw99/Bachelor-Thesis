import tensorflow as tf
from tensorflow import keras
import pandas as pd

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