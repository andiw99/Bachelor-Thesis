import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import Layers
import time

#Daten einlesem
diff_WQ_theta_data_raw = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/diff_WQ_theta_data")
#best_losses_theta = pd.read_csv("best_losses_theta")

#Variablen
total_data = len(diff_WQ_theta_data_raw["Theta"])
print("Messwerte:", total_data)
train_frac = 0.8
batch_size = 64
buffer_size = int(total_data * train_frac)
epochs = 80
units = 64
learning_rate  = 4e-4
nr_hidden_layers = 3
l2_kernel = 0.01
l2_bias = 0.01
#best_total_loss = best_losses_theta["best_total_loss"]
#best_total_loss_squared = best_losses_theta["best_total_loss_squared"]

#Daten vorbereiten
dataset = diff_WQ_theta_data_raw.copy()
#In Training und Testdaten unterteilen
train_dataset = dataset.sample(frac=train_frac, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#aus den Daten einen Tensor machen
train_features = tf.constant([train_dataset["Theta"]], shape=(len(train_dataset["Theta"]), 1), dtype="float32")
test_features = tf.constant([test_dataset["Theta"]], shape=(len(test_dataset["Theta"]), 1), dtype="float32")

train_labels = tf.constant([train_dataset["WQ"]], shape=(len(train_dataset["WQ"]), 1), dtype="float32")
test_labels = tf.constant([test_dataset["WQ"]], shape=(len(test_dataset["WQ"]), 1), dtype="float32")


print("train_features:", train_features)
print("train_labels:", train_labels)
"""
training_data = tf.data.Dataset.from_tensor_slices((training_data_theta, train_dataset["WQ"]))
training_data = training_data.batch(batch_size=batch_size)
"""

#Architektur des Models festlegen, weiterhin activation, units, initialisierung und regularisierung
high_theta_model = keras.Sequential(
    [
        layers.Dense(units=units, activation="relu", name="layer1", kernel_initializer="RandomNormal", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="relu", name="layer2", kernel_initializer="RandomNormal", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="relu", name="layer3", kernel_initializer="RandomNormal", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=1, activation="relu", name="output_layer", kernel_initializer="RandomNormal",kernel_regularizer=None, bias_regularizer=None)
    ]
)
#Model compilen, optimizer und loss festlegen
high_theta_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=Layers.MeanAbsoluteError())
"""
#Debuggin, Summary und gewichte angucken
print("Summary:", high_theta_model.summary())
print("weights", high_theta_model.get_weights())
"""

#hier sollte eigentlich das Training stattfinden
history = high_theta_model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
#History der losses ausgeben, momentan leider konstant
print("history:", history.history)

#Model testen, indem fehler auf die Testdaten berechnet wird.
test_results = high_theta_model.evaluate(test_features, test_labels, batch_size=batch_size, verbose=2)

#Plotte den Loss
plt.plot(history.history["loss"], label="loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.show()

#test results ausgeben
print(test_results)

#plotte Graphen für theta_pred
results = high_theta_model(test_features)
plt.plot(test_features, results)
plt.plot(test_features, test_labels)
plt.ylabel("WQ")
plt.xlabel(r"$\theta$")
plt.show()

plt.plot(test_features, results)
plt.ylabel("WQ")
plt.xlabel(r"$\theta$")
plt.show()

print("Summary nach Training:", high_theta_model.summary())

#percentage error
results = high_theta_model(test_features)
percentage_loss = keras.losses.MeanAbsolutePercentageError()
validation_loss = percentage_loss(y_true=test_labels, y_pred=results)
print("percentage loss:", validation_loss)
