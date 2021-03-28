import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import Layers
import time

#Daten einlesem
hadronic_WQ_data_raw = pd.read_csv("less_hadronic_WQ_data")

#Variablen
units = 64
train_frac = 0.90
learning_rate = 1e-2
batch_size = 128
epochs = 10

#Daten vorbereiten
train_dataset = hadronic_WQ_data_raw.sample(frac=train_frac, random_state=0)
test_dataset = hadronic_WQ_data_raw.drop(train_dataset.index)

#In Features und Labels unterteilen
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("WQ")
test_labels = test_features.pop("WQ")

#Schauen wir uns mal die Daten an
print("train_features:", type(train_features))
print("train_labels:", type(train_labels))

#Aus den Pandas Dataframes tf-Tensoren machen, ist das richtig?
train_features = tf.constant([train_features["x_1"], train_features["x_2"], train_features["eta"]],  dtype="float32")
test_features = tf.constant([test_features["x_1"], test_features["x_2"], test_features["eta"]],  dtype="float32")

train_labels = tf.transpose(tf.constant([train_labels.to_numpy(dtype="float32")]))
test_labels = tf.transpose(tf.constant([test_labels.to_numpy(dtype="float32")]))

#Dimensionen arrangieren, sicher? lassen wir erstmal weg
train_features = tf.transpose(train_features)
test_features = tf.transpose(test_features)

print("train_features:", train_features)
print("train_labels:", train_labels)

#Architektur des Models festlegen, weiterhin activation, units, initialisierung und regularisierung
high_hadronic_model = keras.Sequential(
    [
        layers.Dense(units=units, activation="tanh", name="layer1", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="tanh", name="layer2", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="tanh", name="layer3", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=1, activation="tanh", name="output_layer", kernel_regularizer=None, bias_regularizer=None)
    ]
)
#Model compilen, optimizer und loss festlegen
high_hadronic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_absolute_error")
#Model bauen, eigentlich nur damit summary aufgerufen werden kann
high_hadronic_model.build(input_shape=(1, 3))

#Debuggin, Summary und gewichte angucken
print("Summary:", high_hadronic_model.summary())
print("weights", high_hadronic_model.get_weights())

#hier sollte eigentlich das Training stattfinden
history = high_hadronic_model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
#History der losses ausgeben, momentan leider konstant
print("history:", history.history)

#Model testen, indem fehler auf die Testdaten berechnet wird.
test_results = high_hadronic_model.evaluate(test_features, test_labels, batch_size=batch_size, verbose=2)

#Plotte den Loss
plt.plot(history.history["loss"], label="loss")
plt.legend()
plt.show()

#test results ausgeben
print(test_results)

#plotte Graphen für eta, für x_1=x_2=0.2
x_1 = 0.2
x_2 = x_1
WQ_list = []
eta_list = []
step = 0
for i in range(-200, 200):
    eta = 3*i/200
    x = tf.constant([[x_1, x_2, eta]])
    step += 1
    if step % 50 == 0:
        print("Prediction für", eta, "ist", float(high_hadronic_model(x)))
    WQ_list.append(float(high_hadronic_model(x)))
    eta_list.append(eta)

plt.plot(eta_list, WQ_list)
plt.xlabel(r"$\eta$")
plt.ylabel("WQ")
plt.show()

print("Summary nach Training:", high_hadronic_model.summary())
