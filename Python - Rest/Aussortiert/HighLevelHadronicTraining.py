import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import time

#Daten einlesem
hadronic_WQ_data_raw = pd.read_csv("HadronicData/some_hadronic_data")

#Variablen
units = 64
train_frac = 0.90
learning_rate = 1
batch_size = 64
epochs = 5
loss_function = keras.losses.MeanSquaredError()

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
hadronic_model = keras.Sequential(
    [
        layers.Dense(units=units, activation="relu", name="layer1", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="relu", name="layer2", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=units, activation="relu", name="layer3", kernel_regularizer=None, bias_regularizer=None),
        layers.Dense(units=1, activation="linear", name="output_layer", kernel_regularizer=None, bias_regularizer=None)
    ]
)
#Model compilen, optimizer und loss festlegen
hadronic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.MeanAbsoluteError())
"""
#Model bauen, eigentlich nur damit summary aufgerufen werden kann
high_hadronic_model.build(input_shape=(1, 3))

#Debuggin, Summary und gewichte angucken
print("Summary:", high_hadronic_model.summary())
print("weights", high_hadronic_model.get_weights())
"""
#hier sollte eigentlich das Training stattfinden
history =hadronic_model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
#History der losses ausgeben, momentan leider konstant
print("history:", history.history)

#Model testen, indem fehler auf die Testdaten berechnet wird.
test_results = hadronic_model.evaluate(test_features, test_labels, batch_size=batch_size, verbose=2)

#Plotte den Loss
plt.plot(history.history["loss"], label="loss")
plt.legend()
plt.show()

#test results ausgeben
print(test_results)

#Daten einlesen
hadronic_data_x_constant = pd.read_csv("HadronicData/some_hadronic_data__x_constant__0.07083333333333333")
hadronic_data_eta_constant = pd.read_csv("HadronicData/some_hadronic_data__eta_constant__1.5226130653266328")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
predictions_x_constant = (hadronic_model(pred_feature_x_constant))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["x_2"], hadronic_data_eta_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
predictions_eta_constant = (hadronic_model(pred_feature_eta_constant))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    losses_x_constant.append(float(loss_function((hadronic_model(feature)), hadronic_data_x_constant["WQ"][i])))

for i, feature in enumerate(pred_feature_eta_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function((hadronic_model(feature)), hadronic_data_eta_constant["WQ"][i]))
    losses_eta_constant.append(error)
    if (i+1) % 10 == 0:
        print("feature:", feature)
        print("error:", error)


#Plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
plt.plot(hadronic_data_x_constant["eta"], predictions_x_constant)
plt.tight_layout()
plt.xlabel(r"$\eta$")
plt.ylabel("WQ")
plt.show()

#Losses plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], losses_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta,x_2
plt.plot(hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["WQ"])
plt.plot(hadronic_data_eta_constant["x_1"], predictions_eta_constant)
plt.ylim(0, 0.08)
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
plt.plot(hadronic_data_eta_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.show()
