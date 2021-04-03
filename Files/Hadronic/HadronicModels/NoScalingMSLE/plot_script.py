import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt 
import Layers
#Modell laden 
hadronic_model = keras.models.load_model(filepath="NoScalingMSLE/")
logarithm=False
scaling=1.0
loss_function=keras.losses.MeanAbsoluteError()
loss_fn=Layers.MeanSquaredLogarithmicError()
#Daten einlesen
hadronic_data_x_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_x12/logarithmic_hadronic_data_no_negative__x_2_constant__0.11")
hadronic_data_eta_x_2_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_x12/logarithmic_hadronic_data_no_negative__eta_x_2_constant__0.45")
hadronic_data_eta_x_1_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_x12/logarithmic_hadronic_data_no_negative__eta_x_1_constant__0.45")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
if logarithm:
    predictions_x_constant = ((1 / scaling) * np.exp(hadronic_model(pred_feature_x_constant)))
else:
    predictions_x_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_x_constant)))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["x_2"], hadronic_data_eta_x_2_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
if logarithm:
    predictions_eta_x_2_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))
else:
    predictions_eta_x_2_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_1_constant["x_1"], hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
if logarithm:
    predictions_eta_x_1_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))
else:
    predictions_eta_x_1_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
train_losses_eta_constant=[]
train_losses_x_constant=[]
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    losses_x_constant.append(float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_x_constant["WQ"][i])))
    train_losses_x_constant.append(float(loss_fn((1/scaling)*hadronic_model(feature),hadronic_data_x_constant["WQ"][i])))

for i, feature in enumerate(pred_feature_eta_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_eta_x_2_constant["WQ"][i]))
    train_losses_eta_constant.append(float(loss_fn((1/scaling)*hadronic_model(feature),hadronic_data_eta_x_2_constant["WQ"][i])))
    losses_eta_constant.append(error)


#Plot mit konstantem x_1, x_2 und losses im subplot
fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
ax0.plot(hadronic_data_x_constant["eta"], predictions_x_constant)
ax0.set(xlabel=r"$\eta$", ylabel="WQ")
ax0.set_title("x_1, x_2 constant")
ax1.plot(hadronic_data_x_constant["eta"], train_losses_x_constant)
ax1.set(xlabel=r"$\eta$", ylabel="Loss")
plt.tight_layout()
plt.show()

#Losses plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], losses_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta,x_2
plt.plot(hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["WQ"])
plt.plot(hadronic_data_eta_x_2_constant["x_1"], predictions_eta_x_2_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta, x_1
plt.plot(hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["WQ"])
plt.plot(hadronic_data_eta_x_1_constant["x_2"], predictions_eta_x_1_constant)
plt.xlabel(r"$x_2$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
plt.plot(hadronic_data_eta_x_2_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.show()