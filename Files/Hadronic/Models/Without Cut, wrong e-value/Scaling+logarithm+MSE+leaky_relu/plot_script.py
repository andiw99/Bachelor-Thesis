import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt 
import ml
from matplotlib import cm

#Modell laden 
hadronic_model = keras.models.load_model(filepath="model")
logarithm=True
scaling=1.0
show_3D_plots = True
config ={'scaling': True, 'logarithm': True, 'shift': False, 'label_normalization': False, 'scale': 10903854514176.0}
transformer = ml.label_transformation(config=config)
loss_function=keras.losses.mean_absolute_error
loss_fn=loss_function
#Daten einlesen
hadronic_data_x_constant = pd.read_csv("/Files/Hadronic/Data/log_neg_x12/x_constant")
hadronic_data_eta_x_2_constant = pd.read_csv("/Files/Hadronic/Data/log_neg_x12/eta_x_2_constant")
hadronic_data_eta_x_1_constant = pd.read_csv("/Files/Hadronic/Data/log_neg_x12/eta_x_1_constant")
hadronic_data_x_2_constant = pd.read_csv("/Files/Hadronic/Data/log_neg_3D/x_2_constant__3D")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
predictions_x_constant = transformer.retransform(hadronic_model(pred_feature_x_constant))


pred_feature_eta_x_2_constant = tf.constant([hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["x_2"], hadronic_data_eta_x_2_constant["eta"]], dtype="float32")
pred_feature_eta_x_2_constant = tf.transpose(pred_feature_eta_x_2_constant)
predictions_eta_x_2_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_2_constant))

pred_feature_eta_x_1_constant = tf.constant([hadronic_data_eta_x_1_constant["x_1"], hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["eta"]], dtype="float32")
pred_feature_eta_x_1_constant = tf.transpose(pred_feature_eta_x_1_constant)
predictions_eta_x_1_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_1_constant))

pred_feature_x_2_constant = tf.constant([hadronic_data_x_2_constant["x_1"], hadronic_data_x_2_constant["x_2"], hadronic_data_x_2_constant["eta"]], dtype="float32")
pred_feature_x_2_constant = tf.transpose(pred_feature_x_2_constant)
predictions_x_2_constant = transformer.retransform(hadronic_model(pred_feature_x_2_constant))[:,0]
print(len(list(hadronic_data_x_2_constant["x_1"])))
print(len(list(hadronic_data_x_2_constant["eta"])))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
losses_x_2_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(transformer.retransform(hadronic_model(feature)), hadronic_data_x_constant["WQ"][i]))
    losses_x_constant.append(error)
    
for i, feature in enumerate(pred_feature_eta_x_2_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(transformer.retransform(hadronic_model(feature)),
                                    hadronic_data_eta_x_2_constant["WQ"][i]))
    losses_eta_constant.append(error)

print("dauert..")
if show_3D_plots:
    for i, feature in enumerate(pred_feature_x_2_constant):
        feature = tf.reshape(feature, shape=(1,3))
        error = float(loss_function(transformer.retransform(hadronic_model(feature)), hadronic_data_x_2_constant["WQ"][i]))
        losses_x_2_constant.append(error)
    print(len(losses_x_2_constant))
#Plot mit konstantem x_1, x_2 und losses im subplot
fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"], label="analytic")
ax0.plot(hadronic_data_x_constant["eta"], predictions_x_constant, label="ML")
ax0.set(xlabel=r"$\eta$", ylabel="WQ")
ax0.set_title(r"$x_1, x_2$ constant")
ax0.legend()
ax1.plot(hadronic_data_x_constant["eta"], losses_x_constant)
ax1.set(xlabel=r"$\eta$", ylabel="Loss")
plt.tight_layout()
plt.show()

#Losses plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], losses_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta,x_2 und Losses
fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["WQ"], label="analytic")
ax0.plot(hadronic_data_eta_x_2_constant["x_1"], predictions_eta_x_2_constant, label="ML")
ax0.set(xlabel = (r"$x_1$"), ylabel = "WQ", yscale="log")
ax0.set_title(r"$x_2, \eta$ constant")
ax0.legend()
ax1.plot(hadronic_data_eta_x_2_constant["x_1"], losses_eta_constant)
ax1.set(xlabel = r"$x_1$", ylabel="Loss")
ax1.set(yscale = "log")
plt.tight_layout()
plt.show()


#Plot mit konstantem eta, x_1
plt.plot(hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["WQ"], label="analytic")
plt.plot(hadronic_data_eta_x_1_constant["x_2"], predictions_eta_x_1_constant, label="ML")
plt.xlabel(r"$x_2$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
#3D Plot mit konstantem x_2
if show_3D_plots:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(hadronic_data_x_2_constant["x_1"], hadronic_data_x_2_constant["eta"], predictions_x_2_constant, cmap=cm.coolwarm)
    ax.set_xlabel("x_1")
    ax.set_ylabel("eta")
    ax.set_zlabel("WQ")
    ax.set_zscale("log")
    plt.tight_layout()
    plt.show()

#3D losses Plot mit konstantem x_2
if show_3D_plots:
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_trisurf(list(hadronic_data_x_2_constant["x_1"]), list(hadronic_data_x_2_constant["eta"]), losses_x_2_constant, cmap=cm.coolwarm)
    ax.set_xlabel("x_1")
    ax.set_ylabel("eta")
    ax.set_zlabel("Losses")
    ax.set_zscale("log")
    plt.tight_layout()
    plt.show()

