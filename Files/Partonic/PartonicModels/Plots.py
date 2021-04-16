import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import ast

#Pfade eingeben
paths = dict()
model_path="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/PartonicEta/best_model"
training_data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/diff_WQ_eta_data"
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["model"] = model_path
paths["training_data"] = training_data_path

#Daten einlesen
data = dict()
for key,path in paths.items():
    if key != "model":
        data[key] = pd.read_csv(path)

# Modell laden
model = keras.models.load_model(filepath=model_path)

show_3D_plots = False
config = pd.read_csv(model_path + "/config")
config = config.transpose()
transformer_config = ast.literal_eval(config[8][1])
transformer = ml.label_transformation(config=transformer_config)
loss_function = keras.losses.MeanAbsoluteError()

#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset in data:
    features_pd[dataset] = data[dataset]
    labels_pd[dataset] = features_pd[dataset].pop("WQ")
    print(features_pd)
    #Aus den Pandas Dataframes tf-Tensoren machen
    for i,feature in enumerate(features_pd[dataset]):
        print(i)
        print(feature)
        if i == 0:
            features[dataset] = tf.constant([features_pd[dataset][feature]], dtype="float32")
        else:
            more_features = tf.constant([features_pd[[dataset][feature]]], dtype="float32")
            features[dataset] = tf.experimental.numpy.append(features[dataset], more_features, axis=0)
    #transponieren
    features[dataset] = tf.transpose(features[dataset])
    labels[dataset] = tf.transpose(tf.constant([labels_pd[dataset]], dtype="float32"))

print(features)
print(labels)

# Für jedes Dataset predictions und losses berechnen
# predictions:
predictions = dict()
for dataset in features:
    print(features[dataset])
    print(model(features[dataset]))
    predictions[dataset] = transformer.retransform(model(features[dataset]))
#losses
losses = dict()

for dataset in predictions:
    losses[dataset] = []
    for i,label in enumerate(predictions[dataset]):
        losses[dataset].append(float(loss_function(y_true=labels[dataset][i], y_pred=label)))

#Jetzt plotten irgendwie
for dataset in predictions:
    print(features[dataset])

    #Fkt plotten
    order = np.argsort(features[dataset], axis=0)
    plot_features = features[dataset].numpy()[order][:,0,0]
    plot_predictions = np.array(predictions[dataset])[order][:,0,0]
    plot_labels = np.array(labels[dataset])[order][:,0,0]
    plt.plot(plot_features, plot_predictions, label="ML")
    plt.plot(plot_features, plot_labels, label="analytic")
    plt.ylabel("WQ")
    plt.xlabel("Theta")
    plt.title(dataset)
    plt.legend()
    plt.show()

    #losses plotten
    plot_losses = np.array(losses[dataset])[order][:,0]
    plt.plot(plot_features, plot_losses)
    plt.ylabel("Loss")
    plt.yscale("Log")
    plt.title(dataset)
    plt.show()



exit()
pred_feature = tf.constant(
    [data_x_constant["x_1"], data_x_constant["x_2"], data_x_constant["eta"]],
    dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
predictions_x_constant = transformer.retransform(model(pred_feature_x_constant))

pred_feature_eta_x_2_constant = tf.constant(
    [data_eta_x_2_constant["x_1"], data_eta_x_2_constant["x_2"],
     data_eta_x_2_constant["eta"]], dtype="float32")
pred_feature_eta_x_2_constant = tf.transpose(pred_feature_eta_x_2_constant)
predictions_eta_x_2_constant = transformer.retransform(model(pred_feature_eta_x_2_constant))

pred_feature_eta_x_1_constant = tf.constant(
    [data_eta_x_1_constant["x_1"], data_eta_x_1_constant["x_2"],
     data_eta_x_1_constant["eta"]], dtype="float32")
pred_feature_eta_x_1_constant = tf.transpose(pred_feature_eta_x_1_constant)
predictions_eta_x_1_constant = transformer.retransform(model(pred_feature_eta_x_1_constant))

pred_feature_x_2_constant = tf.constant(
    [data_x_2_constant["x_1"], data_x_2_constant["x_2"], data_x_2_constant["eta"]],
    dtype="float32")
pred_feature_x_2_constant = tf.transpose(pred_feature_x_2_constant)
predictions_x_2_constant = transformer.retransform(model(pred_feature_x_2_constant))

# Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
losses_x_2_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function((tf.math.abs((1 / scaling) * tf.math.exp(model(feature)))),
                                data_x_constant["WQ"][i]))
    losses_x_constant.append(error)

for i, feature in enumerate(pred_feature_eta_x_2_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(transformer.retransform(model(feature)),
                                data_eta_x_2_constant["WQ"][i]))
    losses_eta_constant.append(error)

print("dauert..")
if show_3D_plots:
    for i, feature in enumerate(pred_feature_x_2_constant):
        feature = tf.reshape(feature, shape=(1, 3))
        error = float(
            loss_function(transformer.retransform(model(feature)), data_x_2_constant["WQ"][i]))
        losses_x_2_constant.append(error)
# Plot mit konstantem x_1, x_2 und losses im subplot
fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(data_x_constant["eta"], data_x_constant["WQ"])
ax0.plot(data_x_constant["eta"], predictions_x_constant)
ax0.set(xlabel=r"$\eta$", ylabel="WQ")
ax0.set_title("x_1, x_2 constant")
ax1.plot(data_x_constant["eta"], losses_x_constant)
ax1.set(xlabel=r"$\eta$", ylabel="Loss")
plt.tight_layout()
plt.show()

# Losses plot mit konstantem x_1, x_2
plt.plot(data_x_constant["eta"], losses_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# Plot mit konstantem eta,x_2
plt.plot(data_eta_x_2_constant["x_1"], data_eta_x_2_constant["WQ"])
plt.plot(data_eta_x_2_constant["x_1"], predictions_eta_x_2_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

# Plot mit konstantem eta, x_1
plt.plot(data_eta_x_1_constant["x_2"], data_eta_x_1_constant["WQ"])
plt.plot(data_eta_x_1_constant["x_2"], predictions_eta_x_1_constant)
plt.xlabel(r"$x_2$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

# Losses plotten mit konstantem eta, x_2
# print("data_eta_constant[x_1]:", data_eta_constant["x_1"])
# print("losses_eta_constant",losses_eta_constant)
plt.plot(data_eta_x_2_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.show()