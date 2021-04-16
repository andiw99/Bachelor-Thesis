import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import ast
from matplotlib import cm

#Pfade eingeben
paths = dict()
model_path= "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/SourceModel9"

#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["model"] = model_path
paths["$eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/TestData/eta_x_1_constant"
paths["x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/TestData/x_constant"
label_name = "WQ"

#Daten einlesen
data = dict()
for key,path in paths.items():
    if key != "model":
        data[key] = pd.read_csv(path)

# Modell laden
model = keras.models.load_model(filepath=model_path)

show_3D_plots = False
config = pd.read_csv(model_path + "/config", index_col="property")
print(config)
config = config.transpose()
print(config)
transformer_config = ast.literal_eval(config["transformer_config"][0])
transformer = ml.label_transformation(config=transformer_config)
loss_function = keras.losses.MeanAbsoluteError()

#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset in data:
    features_pd[dataset] = data[dataset]
    labels_pd[dataset] = features_pd[dataset].pop(label_name)
    #Aus den Pandas Dataframes tf-Tensoren machen
    for i,feature in enumerate(features_pd[dataset]):
        if i == 0:
            features[dataset] = tf.constant([features_pd[dataset][feature]], dtype="float32")
        else:
            more_features = tf.constant([features_pd[dataset][feature]], dtype="float32")
            features[dataset] = tf.experimental.numpy.append(features[dataset], more_features, axis=0)
    #transponieren
    features[dataset] = tf.transpose(features[dataset])
    labels[dataset] = tf.transpose(tf.constant([labels_pd[dataset]], dtype="float32"))


# Für jedes Dataset predictions und losses berechnen
# predictions:
predictions = dict()
for dataset in features:
    predictions[dataset] = transformer.retransform(model(features[dataset]))

#losses
losses = dict()

for dataset in predictions:
    losses[dataset] = []
    for i,label in enumerate(predictions[dataset]):
        losses[dataset].append(float(loss_function(y_true=labels[dataset][i], y_pred=label)))

#Jetzt plotten irgendwie
for dataset in predictions:
    #überprüfen, ob es sich um 3d-data handelt
    plotting_data = 0
    keys = []
    for key in data[dataset]:
        value = data[dataset][key][0]
        if not all(values == value for values in data[dataset][key]):
            plotting_data += 1
            keys.append(key)

    if plotting_data == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot the surface
        print("data[dataset][keys[0]]:",data[dataset][keys[0]])
        print("data[dataset][keys[1]]",data[dataset][keys[1]])
        print("labels[dataset]", labels[dataset])
        plot_labels = labels[dataset][:,0]
        print(plot_labels)
        surf = ax.plot_trisurf(data[dataset][keys[0]], data[dataset][keys[1]], plot_labels, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel("rewait")
        ax.set_zscale("linear")
        plt.tight_layout()
        ax.view_init(10, 50)
        plt.show()

        #losses plotten
        plot_losses = losses[dataset]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot the surface
        surf = ax.plot_trisurf(data[dataset][keys[0]], data[dataset][keys[1]], losses[dataset], cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel("reweight")
        ax.set_zscale("linear")
        plt.tight_layout()
        ax.view_init(10, 50)
        plt.show()

    #Überprüfen, ob das feature konstant ist:
    if plotting_data == 1:
        for key in data[dataset]:
            value = data[dataset][key][0]
            if not all(values == value for values in data[dataset][key]):
                #Fkt plotten
                order = np.argsort(data[dataset][key], axis=0)
                plot_features = np.array(data[dataset][key])[order]
                plot_predictions = np.array(predictions[dataset])[order]
                plot_labels = np.array(labels[dataset])[order]
                plt.plot(plot_features, plot_predictions, label="ML")
                plt.plot(plot_features, plot_labels, label="analytic")
                if key == "x_1" or key == "x_2":
                    plt.yscale("log")
                plt.ylabel(label_name)
                plt.xlabel(str(key))
                plt.title("r" + str(dataset))
                plt.legend()
                plt.show()

                #losses plotten
                plot_losses = np.array(losses[dataset])[order]
                plt.plot(plot_features, plot_losses)
                plt.ylabel("Loss")
                plt.xlabel(str(key))
                plt.yscale("Log")
                plt.title(dataset)
                plt.show()

