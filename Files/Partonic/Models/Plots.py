import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import MC

#Pfade eingeben
paths = dict()
model_path= "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/RandomSearchTheta/batch_size_128_units_64_nr_layers_4_learning_rate_0.005_mean_absolute_error_RMSprop_leaky_relu_logarithm_True_scaling_bool_True_base10_True_label_normalization_True_dataset_TrainingData60k_ep_0.01_"
#model_path= "/Files/Hadronic/Models/best_guess_4M"
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["model"] = model_path
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TrainingData60k_ep_0.01/all"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/"
name = "partonic_theta_best_001_mdoel_grid_search"
input("namen geändert?")
save_path = save_path + name
label_name = "WQ"

#Daten einlesen
# Modell und transformer laden
(model, transformer) = ml.load_model_and_transormer(model_path=model_path)

show_3D_plots = False
use_cut = False
loss_function = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.NONE)

#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset, path in paths.items():
    if dataset != "model":
        (_, features[dataset], labels[dataset], _, _, features_pd[dataset], labels_pd[dataset], _) =\
            ml.data_handling(data_path=path, label_name=label_name, return_pd=True, label_cutoff=False)


# Für jedes Dataset predictions und losses berechnen
# predictions, losses:
predictions = dict()
losses = dict()
for dataset in features:
    if use_cut:
        features[dataset], cut = MC.cut(features=features[dataset], return_cut=True)
        features_pd[dataset] = features_pd[dataset][cut].reset_index(drop=True)
        labels[dataset] = labels[dataset][cut]
        labels_pd[dataset] = labels_pd[dataset][cut].reset_index(drop=True)
    predictions[dataset] = transformer.retransform(model.predict(transformer.rescale(features[dataset])))
    losses[dataset] = loss_function(y_true=labels[dataset], y_pred=predictions[dataset])


#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                 losses=losses[dataset], keys=keys, title=dataset, label_name=label_name, save_path=save_path)

