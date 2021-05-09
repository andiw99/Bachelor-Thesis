import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import MC

#Pfade eingeben
model_paths = dict()
model_paths["Imortant Range"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/theta_model_important_range"
model_paths["Full Range"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/theta_model_full_range"
model_paths["Full Range + IS"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/theta_model_full_range_IS"
#model_path= "/Files/Hadronic/Models/best_guess_4M"
#more data to plot?
#plotting_data = ...

#ticks festlegen ggf
pi_ticks = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
pi_names = np.array(["0", r"$\frac{1}{4} \pi$", r"$\frac{1}{2} \pi$", r"$\frac{3}{4} \pi$", r"$\pi$"])

#Pfade in dict speichern
paths = dict()
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/PlottingData250_ep_0.01_IS/all"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "theta_model_comparison"
input("namen geändert?")
save_path = save_path + name
label_name = "WQ"
trans_to_pb = True

#Daten einlesen
# Modell und transformer laden
models = dict()
transformers = dict()
for model_name in model_paths:
    (models[model_name], transformers[model_name]) = ml.load_model_and_transformer(model_path=model_paths[model_name])
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
    predictions[dataset] = dict()
    losses[dataset] = dict()
    for model_name in models:
        predictions[dataset][model_name] = transformers[model_name].retransform(
            models[model_name].predict(transformers[model_name].rescale(features[dataset])))
        losses[dataset][model_name] = loss_function(y_true=labels[dataset], y_pred=predictions[dataset][model_name])


#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                  keys=keys, save_path=save_path, trans_to_pb=trans_to_pb, automatic_legend=True, xticks=pi_ticks,
                  xtick_labels=pi_names, autoscale=True, ratio_size=80)

