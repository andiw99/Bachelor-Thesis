import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import MC

#Pfade eingeben
model_paths = dict()
model_paths["Prediction"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/theta_model_important_range"
#model_paths["importance sampling"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/theta_model_full_range_IS"
#model_paths["full range"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/RandomSearchTheta/batch_size_128_units_64_nr_layers_3_learning_rate_0.005_mean_absolute_error_RMSprop_leaky_relu_logarithm_True_scaling_bool_True_base10_False_label_normalization_False_dataset_TrainingData60k_ep_0.01_"
#model_path= "/Files/Hadronic/Models/best_guess_4M"
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths = dict()
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/PlottingData5k_ep_0.163/all"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "2"
input("namen geändert?")
save_path = save_path + name
label_name = "WQ"
trans_to_pb = True

#ticks festlegen ggf
pi_ticks = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
pi_names = np.array(["0", r"$\frac{1}{4} \pi$", r"$\frac{1}{2} \pi$", r"$\frac{3}{4} \pi$", r"$\pi$"])

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
                  keys=keys, save_path=save_path, trans_to_pb=trans_to_pb, xtick_labels=pi_names, xticks=pi_ticks)

