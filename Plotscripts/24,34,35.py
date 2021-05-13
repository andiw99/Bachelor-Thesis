import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import MC

#Pfade eingeben
paths = dict()
model_paths = dict()
model_paths["transferred model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/test"
model_paths["source model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_model"
#model_path= "/Files/Hadronic/Models/best_guess_4M"
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/eta_x_1_constant"
paths["$x_1, x_2$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/x_constant"
paths["$\eta, x_2$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/eta_x_2_constant"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "24,34,35.1"
input("namen geändert?")
save_path = save_path + name
label_name = "WQ"

#Daten einlesen
# Modell und transformer laden
models = dict()
transformers = dict()
for model_name in model_paths:
    (models[model_name], transformers[model_name]) = ml.load_model_and_transformer(model_path=model_paths[model_name])

show_3D_plots = False
use_cut = False
replace_with_nan = True
text_loc = (0.4, 0.88)
#x_interval = (0.05, 0.15)
x_interval = (0.15, 0.31)
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
    ml.make_reweight_plot(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset], use_sci_fct=True,
                  keys=keys, save_path=save_path, x_cut=True, replace_with_nan=True, lower_x_cut=x_interval[0], upper_x_cut=x_interval[1])

