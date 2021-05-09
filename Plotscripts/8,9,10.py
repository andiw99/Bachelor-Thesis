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
model_paths["Predictions"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_guess_4M"
#model_path= "/Files/Hadronic/Models/best_guess_4M"
#more data to plot?
#PlottingDataLowX_data = ...

#Pfade in dict speichern
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/PlottingDataHighX/eta_x_1_constant"
paths["$x_1, x_2$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/PlottingDataHighX/x_constant"
paths["$\eta, x_2$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/PlottingDataHighX/eta_x_2_constant"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "8,9,10"
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
use_cut = True
replace_cut = True
loss_function = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.NONE)
# y versatz 0.03
text_loc = (0.47, 0.87)

#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset, path in paths.items():
    if dataset != "model":
        (_, features[dataset], labels[dataset], _, _, features_pd[dataset], labels_pd[dataset], _) =\
            ml.data_handling(data_path=path, label_name=label_name, return_pd=True, label_cutoff=False, return_as_tensor=False)


# Für jedes Dataset predictions und losses berechnen
# predictions, losses:
predictions = dict()
losses = dict()
for dataset in features:
    if use_cut:
        if replace_cut:
            _, cut = MC.cut(features=features[dataset],
                                            return_cut=True)
            labels[dataset][~cut] = np.nan
            print("labels", labels[dataset])
            labels_pd[dataset][~cut] = np.nan
        else:
            features[dataset], cut = MC.cut(features=features[dataset], return_cut=True)
            features_pd[dataset] = features_pd[dataset][cut].reset_index(drop=True)
            labels[dataset] = labels[dataset][cut]
            labels_pd[dataset] = labels_pd[dataset][cut].reset_index(drop=True)
    predictions[dataset] = dict()
    losses[dataset] = dict()
    for model_name in models:
        predictions[dataset][model_name] = transformers[model_name].retransform(
            models[model_name].predict(transformers[model_name].rescale(features[dataset])))
        if replace_cut:
            predictions[dataset][model_name][~cut] = np.nan
        print("predictions", predictions[dataset][model_name])


#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                  keys=keys, save_path=save_path, trans_to_pb=True, text_loc=text_loc)

