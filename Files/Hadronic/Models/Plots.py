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
#model_path= "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/data_amount_comparison/dataset_MuchTrainingDataMidRange_"
model_path= "/Files/Hadronic/Models/Without Cut, wrong e-value/transferred_model_mid"
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["model"] = model_path
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT2014/eta_x_1_constant"
paths["x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT2014/x_constant"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/Meeting/"
name = "mid_transfer_data"
input("namen geändert?")
save_path = save_path + name
label_name = "WQ"

#Daten einlesen
# Modell und transformer laden
(model, transformer) = ml.load_model_and_transormer(model_path=model_path)

show_3D_plots = False
loss_function = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.NONE)

#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset, path in paths.items():
    if dataset != "model":
        (_, features[dataset], labels[dataset], _, _, features_pd[dataset], labels_pd[dataset], _) =\
            ml.data_handling(data_path=path, label_name=label_name, return_pd=True)


# Für jedes Dataset predictions und losses berechnen
# predictions, losses:
predictions = dict()
losses = dict()
for dataset in features:
    predictions[dataset] = transformer.retransform(model(transformer.rescale(features[dataset])))
    losses[dataset] = loss_function(y_true=labels[dataset], y_pred=predictions[dataset])


#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                 losses=losses[dataset], keys=keys, title=dataset, label_name=label_name, save_path=save_path)

