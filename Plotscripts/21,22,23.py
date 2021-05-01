import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
from matplotlib import cm
import ml
from tensorflow import keras

#Pfade eingeben
paths = dict()
analytic_paths = dict()
model_paths = dict()
#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["CT14 x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14 PlottingData/x_constant"
paths["CT14 eta,x_1 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14 PlottingData/eta_x_1_constant"
paths["CT14 eta,x_2 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14 PlottingData/eta_x_2_constant"
analytic_paths["MMHT x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/x_constant"
analytic_paths["MMHT eta, x_1 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/eta_x_1_constant"
analytic_paths["MMHT eta, x_2 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT PlottingData/eta_x_2_constant"
model_paths["CT14nnlo reweighted"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/RandomSearch_logartihm_false/best_model"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "reweight_lower_x"
label_name = "WQ"

#modelle laden
models = dict()
transformers = dict()
for model_name in model_paths:
    models[model_name], transformers[model_name] = ml.load_model_and_transormer(model_path=model_paths[model_name])

#Daten einlesen
data = dict()
for key,path in paths.items():
    if key != "model":
        data[key] = pd.read_csv(path)

show_3D_plots = False

yticks = [0.9994, 0.9996, 0.9998, 1.0000, 1.0002, 1.0004, 1.0006]
ytick_labels = ["1-6e-4","1-4e-4", "1-2e-4", "1", "1+2e-4", "1+4e-4", "1+6e-4"]


#Features und labels unterteilen
analytic_prep_data = dict()
source_prep_data = dict()
for key,path in analytic_paths.items():
        analytic_prep_data[key] = ml.data_handling(data_path=path, label_name=label_name, return_pd=True,
                                                   label_cutoff=False, return_as_tensor=False, dtype="float32")
for key,path in paths.items():
        source_prep_data[key] = ml.data_handling(data_path=path, label_name=label_name, return_pd=True,
                                                 label_cutoff=False, return_as_tensor=False, dtype="float32")


analytic_features = dict()
analytic_labels = dict()
analytic_features_pd = dict()
analytic_labels_pd = dict()
for dataset in analytic_prep_data:
    analytic_features[dataset] = analytic_prep_data[dataset][1]
    analytic_labels[dataset] = analytic_prep_data[dataset][2]
    analytic_features_pd[dataset] = analytic_prep_data[dataset][5]
    analytic_labels_pd[dataset] = analytic_prep_data[dataset][6]

source_features = dict()
source_labels = dict()
source_features_pd = dict()
source_labels_pd = dict()
for dataset in source_prep_data:
    source_features[dataset] = source_prep_data[dataset][1]
    source_labels[dataset] = source_prep_data[dataset][2]
    source_features_pd[dataset] = source_prep_data[dataset][5]
    source_labels_pd[dataset] = source_prep_data[dataset][6]



#Reweightning
predictions = dict()
for i,source_dataset in enumerate(source_features.keys()):
    predictions[i] = dict()
    predictions[i]["CT14nnlo"] = source_labels[source_dataset]
    for model_name in models:
        predictions[i][model_name] = 1/transformers[model_name].retransform(
            models[model_name].predict(source_features[source_dataset][:,:2])) * \
            source_labels[source_dataset]
# Jeweilige predictions mit den analytischen werten verbinden


for i,analytic_dataset in enumerate(analytic_features.keys()):
    predictions[analytic_dataset] = predictions.pop(i)

print("predictions", predictions)
print("analytic_labels", analytic_labels)

#Dictionary mit den Werten anlegen, die jeweils variabel sind im jeweiligen Dataset
variabel = dict()
#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=analytic_features_pd[dataset])
    ml.make_reweight_plot(features_pd=analytic_features_pd[dataset], labels=analytic_labels[dataset],
                  predictions=predictions[dataset], keys=keys, save_path=save_path + name, trans_to_pb=True,
                  autoscale_ratio=True, yticks_ratio=yticks, ytick_labels_ratio=ytick_labels)
