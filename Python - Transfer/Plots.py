import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
from matplotlib import cm

#Pfade eingeben
import ml

paths = dict()

#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["$\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/PlottingData_CT14nnlo/eta_x_1_constant"
paths["x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/PlottingData_CT14nnlo/x_constant"
paths["MMHT $\eta, x_1$ constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/eta_x_1_constant"
paths["MMHT x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/x_constant"
paths["3D-Plot, MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/x_2_constant__3D"
paths["CT14-model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_guess_important_range"
paths["MMHT-model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/transferred_model"
model_keys = {"model", "CT14-model", "MMHT-model"}
label_name = "WQ"

#Daten einlesen
data = dict()
models = dict()
transformers = dict()
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for key,path in paths.items():
    if key not in model_keys:
        (_, features[key], labels[key], _, _, features_pd[key], labels_pd[key], _) = ml.data_handling(data_path=path, label_name="WQ")
    if key in model_keys:
        (models[key], transformers[key]) = ml.load_model_and_transormer(model_path=path)

show_3D_plots = False

#Dictionary mit den Werten anlegen, die jeweils variabel sind im jeweiligen Dataset
variabel = dict()
#Jetzt plotten irgendwie
for dataset in features_pd:
    #jeder dictionary eintrag muss liste sein, falls 3D plot
    variabel[dataset] = []
    #überprüfen, ob es sich um 3d-data handelt
    for key in features_pd[dataset]:
        value = features_pd[dataset][key][0]
        if not all(values == value for values in features_pd[dataset][key]):
            variabel[dataset].append(key)

for dataset in variabel:
    order = np.argsort(features_pd[dataset][variabel[dataset][0]], axis=0)
    for dataset2 in variabel:
        if variabel[dataset] == variabel[dataset2]:
            plot_features = np.array(features_pd[dataset2][variabel[dataset][0]])[order]
            plot_labels = np.array(labels_pd[dataset2])[order]
            plt.plot(plot_features, plot_labels, label=dataset2)

    plt.ylabel(label_name)
    plt.xlabel(str(variabel[dataset][0]))
    if variabel[dataset][0] == "x_1" or variabel[dataset][0] == "x_2":
        plt.yscale("log")
    plt.title(dataset)
    plt.legend()
    plt.show()

print(variabel)
for dataset in features_pd:
    exit()

if plotting_data == 2:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # plot the surface
    surf = ax.plot_trisurf(features_pd[dataset][keys[0]], features_pd[dataset][keys[1]], labels_pd[dataset], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    ax.set_zlabel(label_name)
    ax.set_zscale("log")
    plt.tight_layout()
    ax.view_init(10, 50)
    plt.show()


#Überprüfen, ob das feature konstant ist:
if plotting_data == 1:
    for key in features_pd[dataset]:
        value = features_pd[dataset][key][0]
        if not all(values == value for values in features_pd[dataset][key]):
            #Fkt plotten
            order = np.argsort(features_pd[dataset][key], axis=0)
            plot_features = np.array(features_pd[dataset][key])[order]
            plot_labels = np.array(labels_pd[dataset])[order]
            plt.plot(plot_features, plot_labels, label="analytic")
            plt.ylabel(label_name)
            plt.xlabel(str(key))
            if key == "x_1" or key == "x_2":
                plt.yscale("log")
            plt.title(dataset)
            plt.legend()
            plt.show()