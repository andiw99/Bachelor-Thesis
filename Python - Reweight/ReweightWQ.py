import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
from matplotlib import cm
import ml
from tensorflow import keras

#Pfade eingeben
paths = dict()

#more data to plot?
#plotting_data = ...

#Pfade in dict speichern
paths["CT14 x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/CT14nnlo/x_constant"
paths["CT14 eta,x_1 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/CT14nnlo/eta_x_1_constant"
paths["MMHT eta, x_1 constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/eta_x_1_constant"
paths["MMHT x constant"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/x_constant"
paths["3D-Plot, MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Reweight_of_diff_WQ/MMHT2014/x_2_constant__3D"
paths["model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/best_model"
label_name = "WQ"

#model laden
model = keras.models.load_model(filepath=paths["model"])
config = pd.read_csv(paths["model"] + "/config")
config = config.transpose()
transformer_config = ast.literal_eval(config["transformer_config"][0])
transformer = ml.LabelTransformation(config=transformer_config)


#Daten einlesen
data = dict()
for key,path in paths.items():
    if key != "model":
        data[key] = pd.read_csv(path)

show_3D_plots = False

#Features und labels unterteilen
prep_data = dict()
for key,path in paths.items():
    if key != "model":
        prep_data[key] = ml.data_handling(data_path=path, label_name=label_name, return_pd=True)

features = dict()
labels = dict()
features_pd = dict()
labels_pd = dict()
for dataset in prep_data:
    features[dataset] = prep_data[dataset][1]
    labels[dataset] = prep_data[dataset][2]
    features_pd[dataset] = prep_data[dataset][5]
    labels_pd[dataset] = prep_data[dataset][6]

#Reweightning

predictions = dict()
"""
for dataset in features:
    print(model(features[dataset][:,:2]))
    predictions[dataset] = model(features[dataset][:,:2]) * labels[dataset]
    print(predictions[dataset])
    print(labels[dataset])
"""
predictions["CT14 auf MMHT"] = (1/model(features["CT14 x constant"][:,:2])) * labels["CT14 x constant"]
print(predictions["CT14 auf MMHT"], labels["MMHT x constant"])

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
    prediction = np.array(predictions["CT14 auf MMHT"])[order]
    plt.plot(plot_features, prediction, label="Reweight CT14 auf MMHT")
    plt.ylabel(label_name)
    plt.xlabel(str(variabel[dataset][0]))
    if variabel[dataset][0] == "x_1" or variabel[dataset][0] == "x_2":
        plt.yscale("log")
    plt.title(dataset)
    plt.legend()
    plt.show()