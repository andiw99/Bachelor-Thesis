import pandas as pd
import numpy as np
import ml
import MC
from scipy import integrate


#Pfade in dict speichern
paths = dict()
paths["Distribution"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TrainingData60k_ep_0.163/"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/"
name = "IS_dist_comparison"
#input("namen geändert?")
save_path = save_path + name
label_name = "WQ"
trans_to_pb = False


#In Features und Labels unterteilen
features_pd = dict()
labels_pd = dict()
features = dict()
labels = dict()
for dataset, path in paths.items():
    if dataset != "model":
        (_, features[dataset], labels[dataset], _, _, features_pd[dataset], labels_pd[dataset], _) =\
            ml.data_handling(data_path=path + "all", label_name=label_name, return_pd=True, label_cutoff=False)

# Wk verteilung initialisieren.
configs = dict()
variables = dict()
for dataset in paths:
    variables[dataset] = dict()
    configs[dataset] = pd.read_csv(paths[dataset] + "config")
    for key in configs[dataset]:
        variables[dataset][key] = float(configs[dataset][key][0])

dists = dict()
for dataset in variables:
    print(variables[dataset])
    dists[dataset] = MC.x_power_dist(power=variables[dataset]["power"], offset=variables[dataset]["offset"],
                                     a=variables[dataset]["epsilon"], b=np.pi-variables[dataset]["epsilon"], normed=True)



# Für jedes Dataset predictions und losses berechnen
# predictions, losses:
predictions = dict()
losses = dict()
for dataset in features:
    # analytische werte ordnen
    order = np.argsort(np.array(features[dataset][:,0]))
    ordered_features = np.array(features[dataset])[order]
    ordered_labels = np.array(labels[dataset])[order]
    predictions[dataset] = dict()
    predictions[dataset][dataset] = dists[dataset](features[dataset])
    labels[dataset] = np.array(MC.gev_to_pb(labels[dataset]))
    print(predictions[dataset], predictions[dataset][dataset])
    print("labels vor trans", labels[dataset])
    print("features vor trans", features[dataset])
    #Rescaling der lables sodass integral 1
    I = integrate.trapezoid(MC.gev_to_pb(ordered_features[:,0]), MC.gev_to_pb(ordered_labels[:,0]))
    scaling = 1 / I
    print("Integral", I, "integral/2pi", I/(2*np.pi))
    labels[dataset] = scaling * labels[dataset]
    print(labels[dataset])




#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                 keys=keys, save_path=save_path, trans_to_pb=trans_to_pb)

