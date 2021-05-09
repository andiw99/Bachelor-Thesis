import pandas as pd
import numpy as np
import ml
import MC
from scipy import integrate
from matplotlib import pyplot as plt

#Pfade in dict speichern
paths = dict()
paths[r"$x^4$-Distribution"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TrainingData60k_ep_0.163/"
save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
name = "4"
#input("namen geändert?")
save_path = save_path + name
label_name = "WQ"
trans_to_pb = False

#ticks festlegen ggf
pi_ticks = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
pi_names = np.array(["0", r"$\frac{1}{4} \pi$", r"$\frac{1}{2} \pi$", r"$\frac{3}{4} \pi$", r"$\pi$"])

#In Features und Labels unterteilen
labels_pd = dict()
features_pd = dict()
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
#Test x³ dist
#dists[r"$x^3$"] = MC.x_power_dist(power=3, offset=0.5, a=0.163, b=np.pi-0.163, normed=True)

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
    #predictions[dataset][r"$x^3$"] = dists[r"$x^3$"](features[dataset])
    labels[dataset] = np.array(MC.gev_to_pb(labels[dataset]))
    #Rescaling der lables sodass integral 1
    plt.plot(ordered_features, MC.gev_to_pb(ordered_labels))
    I = integrate.trapezoid(MC.gev_to_pb(ordered_labels)[:,0], ordered_features[:,0])
    scaling = 1 / I
    labels[dataset] = scaling * labels[dataset]




#Jetzt plotten irgendwie
for dataset in predictions:
    keys = ml.get_varying_value(features_pd=features_pd[dataset])
    ml.plot_model(features_pd=features_pd[dataset], labels=labels[dataset], predictions=predictions[dataset],
                 keys=keys, save_path=save_path, trans_to_pb=trans_to_pb, set_ylabel=r"$\rho$", set_ratio_yscale="linear",
                  autoscale_ratio=True, autoscale=True, automatic_legend=True, xticks=pi_ticks, xtick_labels=pi_names)

