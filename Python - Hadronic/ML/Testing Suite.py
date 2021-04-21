from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import ml
import time
import ast

#Was wird untersucht
investigated_parameter = "models"
label_name = "WQ"

#Pfade eingeben
testing_data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/TestData"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/"

#Zu vergleichende models laden
model_paths = dict()
#model_paths["best model "] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/Loss_comparison/best_model"
model_paths["best guess"] = project_path + "best_guess_important_range"
model_paths["transferred \n model"] = project_path + "/transferred_model_mid"
#model_paths["2000000"] = project_path + "/dataset_MuchTrainingDataMidRange_"
#model_paths["Scaling \n Label-Norm"] = project_path + "/scaling_bool_True_base10_False_label_normalization_True_"
#model_paths["Scaling"] = project_path + "/scaling_bool_True_base10_False_label_normalization_False_"

#...model_paths["MSE"] =
config_paths = dict()
for model in model_paths:
    config_paths[model] = model_paths[model] + "/" + "config"
#Zu testende Modelle und transformer laden
models = dict()
transformers = dict()
for model in model_paths:
    models[model], transformers[model] = ml.import_model_transformer(model_paths[model])


(_, test_features, test_labels,_, _, test_features_pd, test_labels_pd, _) =\
    ml.data_handling(data_path=testing_data_path, label_name=label_name, return_pd=True)

#Zu testende Kriterien: Berechnungszeit, Trainingszeit(schon gegeben), Validation loss
#Validation Loss berechnen, am besten MSE und MAPE
MSE_loss_fn = keras.losses.MeanSquaredError()
MAPE_loss_fn = keras.losses.MeanAbsolutePercentageError()

MSE = dict()
MAPE = dict()
Calc_times = dict()
Predictions = dict()
for model in models:
    #Berechnung der Labels timen
    time_pre_calc = time.time()
    Predictions[model] = transformers[model].retransform(models[model](test_features))
    time_post_calc = time.time()
    #Zeit auf Berechnung von 1M Datenpunkten normieren
    time_per_million = ((time_post_calc - time_pre_calc)/(float(tf.size(test_labels)))) * 1e+6
    #In dicts abspeichern
    print("model", model, "MSE", float(MSE_loss_fn(y_true= test_labels, y_pred= Predictions[model])), "MAPE",
          float((MAPE_loss_fn(y_true= test_labels, y_pred=Predictions[model]))))
    MSE[model] = float(MSE_loss_fn(y_true= test_labels, y_pred= Predictions[model]))
    MAPE[model] = float((MAPE_loss_fn(y_true= test_labels, y_pred=Predictions[model])))
    Calc_times[model] = time_per_million

#Ergebnisse speichern:
Results = pd.DataFrame(
    {
        "model": list(models.keys()),
        "MSE": list(MSE.values()),
        "MAPE": list(MAPE.values()),
        "Time per 1M": list(Calc_times.values())
    }
)
Results.to_csv(project_path + "/results")

#Vergleich plotten
names = list(models.keys())
MSE_losses = list(MSE.values())
MAPE_losses = list(MAPE.values())

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
rects1 = ax1.bar(x-width/2, MSE_losses, width, label="MSE", color="orange")
rects2 = ax2.bar(x+width/2, MAPE_losses, width, label="MAPE")

ax1.set_title("Validation loss for different " + investigated_parameter)
ax1.set_ylabel("MSE")
ax2.set_ylabel("MAPE")
ax1.set_xticks(x)
ax2.set_xticks(x)
ax1.set_xticklabels(names)
fig.legend()
fig.tight_layout()
plt.show()


