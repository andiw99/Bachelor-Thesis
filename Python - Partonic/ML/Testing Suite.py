from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import ml
import time
import ast

#Projekt festlegen
project = "Partonic"
arg = "Theta"
investigated_parameter = "Training losses"

#Pfade eingeben
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/" + project
testing_data_path = project_path + "/" + project + "Data/" + "Test" + arg + "Data"
#Zu vergleichende models laden
model_paths = dict()
model_paths["best model"] = project_path + "/" + project + "Models/" + project+arg + "/" + "best_model"
model_paths["MSE"] = project_path + "/" + project + "Models/" + project+arg + "/" + "Logarithm+MSE"
model_paths["MAE"] = project_path + "/" + project + "Models/" + project+arg + "/" + "Logarithm+MAE"
#...model_paths["MSE"] =
config_paths = dict()
for model in model_paths:
    config_paths[model] = model_paths[model] + "/" + "config"
#Zu testende Modelle laden
models = dict()
for model in model_paths:
    models[model] = keras.models.load_model(model_paths[model])
transformers = dict()
#transformer der Modell laden
for model in config_paths:
    config = pd.read_csv(config_paths[model])
    config = config.transpose()
    transformer_config = ast.literal_eval(config[8][1])
    transformers[model] = ml.LabelTransformation(config=transformer_config)

#Testdaten laden, vermutlich nur ein Set mit Testdaten?
data_raw = pd.read_csv(testing_data_path)

#Daten vorbereiten
#In Features und Labels unterteilen
test_labels_pd = data_raw.pop("WQ")
test_features_pd = data_raw

#Aus den Pandas Dataframes tf-Tensoren machen
for i,key in enumerate(test_features_pd):
    if i == 0:
        test_features = tf.constant([test_features_pd[key]], dtype="float32")
    else:
        more_features = tf.constant([test_features_pd[key]], dtype="float32")
        test_features = tf.experimental.numpy.append(test_features, more_features, axis=0)

#Dimensionen arrangieren
test_features = tf.transpose(test_features)
test_labels = tf.math.abs(tf.transpose(tf.constant([test_labels_pd], dtype="float32")))


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
Results.to_csv(project_path + "/" + project + "Models/" + project+arg + "/Testresults")

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


