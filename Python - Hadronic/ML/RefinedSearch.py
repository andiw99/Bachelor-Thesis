import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
import ml
import time
import os
import ast

#Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
pools = dict()
pools["batch_size"] = [64, 128, 256]
pools["units"] = [512]
pools["nr_layers"] =  [2]
pools["learning_rate"]= np.linspace(5e-4,15e-4,11)
pools["l2_kernel"] = [0.0]
pools["l2_bias"] = [0.0]
pools["loss_fn"] = [keras.losses.MeanAbsoluteError()]
pools["optimizer"] = [keras.optimizers.Adam]
pools["momentum"] = [0.1]
pools["dropout"] = [False]
pools["dropout_rate"] = [0]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal()]
pools["bias_initializer"] = [tf.keras.initializers.Zeros()]
pools["hidden_activation"] = [tf.nn.relu]
pools["output_activation"] = [ml.LinearActiavtion()]
#Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
names = {"batch_size", "learning_rate"}

time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14nnlo/"
data_name = "all"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicModels/RefinedSearch/"
loss_name = "best_loss"
project_name = ""

label_name = "WQ"

#Variablen...
train_frac = 0.95
training_epochs = 30
size = 15
repeat = 2
loss_function = keras.losses.MeanAbsolutePercentageError()
nesterov=True
scaling_bool = True
logarithm = True
shift = False
label_normalization = False

custom = False
new_model=True

#Daten einlsen
# Daten einlesen:
(training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(
    data_path=data_path + data_name, label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm, shift=shift, label_normalization=label_normalization,
    train_frac=train_frac)

#Menge mit bereits gesehen konfigurationen
checked_configs = ml.create_param_configs(pools=pools, size=size)
results_list = dict()

for config in checked_configs:
    #Schönere accessability
    params = dict()
    for i,param in enumerate(pools):
        params[param] = config[i]

    #Create path to save model
    model_name = ml.construct_name(params, names_set=names)
    save_path = project_path + model_name
    print("Wir initialisieren Modell ", model_name)

    #Best loss einlesen
    best_losses = None
    if os.path.exists(project_path + project_name + loss_name):
        best_losses = pd.read_csv(project_path + project_name + loss_name)

    # Verzeichnis erstellen
    if not os.path.exists(path=save_path):
        os.makedirs(save_path)

    #zweimal initialisiern um statistische Schwankungen zu verkleinern
    #trainin_time und total loss über die initialisierungen mitteln
    training_time = 0
    total_loss = 0
    for i in range(repeat):
        #Modell initialisieren
        model = ml.initialize_model(nr_layers=params["nr_layers"], units=params["units"], loss_fn=params["loss_fn"],
                                            optimizer=params["optimizer"], hidden_activation=params["hidden_activation"],
                                            output_activation=params["output_activation"], kernel_initializer=params["kernel_initializer"],
                                            bias_initializer=params["bias_initializer"], l2_kernel=params["l2_kernel"],
                                            learning_rate=params["learning_rate"], momentum=params["momentum"],
                                            nesterov=True, l2_bias=params["l2_bias"], dropout=params["dropout"],
                                            dropout_rate=params["dropout_rate"], new_model=new_model, custom=custom)

        # Training starten
        time4 = time.time()
        history = model.fit(x=train_features, y=train_labels, batch_size=params["batch_size"], epochs=training_epochs, verbose=2,
                            shuffle=True)
        time5 = time.time()
        training_time += time5 - time4

        # Losses plotten
        ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
        plt.savefig(save_path + "/training_losses")
        plt.show()

        # Überprüfen wie gut es war
        results = model(test_features)
        total_loss += float(loss_function(y_pred=transformer.retransform(results), y_true=transformer.retransform(test_labels)))

    #training_time und total loss mitteln:
    training_time = 1/repeat * training_time
    total_loss = 1/repeat * total_loss
    print("total loss:", float(total_loss))

    # Modell und config speichern
    model.save(filepath=save_path, save_format="tf")
    (config, index) = ml.save_config(new_model=new_model, model=model, learning_rate=params["learning_rate"],
                                     training_epochs=training_epochs, batch_size=params["batch_size"],
                                     total_loss=total_loss, transformer=transformer, training_time=training_time,
                                     custom=custom, loss_fn=params["loss_fn"], save_path=save_path)

    #Überprüfen ob Fortschritt gemacht wurde
    ml.check_progress(model=model, transformer=transformer, test_features=test_features, test_labels=test_labels,
                      best_losses=best_losses, project_path=project_path, project_name=project_name,
                      index=index, config=config, loss_name=loss_name)

    #Ergebnis im dict festhalten
    results_list[model_name] = "{:.2f}".format(float(total_loss))

#Ergebnisse speichern
results_list_pd = pd.DataFrame(
    results_list,
    index = [0]
)
results_list_pd = results_list_pd.transpose()
results_list_pd.to_csv(project_path + "results")


