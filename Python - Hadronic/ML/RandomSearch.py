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


#Grid erstellen
pools = dict()
pools["batch_size"] = [32, 128, 512]
pools["units"] = [128, 256, 512]
pools["nr_layers"] =  [2, 3]
pools["learning_rate"]=[1e-2, 1e-3, 1e-4]
pools["l2_kernel"] = [0, 0.0001]
pools["loss_fn"] = [keras.losses.MeanSquaredError(), keras.losses.MeanAbsoluteError()]
pools["optimizer"] = [keras.optimizers.Adam, keras.optimizers.SGD]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal(), tf.keras.initializers.RandomNormal()]
pools["hidden_activation"] = [tf.nn.leaky_relu, tf.nn.sigmoid, tf.nn.relu]


time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14nnlo/"
data_name = "all"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicModels/RandomSearch/"
loss_name = "best_loss"
project_name = ""

label_name = "WQ"

#Variablen...
train_frac = 0.95
training_epochs = 30
size = 60
output_activation = ml.LinearActiavtion()
bias_initializer =tf.keras.initializers.Zeros()
l2_bias = 0
momentum = 0.1
nesterov = True
loss_function = keras.losses.MeanAbsolutePercentageError()
dropout = False
dropout_rate = 0.1
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
    model_name = ml.construct_name(params)
    save_path = project_path + model_name
    print("Wir initialisieren Modell ", model_name)

    #Best loss einlesen
    best_losses = None
    if os.path.exists(project_path + project_name + loss_name):
        best_losses = pd.read_csv(project_path + project_name + loss_name)

    # Verzeichnis erstellen
    if not os.path.exists(path=save_path):
        os.makedirs(save_path)

    #Modell initialisieren
    model = ml.initialize_model(nr_layers=params["nr_layers"], units=params["units"], loss_fn=params["loss_fn"], optimizer=params["optimizer"],
                                        hidden_activation=params["hidden_activation"], output_activation=output_activation,
                                        kernel_initializer=params["kernel_initializer"], bias_initializer=bias_initializer, l2_kernel=params["l2_kernel"],
                                        learning_rate=params["learning_rate"], momentum=momentum, nesterov=nesterov,
                                        l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                                        new_model=new_model, custom=custom)

    # Training starten
    time4 = time.time()
    history = model.fit(x=train_features, y=train_labels, batch_size=params["batch_size"], epochs=training_epochs, verbose=2,
                        shuffle=True)
    time5 = time.time()
    training_time = time5 - time4

    # Losses plotten
    ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
    plt.savefig(save_path + "/training_losses")
    plt.show()

    # Überprüfen wie gut es war
    results = model(test_features)
    total_loss = loss_function(y_pred=transformer.retransform(results), y_true=transformer.retransform(test_labels))
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


