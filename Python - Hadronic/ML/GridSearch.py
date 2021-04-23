import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import time
import os
import numpy as np
import sys

time1 = time.time()
#Daten einlesen
location = input("Auf welchem Rechner?")
root_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/"
if location == "Taurus" or location == "taurus":
    root_path = "/home/s1388135/Bachelor-Thesis/"
sys.path.insert(0, root_path)
import ml


#Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
# wichtig: Standardmodell hat index 0 in jedem pool
pools = dict()
pools["batch_size"] = [512, 128, 768, 2048, 8196]
pools["units_nr_layers"] = [(512,3), (256,5), (64,7), (1024, 2)]
pools["learning_rate"]= [1e-3, 1e-2, 1e-4, 5e-3]
pools["l2_kernel"] = [0.0]
pools["l2_bias"] = [0.0]
pools["loss_fn"] = [keras.losses.MeanAbsoluteError(), keras.losses.MeanSquaredError()]
pools["optimizer"] = [keras.optimizers.Adam, keras.optimizers.RMSprop, keras.optimizers.SGD]
pools["momentum"] = [0.1]
pools["dropout"] = [False]
pools["dropout_rate"] = [0]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal()]
pools["bias_initializer"] = [tf.keras.initializers.Zeros()]
pools["hidden_activation"] = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.elu, tf.nn.tanh, tf.nn.sigmoid]
pools["output_activation"] = [ml.LinearActiavtion()]
pools["feature_normalization"] = ["normalization"]
pools["dataset"] =["TrainingData1M", "TrainingData500k", "TrainingData2M", "TrainingData4M"]
#Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
names = {"dataset"}

vary_multiple_parameters = False

#Variablen...
train_frac = 0.95
training_epochs = 100
size = 3
output_activation = ml.LinearActiavtion()
bias_initializer =tf.keras.initializers.Zeros()
l2_bias = 0
momentum = 0.1
min_lr = 5e-8
lr_reduction=0.05
lr_factor = 0.5
nesterov = True
loss_function = keras.losses.MeanAbsolutePercentageError()
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = True
base10 = True
shift = False
label_normalization = True
feature_normalization = True
feature_rescaling = False

custom = False
new_model=True

lr_patience = 1
stopping_patience = 3
repeat = 5


#Menge mit bereits gesehen konfigurationen
checked_configs = ml.create_param_configs(pools=pools, size=size, vary_multiple_parameters=vary_multiple_parameters)
print(checked_configs)
print(len(checked_configs))
exit()
results_list = dict()

for config in checked_configs:
    #Schönere accessability
    params = dict()
    for i,param in enumerate(pools):
        params[param] = config[i]

    data_path = root_path + "/Files/Hadronic/Data/" + params["dataset"] +  "/"
    data_name = "all"
    project_path = root_path + "Files/Hadronic/Models/optimizers_comparison/"
    loss_name = "best_loss"
    project_name = ""

    label_name = "WQ"

    if params["feature_normalization"] == "rescaling":
        feature_rescaling = True
    elif params["feature_normalization"] == "normalization":
        feature_normalization = True

    #training_epochs = int(1/200 * params["batch_size"]) + 10

    #Callbacks initialisieren
    #min delta initialiseren
    #Fall dass msle benutzt werden soll behandeln, label normalization auf [-1,1] verhindern
    actual_label_normalization = label_normalization
    actual_hidden_activation = params["hidden_activation"]
    if params["loss_fn"].name == "mean_absolute_error":
        min_delta = 1e-5
        label_normalization = actual_label_normalization
        params["hidden_activation"] = actual_hidden_activation
    elif params["loss_fn"].name == "mean_squared_error":
        min_delta = 1e-6
        label_normalization = actual_label_normalization
        params["hidden_activation"] = actual_hidden_activation
    elif params["loss_fn"].name == "mean_squared_logarithmic_error":
        min_delta = 1e-7
        label_normalization = False
        params["hidden_activation"] = tf.nn.leaky_relu

    reduce_lr = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=lr_reduction, min_lr=min_lr))
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience, min_delta=min_delta, min_lr=min_lr)
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * min_delta, patience=stopping_patience)
    callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

    # Daten einlsen
    # Daten einlesen:
    (training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(
        data_path=data_path + data_name, label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm, base10=base10,
        shift=shift, label_normalization=label_normalization, feature_rescaling=feature_rescaling,
        train_frac=train_frac)


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
    total_losses = []
    models = []
    for i in range(repeat):
        #Modell initialisieren
        models.append(ml.initialize_model(nr_layers=params["units_nr_layers"][1], units=params["units_nr_layers"][0], loss_fn=params["loss_fn"], optimizer=params["optimizer"],
                                            hidden_activation=params["hidden_activation"], output_activation=output_activation,
                                            kernel_initializer=params["kernel_initializer"], bias_initializer=bias_initializer, l2_kernel=params["l2_kernel"],
                                            learning_rate=params["learning_rate"], momentum=momentum, nesterov=nesterov,
                                            l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                                            new_model=new_model, custom=custom, feature_normalization=feature_normalization))
    for i,model in enumerate(models):
    # Training starten
        time4 = time.time()
        history = model.fit(x=train_features, y=train_labels, batch_size=params["batch_size"], epochs=training_epochs,
                            callbacks = callbacks, verbose=2, shuffle=True)
        time5 = time.time()
        training_time += time5 - time4

        # Losses plotten
        ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
        plt.savefig(save_path + "/training_losses")
        plt.show()

        # Überprüfen wie gut es war
        results = model(test_features)
        loss = float(loss_function(y_pred=transformer.retransform(results), y_true=transformer.retransform(test_labels)))
        print("Loss von Durchgang Nummer ", i, " : ", loss)
        total_losses.append(loss)

    #training_time und total loss mitteln:
    avg_total_loss = total_losses[np.argmin(total_losses)]
    smallest_loss = np.min(total_losses)
    loss_error = np.std(total_losses)
    training_time = 1 / repeat * training_time
    print("Losses of the specific cycle:", total_losses)
    print("average Loss over ", repeat, "cycles:", np.mean(total_losses))
    print("Das beste Modell (Modell Nr.", np.argmin(total_losses), ") wird gespeichert")
    # Modell und config speichern
    model = models[np.argmin(total_losses)]
    model.save(filepath=save_path, save_format="tf")
    (config, index) = ml.save_config(new_model=new_model, save_path=save_path, model=model, learning_rate=params["learning_rate"],
                                     training_epochs=training_epochs, batch_size=params["batch_size"],
                                     avg_total_Loss=avg_total_loss, smallest_loss=smallest_loss, loss_error=loss_error, total_losses=total_losses,
                                     transformer=transformer, training_time=training_time,
                                     custom=custom, loss_fn=params["loss_fn"], feature_handling= params["feature_normalization"],
                                     min_delta = min_delta, nr_hidden_layers=params["nr_layers"])

    #Überprüfen ob Fortschritt gemacht wurde
    ml.check_progress(model=models[np.argmin(total_losses)], transformer=transformer, test_features=test_features, test_labels=test_labels,
                      best_losses=best_losses, project_path=project_path, project_name=project_name,
                      index=index, config=config, loss_name=loss_name)

    #Ergebnis im dict festhalten
    results_list[model_name] = "{:.2f}".format(float(avg_total_loss))

    #Ergebnisse speichern
    results_list_pd = pd.DataFrame(
        results_list,
        index = [0]
    )
    results_list_pd = results_list_pd.transpose()
    results_list_pd.to_csv(project_path + "results")


