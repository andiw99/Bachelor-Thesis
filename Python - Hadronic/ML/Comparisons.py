import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import time
import os
import numpy as np
import sys

time1 = time.time()
# Daten einlesen
location = input("Auf welchem Rechner?")
root_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/"
if location == "Taurus" or location == "taurus":
    root_path = "/home/s1388135/Bachelor-Thesis/"
sys.path.insert(0, root_path)
import ml

# Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
# wichtig: Standardmodell hat index 0 in jedem pool
pools = dict()
pools["batch_size"] = [256, 512, 768, 1024, 128]
pools["units_nr_layers"] = [(256, 5), (512, 3), (64, 7), (1024, 2), (128, 6)]
pools["learning_rate"] = [1e-2, 1e-3, 1e-4, 5e-3]
pools["l2_kernel"] = [0.0]
pools["l2_bias"] = [0.0]
pools["loss_fn"] = [keras.losses.MeanAbsoluteError(),
                    keras.losses.MeanSquaredError(), keras.losses.Huber()]
pools["optimizer"] = [keras.optimizers.Adam, keras.optimizers.RMSprop, keras.optimizers.SGD]
pools["momentum"] = [0.1]
pools["dropout"] = [False]
pools["dropout_rate"] = [0]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal()]
pools["bias_initializer"] = [tf.keras.initializers.Zeros()]
pools["hidden_activation"] = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.elu,
                              tf.nn.sigmoid, tf.nn.tanh]
pools["output_activation"] = [ml.LinearActiavtion()]
pools["feature_normalization"] = ["normalization"]
pools["scaling_bool"] = [True]
pools["logarithm"] = [True]
pools["base10"] = [True]
pools["label_normalization"] = [False]
pools["min_delta"] = [5e-6]
pools["min_lr"] = [5e-8]
pools["dataset"] = ["TrainingData2M"]
# Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
names = {"loss_fn", "units_nr_layers", "optimizer", "hidden_activation",
         "dataset", "batch_size",
         "learning_rate", "units_nr_layers", "label_normalization", "base10",
         "feature_normalization", }

vary_multiple_parameters = False

# Variablen...
train_frac = 0.95
training_epochs = 100
size = 100
lr_reduction = 0.05
lr_factor = 0.5
nesterov = True
loss_function = keras.losses.MeanAbsolutePercentageError()
feature_rescaling = False

custom = False
new_model = True

lr_patience = 1
stopping_patience = 3
repeat = 5

# Menge mit bereits gesehen konfigurationen
checked_configs = ml.create_param_configs(pools=pools, size=size,
                                          vary_multiple_parameters=vary_multiple_parameters)
results_list = dict()

for config in checked_configs:
    # Schönere accessability
    params = dict()
    for i, param in enumerate(pools):
        params[param] = config[i]

    data_path = root_path + "/Files/Hadronic/Data/" + params["dataset"] + "/"
    data_name = "all"
    project_path = root_path + "Files/Hadronic/Models/LastRandomSearch/"
    if not vary_multiple_parameters:
        project_path += str(config[-1]) + "/"
    loss_name = "best_loss"
    project_name = ""

    label_name = "WQ"

    if params["feature_normalization"] == "rescaling":
        feature_rescaling = True
    elif params["feature_normalization"] == "normalization":
        feature_normalization = True

    # Trainingsparameter ein wenig nach batches anpassen
    training_epochs = int(1 / 100 * params["batch_size"]) + 90
    lr_reduction = 25 / params["batch_size"]

    # Callbacks initialisieren
    # min delta initialiseren
    reduce_lr = keras.callbacks.LearningRateScheduler(
        ml.class_scheduler(reduction=lr_reduction, min_lr=params["min_lr"]))
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss",
                                                             factor=lr_factor,
                                                             patience=lr_patience,
                                                             min_delta=params[
                                                                 "min_delta"],
                                                             min_lr=params[
                                                                 "min_lr"])
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss",
                                                   min_delta=1e-1 * params[
                                                       "min_delta"],
                                                   patience=stopping_patience)
    callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

    # Daten einlsen
    # Daten einlesen:
    (training_data, train_features, train_labels, test_features, test_labels,
     transformer) = ml.data_handling(
        data_path=data_path + data_name, label_name=label_name,
        scaling_bool=params["scaling_bool"], logarithm=pools["logarithm"],
        base10=params["base10"],
        label_normalization=params["label_normalization"],
        feature_rescaling=feature_rescaling,
        train_frac=train_frac)

    # Create path to save model
    if not vary_multiple_parameters:
        names = {config[-1]}
    model_name = ml.construct_name(params, names_set=names)
    save_path = project_path + model_name
    print("Wir initialisieren Modell ", model_name)

    # Best loss einlesen
    best_losses = None
    if os.path.exists(project_path + project_name + loss_name):
        best_losses = pd.read_csv(project_path + project_name + loss_name)

    # Verzeichnis erstellen
    if not os.path.exists(path=save_path):
        os.makedirs(save_path)

    # zweimal initialisiern um statistische Schwankungen zu verkleinern
    # trainin_time und total loss über die initialisierungen mitteln
    training_time = 0
    total_losses = []
    models = []
    for i in range(repeat):
        # Modell initialisieren
        models.append(
            ml.initialize_model(nr_layers=params["units_nr_layers"][1],
                                units=params["units_nr_layers"][0],
                                loss_fn=params["loss_fn"],
                                optimizer=params["optimizer"],
                                hidden_activation=params["hidden_activation"],
                                output_activation=params["output_activation"],
                                kernel_initializer=params[
                                    "kernel_initializer"],
                                bias_initializer=params["bias_initializer"],
                                l2_kernel=params["l2_kernel"],
                                learning_rate=params["learning_rate"],
                                momentum=params["momentum"], nesterov=nesterov,
                                l2_bias=params["l2_bias"],
                                dropout=params["dropout"],
                                dropout_rate=params["dropout_rate"],
                                new_model=new_model, custom=custom,
                                feature_normalization=pools[
                                    "feature_normalization"]))
    for i, model in enumerate(models):
        # Training starten
        time4 = time.time()
        history = model.fit(x=train_features, y=train_labels,
                            batch_size=params["batch_size"],
                            epochs=training_epochs,
                            callbacks=callbacks, verbose=2, shuffle=True)
        time5 = time.time()
        training_time += time5 - time4

        # Losses plotten
        ml.make_losses_plot(history=history)
        plt.savefig(save_path + "/training_losses")
        plt.show()

        # Überprüfen wie gut es war
        results = model(test_features)
        loss = float(loss_function(y_pred=transformer.retransform(results),
                                   y_true=transformer.retransform(
                                       test_labels)))
        print("Loss von Durchgang Nummer ", i, " : ", loss)
        total_losses.append(loss)

    # training_time und total loss mitteln:
    avg_total_loss = np.mean(total_losses)
    smallest_loss = np.min(total_losses)
    loss_error = np.std(total_losses)
    training_time = 1 / repeat * training_time
    print("Losses of the specific cycle:", total_losses)
    print("average Loss over ", repeat, "cycles:", np.mean(total_losses))
    print("Das beste Modell (Modell Nr.", np.argmin(total_losses),
          ") wird gespeichert")
    # Modell und config speichern
    model = models[np.argmin(total_losses)]
    model.save(filepath=save_path, save_format="tf")
    (config, index) = ml.save_config(new_model=new_model, save_path=save_path,
                                     model=model,
                                     learning_rate=params["learning_rate"],
                                     training_epochs=training_epochs,
                                     batch_size=params["batch_size"],
                                     avg_total_Loss=avg_total_loss,
                                     smallest_loss=smallest_loss,
                                     loss_error=loss_error,
                                     total_losses=total_losses,
                                     transformer=transformer,
                                     training_time=training_time,
                                     custom=custom, loss_fn=params["loss_fn"],
                                     feature_handling=params[
                                         "feature_normalization"],
                                     min_delta=params["min_delta"],
                                     nr_hidden_layers=
                                     params["units_nr_layers"][1],
                                     units=params["units_nr_layers"][0])

    # Überprüfen ob Fortschritt gemacht wurde
    ml.check_progress(model=models[np.argmin(total_losses)],
                      transformer=transformer, test_features=test_features,
                      test_labels=test_labels,
                      best_losses=best_losses, project_path=project_path,
                      project_name=project_name,
                      index=index, config=config, loss_name=loss_name)

    # Ergebnis im dict festhalten
    results_list[model_name] = "{:.2f}".format(float(avg_total_loss))

    # Ergebnisse speichern
    results_list_pd = pd.DataFrame(
        results_list,
        index=[0]
    )
    results_list_pd = results_list_pd.transpose()
    results_list_pd.to_csv(project_path + "results")



