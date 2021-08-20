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
root_path = "//"
if location == "Taurus" or location == "taurus":
    root_path = "/home/s1388135/Bachelor-Thesis/"
sys.path.insert(0, root_path)
import ml

# Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
# wichtig: Standardmodell hat index 0 in jedem pool
pools = dict()
pools["batch_size"] = [512, 128, 768, 2048, 8196]
pools["learning_rate"] = [1e-3, 1e-4, 5e-3, 1e-5]
pools["loss_fn"] = [keras.losses.MeanAbsoluteError()]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal()]
pools["bias_initializer"] = [tf.keras.initializers.Zeros()]
pools["hidden_activation"] = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.sigmoid]
pools["output_activation"] = [ml.LinearActiavtion()]
pools["min_delta"] = [2e-6]
pools["min_lr"] = [5e-8]
pools["source_model"] = ["best_guess_4M"]
pools["dataset"] = ["TransferData1M", "TransferData2M", "TransferData500k"]
pools["rm_layers"] = [1, 2]
pools["add_layers"] = [0, 1, 2]
pools["units"] = [64, 128, 512]
pools["fine_tuning"] = [False, True]
pools["offset"] = [2, 4, 6]
# Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
names = {"loss_fn", "hidden_activation", "rm_layers", "add_layers", "units",
         "dataset", "batch_size", "learning_rate", "fine_tuning", "offset"}

vary_multiple_parameters = True
grid_search = True

# Variablen...
train_frac = 0.95 # TODO train_frac so einstellen, dass man immer die gleiche anzahl an validation daten hat
validation_total = 15000
size = 4
if grid_search:
    size = 1	
    for param in pools:
    	size *= len(pools[param])
    size = np.minimum(100, size)    	
    print("size:", size)
   
lr_reduction = 0.05
lr_factor = 0.5
nesterov = True
loss_function = keras.losses.MeanAbsolutePercentageError()
new_model = False
transfer = True

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
    # Pfad mit daten konstruieren
    data_path = root_path + "/Files/Transfer/Data/" + params["dataset"] + "/"
    data_name = "all"
    project_path = root_path + "Files/Transfer/Models/transferred_search/"
    if not vary_multiple_parameters:
        project_path += str(config[-1]) + "/"
    loss_name = "best_loss"
    project_name = ""

    label_name = "WQ"

    # Transformer und zu transformierendes Model laden
    model_path = root_path + "/Files/Hadronic/Models/" + params["source_model"]
    (source_model, transformer) = ml.load_model_and_transormer(model_path=model_path)

    # Trainingsparameter ein wenig nach batches anpassen
    training_epochs = int(1 / 100 * params["batch_size"]) + 90
    lr_reduction = 50 / params["batch_size"]

    # Callbacks initialisieren
    # min delta initialiseren
    reduce_lr = keras.callbacks.LearningRateScheduler(
        ml.class_scheduler(reduction=lr_reduction, min_lr=params["min_lr"], offset=params["offset"]))
    reduce_lr_on_plateau =\
        keras.callbacks.ReduceLROnPlateau(monitor="loss",
                                        factor=lr_factor,
                                        patience=lr_patience,
                                        min_delta=params["min_delta"],
                                        min_lr=params["min_lr"])
    early_stopping =\
        keras.callbacks.EarlyStopping(monitor="loss",
                                       min_delta=1e-1 * params["min_delta"],
                                       patience=stopping_patience)
    callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

    # Daten einlsen
    # Daten einlesen:
    (training_data, train_features, train_labels, test_features, test_labels,
     _) = ml.data_handling(
        data_path=data_path + data_name, label_name=label_name,
        transformer=transformer, validation_total=validation_total)

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
        #Wir müssen das Source_model jedes mal neu laden
        (source_model, _) = ml.load_model_and_transormer(model_path=model_path)

        # Modell initialisieren
        models.append(ml.initialize_model(transfer=transfer, new_model=new_model,
                                          source_model=source_model, units=params["units"],
                                          learning_rate=params["learning_rate"],
                                          loss_fn=params["loss_fn"],
                                          hidden_activation=params["hidden_activation"],
                                          output_activation=params["output_activation"],
                                          kernel_initializer=params["kernel_initializer"],
                                          bias_initializer=params["bias_initializer"],
                                          rm_layers=params["rm_layers"],
                                          add_layers=params["add_layers"]
                                          ))
        # Training starten
        time4 = time.time()
        history = models[i].fit(x=train_features, y=train_labels,
                            batch_size=params["batch_size"],
                            epochs=training_epochs,
                            callbacks=callbacks, verbose=2, shuffle=True)
        if params["fine_tuning"]:
            print(models[i].summary())
            # modell auftauen und mit neuer lr compilen
            models[i].trainable = True
            models[i].compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"] * 1e-2,
                                                          clipvalue=1.5),
                          loss=params["loss_fn"])
            print(models[i].summary())
            #history addieren? geht vermutlich nicht
            fine_tuning_history = models[i].fit(x=train_features, y=train_labels,
                            batch_size=params["batch_size"],
                            epochs=training_epochs,
                            callbacks=callbacks, verbose=2, shuffle=True)

            #history losses zusammenpacken
            history = history.history["loss"] + fine_tuning_history.history["loss"]


        time5 = time.time()
        training_time += time5 - time4

        # Losses plotten
        ml.make_losses_plot(history=history)
        plt.savefig(save_path + "/training_losses")
        plt.show()

        # Überprüfen wie gut es war
        results = models[i].predict(test_features)
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
    (config, index) = ml.save_config(new_model=True, save_path=save_path,
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
                                     loss_fn=params["loss_fn"],
                                     min_delta=params["min_delta"],
                                     offset=params["offset"],
                                     fine_tuning=params["fine_tuning"],
                                     source_model=params["source_model"],)

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


