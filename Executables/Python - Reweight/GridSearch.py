import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import time
import os
import numpy as np
import sys


def main():
    time1 = time.time()
    #Daten einlesen
    location = input("Auf welchem Rechner?")
    root_path = "//"
    if location == "Taurus" or location == "taurus":
        root_path = "/home/s1388135/Bachelor-Thesis/"
    sys.path.insert(0, root_path)
    import ml



    #Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
    # wichtig: Standard-modell hat index 0 in jedem pool
    pools = dict()
    pools["batch_size"] = [512, 128, 768, 2048, 64]
    pools["units"] = [32, 64, 128, 256]
    pools["nr_layers"] = [1,2,3,4]
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
    pools["hidden_activation"] = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.sigmoid]
    pools["output_activation"] = [ml.LinearActiavtion()]
    pools["feature_normalization"] = ["normalization", None]
    pools["logarithm"] = [False]
    pools["scaling_bool"] = [False]
    pools["base10"] = [False]
    pools["label_normalization"] = [True, False]
    pools["min_delta"] = [5e-6]
    pools["min_lr"] = [5e-8]
    pools["dataset"] =["TrainingData200k_cut_x_08", "TrainingData500k_cut_x_08", "TrainingData1M_cut_x_08"]
    #Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
    names = {"loss_fn", "units", "nr_layers", "optimizer", "hidden_activation", "dataset",
             "batch_size", "learning_rate", "feature_normalization", "label_normalization"}

    vary_multiple_parameters = True

    #Variablen...
    train_frac = 0.95
    training_epochs = 100
    size = 150
    min_lr = 1e-7
    lr_reduction=0.05
    lr_factor = 0.5
    nesterov = True
    loss_function = keras.losses.MeanAbsolutePercentageError()
    shift = False


    custom = False
    new_model = True

    lr_patience = 1
    stopping_patience = 3
    repeat = 2



    #Menge mit bereits gesehen konfigurationen
    checked_configs = ml.create_param_configs(pools=pools, size=size, vary_multiple_parameters=vary_multiple_parameters)
    results_list = dict()

    for config in checked_configs:
        #Schönere accessability
        params = dict()
        for i,param in enumerate(pools):
            params[param] = config[i]

        data_path = root_path + "Files/Reweight/Data/" + params["dataset"] +  "/"
        data_name = "all"
        project_path = root_path + "Files/Reweight/Models/RandomSearch_logartihm_false/"
        loss_name = "best_loss"
        project_name = ""

        label_name = "reweight"
        
        feature_normalization = None
        if params["feature_normalization"] == "normalization":
            feature_normalization = True

        #training_epochs = int(1/200 * params["batch_size"]) + 10

        #Callbacks initialisieren
        #min delta initialiseren
        reduce_lr = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=lr_reduction, min_lr=params["min_lr"]))
        reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience, min_delta=params["min_delta"], min_lr=params["min_lr"])
        early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * params["min_delta"], patience=stopping_patience)
        callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

        # Daten einlsen
        # Daten einlesen:
        (training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(
            data_path=data_path + data_name, label_name=label_name, scaling_bool=params["scaling_bool"], logarithm=params["logarithm"], base10=params["base10"],
            shift=shift, label_normalization=params["label_normalization"],
            train_frac=train_frac)


        #Create path to save model
        if not vary_multiple_parameters:
            names = {config[-1]}
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
            models.append(ml.initialize_model(nr_layers=params["nr_layers"], units=params["units"], loss_fn=params["loss_fn"], optimizer=params["optimizer"],
                                                hidden_activation=params["hidden_activation"], output_activation=params["output_activation"],
                                                kernel_initializer=params["kernel_initializer"], bias_initializer=params["bias_initializer"], l2_kernel=params["l2_kernel"],
                                                learning_rate=params["learning_rate"], momentum=params["momentum"], nesterov=nesterov,
                                                l2_bias=params["l2_bias"], dropout=params["dropout"], dropout_rate=params["dropout_rate"],
                                                new_model=new_model, custom=custom, feature_normalization=feature_normalization))
        for i,model in enumerate(models):
        # Training starten
            time4 = time.time()
            history = model.fit(x=train_features, y=train_labels, batch_size=params["batch_size"], epochs=training_epochs,
                                callbacks = callbacks, verbose=2, shuffle=True)
            time5 = time.time()
            training_time += time5 - time4

            # Losses plotten
            fig, ax = ml.make_losses_plot(history=history)
            fig.savefig(save_path + "/training_losses")
            plt.show()

            # Überprüfen wie gut es war
            results = model(test_features)
            loss = float(loss_function(y_pred=transformer.retransform(results), y_true=transformer.retransform(test_labels)))
            print("Loss von Durchgang Nummer ", i, " : ", loss)
            total_losses.append(loss)

        #training_time und total loss mitteln:
        avg_total_loss = np.mean(total_losses)
        smallest_loss = np.min(total_losses)
        loss_error = np.std(total_losses, ddof=1)
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
                                         min_delta = params["min_delta"], nr_hidden_layers=params["nr_layers"])

        #Überprüfen ob Fortschritt gemacht wurde
        ml.check_progress(model=models[np.argmin(total_losses)], transformer=transformer, test_features=test_features, test_labels=test_labels,
                          best_losses=best_losses, project_path=project_path, project_name=project_name,
                          index=index, config=config, loss_name=loss_name)

        #Ergebnis im dict festhalten
        results_list[model_name] = "{:.5f}".format(float(avg_total_loss))

        #Ergebnisse speichern
        results_list_pd = pd.DataFrame(
            results_list,
            index = [0]
        )
        results_list_pd = results_list_pd.transpose()
        results_list_pd.to_csv(project_path + "results")


if __name__ == "__main__":
    main()

