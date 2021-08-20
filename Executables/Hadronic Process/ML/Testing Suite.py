from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import ml
import time


def main():
    #Was wird untersucht
    investigated_parameter = "models"
    label_name = "WQ"

    #Pfade eingeben
    testing_data_path = "//Files/Transfer/Data/TransferTestData50k/all"
    project_path = "//Files/Hadronic/Models/"

    #Zu vergleichende models laden
    model_paths = dict()
    #model_paths["best model "] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/Loss_comparison/best_model"
    model_paths["best guess"] = project_path + "best_guess_4M"
    model_paths["transferred \n model"] = project_path + "transferred_model"
    model_paths["transferred \n model, 2M"] = project_path + "transferred_model_2M"
    model_paths["transferred \n model, new layer"] = project_path + "transferred_model_2M_new_layer"
    #model_paths["2000000"] = project_path + "/dataset_MuchTrainingDataMidRange_"
    #model_paths["Scaling \n Label-Norm"] = project_path + "/scaling_bool_True_base10_False_label_normalization_True_"
    #model_paths["Scaling"] = project_path + "/scaling_bool_True_base10_False_label_normalization_False_"
    #...model_paths["MSE"] =

    #festlegen ob modelle an Datenmengen getestet werden oder die Losses aus config ausgelesen werden
    from_config = True

    config_paths = dict()
    for model in model_paths:
        config_paths[model] = model_paths[model] + "/" + "config"

    #Zu testende Kriterien: Berechnungszeit, Trainingszeit(schon gegeben), Validation loss


    MSE = dict()
    MAPE = dict()
    avg_MAPE = dict()
    MAPE_error = dict()
    training_time = dict()
    calc_times = dict()
    # Zu testende Modelle und transformer laden
    models = dict()
    transformers = dict()
    Predictions = dict()

    for model in model_paths:
        models[model], transformers[model] = ml.import_model_transformer(
            model_paths[model])

    (_, test_features, test_labels, _, _, test_features_pd, test_labels_pd, _) = \
        ml.data_handling(data_path=testing_data_path, label_name=label_name,
                         return_pd=True)

    if from_config:
        configs = dict()
        for model in config_paths:
            configs[model] = pd.read_csv(config_paths[model], index_col="property").transpose()
            MAPE[model] = float(configs[model]["smallest loss"][0])
            MAPE_error[model] = float(configs[model]["loss error"][0])
            avg_MAPE[model] = float(configs[model]["avg validation loss"][0])
            MSE[model] = 0.1
            print(configs[model])
            print(configs[model].keys())
            print(configs[model]["training time:"]) #TODO : entfernen
            training_time[model] = float(configs[model]["training time:"][0])

            #Berechnung der Labels timen
            time_pre_calc = time.time()
            Predictions[model] = transformers[model].retransform(models[model](test_features))
            time_post_calc = time.time()
            time_per_million = ((time_post_calc - time_pre_calc)/(float(tf.size(test_labels)))) * 1e+6
            calc_times[model] = time_per_million

    else:
        # Validation Loss berechnen, am besten MSE und MAPE
        MSE_loss_fn = keras.losses.MeanSquaredError()
        MAPE_loss_fn = keras.losses.MeanAbsolutePercentageError()

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
            calc_times[model] = time_per_million
            # Nicht berechenbare dictionarys mit nullen füllen
            MAPE_error[model] = 0
            avg_MAPE[model] = MAPE[model]

        #Ergebnisse speichern:
        Results = pd.DataFrame(
            {
                "model": list(models.keys()),
                "MSE": list(MSE.values()),
                "MAPE": list(MAPE.values()),
                "Time per 1M": list(calc_times.values())
            }
        )
        Results.to_csv(project_path + "/results")



    #Vergleich plotten
    print(MAPE_error)
    names = list(model_paths.keys())
    MSE_losses = list(MSE.values())
    MAPE_losses = list(MAPE.values())
    MAPE_errors = list(MAPE_error.values())
    avg_MAPE_losses = list(avg_MAPE.values())
    calc_times = list(calc_times.values())
    training_times = list(training_time.values())
    print(MAPE_losses)
    print(MAPE_errors)
    print(type(MAPE_losses[0]))
    print(type(MAPE_errors[0]))

    x = np.arange(len(names))
    width = 0.7

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #rects1 = ax1.bar(x-width/2, MSE_losses, width, yerr=0.01, label="MSE", color="orange")
    rects3 = ax2.bar(x, avg_MAPE_losses, width/2, yerr=MAPE_errors,
                     capsize=45, label="Avg")
    rects2 = ax2.bar(x, MAPE_losses, width, label="Min")

    ax1.set_title("Validation loss for different " + investigated_parameter)
    ax1.set_ylabel("MSE")
    ax2.set_ylabel("MAPE")
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color="gray")
    ax2.xaxis.grid(True, color="gray")
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax1.set_xticklabels(names)
    fig.legend()
    fig.tight_layout()
    plt.show()

    print(training_time)
    print(training_times)
    print(calc_times)

    width = 0.35

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    rects1 = ax1.bar(x-width/2, training_times, width, label="training times", color="orange")
    rects2 = ax2.bar(x+width/2, calc_times, width, label="calc times")

    ax1.set_title("Validation loss for different " + investigated_parameter)
    ax1.set_ylabel("training time/s")
    ax2.set_ylabel("calc time/s")
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color="gray")
    ax2.xaxis.grid(True, color="gray")
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax1.set_xticklabels(names)
    fig.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
