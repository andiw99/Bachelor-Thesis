from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import ml
import time
import os


def main():
    #Was wird untersucht
    investigated_parameter = "models"
    label_name = "WQ"

    #Pfade eingeben
    testing_data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/TransferTestData50k/all"
    project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/comparisons"
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/hadronic_comparisons/"

    input("save_path geändert und an slash hinten gedacht?")

    directorys = dict()
    for comparison in os.listdir(project_path):
        directorys[comparison] = project_path + "/" + comparison

    #Zu vergleichende models laden, will ich das? reichen mir die pfade?
    model_paths = dict()
    models = dict()
    for comparison in os.listdir(project_path):
        #Für jede comparison muss es ein dict mit den models geben
        model_paths[comparison] = dict()
        models[comparison] = dict()
        #model names laden, nur die directorys
        model_names = [name for name in os.listdir(directorys[comparison])
                       if (os.path.isdir(os.path.join(directorys[comparison], name)))
                        & (name != "best_model")]
        # pfade in dictionarys packen
        for model in model_names:
            model_paths[comparison][model] = directorys[comparison] + "/" + model
            #models[comparison][model] =

    print(model_paths)

    #festlegen ob modelle an Datenmengen getestet werden oder die Losses aus config ausgelesen werden
    from_config = True

    config_paths = dict()
    for comparison in model_paths:
        config_paths[comparison] = dict()
        for model in model_paths[comparison]:
            config_paths[comparison][model] = model_paths[comparison][model] + "/" + "config"

    #Zu testende Kriterien: Berechnungszeit, Trainingszeit(schon gegeben), Validation loss


    MSE = dict()
    MAPE = dict()
    avg_MAPE = dict()
    MAPE_error = dict()
    if from_config:
        configs = dict()
        for comparison in config_paths:
            configs[comparison] = dict()
            MAPE[comparison] = dict()
            avg_MAPE[comparison] = dict()
            MAPE_error[comparison] = dict()
            for model in config_paths[comparison]:
                configs[comparison][model] = pd.read_csv(config_paths[comparison][model], index_col="property").transpose()
                MAPE[comparison][model] = float(configs[comparison][model]["smallest loss"][0])
                MAPE_error[comparison][model] = float(configs[comparison][model]["loss error"][0])
                avg_MAPE[comparison][model] = float(configs[comparison][model]["avg validation loss"][0])
                #MSE[comparison][model] = 0.1


    else:
        # Zu testende Modelle und transformer laden
        models = dict()
        transformers = dict()
        for model in model_paths:
            models[model], transformers[model] = ml.import_model_transformer(
                model_paths[model])

        (
        _, test_features, test_labels, _, _, test_features_pd, test_labels_pd, _) = \
            ml.data_handling(data_path=testing_data_path, label_name=label_name,
                             return_pd=True)

        # Validation Loss berechnen, am besten MSE und MAPE
        MSE_loss_fn = keras.losses.MeanSquaredError()
        MAPE_loss_fn = keras.losses.MeanAbsolutePercentageError()

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
            #MSE[model] = float(MSE_loss_fn(y_true= test_labels, y_pred= Predictions[model]))
            MAPE[model] = float((MAPE_loss_fn(y_true= test_labels, y_pred=Predictions[model])))
            Calc_times[model] = time_per_million
            # Nicht berechenbare dictionarys mit nullen füllen
            MAPE_error[model] = 0
            avg_MAPE[model] = MAPE[model]

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
    #für jede comparison ein plot
    MAPE_losses = dict()
    MAPE_errors = dict()
    avg_MAPE_losses = dict()
    for comparison in model_paths:
        print(MAPE_error[comparison])
        names = list(model_paths[comparison].keys())
        #MSE_losses = list(MSE.values())
        MAPE_losses[comparison] = list(MAPE[comparison].values())
        MAPE_errors[comparison] = list(MAPE_error[comparison].values())
        avg_MAPE_losses[comparison] = list(avg_MAPE[comparison].values())
        print(MAPE_losses[comparison])
        print(MAPE_errors[comparison])
        print(type(MAPE_losses[comparison][0]))
        print(type(MAPE_errors[comparison][0]))

        x = np.arange(len(names))
        width = 0.7

        fig, ax = plt.subplots()
        #rects1 = ax.bar(x-width/2, MSE_losses, width, yerr=0.01, label="MSE", color="orange")
        rects3 = ax.bar(x, avg_MAPE_losses[comparison], width/1.5, yerr=MAPE_errors[comparison],
                         capsize=50/len(names), label="Avg")
        rects2 = ax.bar(x, MAPE_losses[comparison], width, label="Min", alpha=0.75)

        ax.set_title("Validation loss for different " + str(comparison))
        ax.set_ylabel("MAPE")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="lightgray")
        #ax.xaxis.grid(True, color="gray")
        ax.set_xticks(x)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        fig.legend()
        fig.tight_layout()
        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path + str(comparison) + "_comparison")
        plt.show()


if __name__ == "__main__":
    main()
