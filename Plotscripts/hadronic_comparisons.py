import ast
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
    project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/final comparison"
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/comparisons/"

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
        model_names = sorted([name for name in os.listdir(directorys[comparison])
                       if (os.path.isdir(os.path.join(directorys[comparison], name)))
                        & (name != "best_model")])
        print(model_names)
        try:
            model_names = [int(name) for name in model_names]
            model_names.sort()
            model_names = [str(name) for name in model_names]
        except ValueError:
            pass
        print(model_names)
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
    all_MAPE = dict()
    MAPE_error = dict()
    if from_config:
        configs = dict()
        for comparison in config_paths:
            configs[comparison] = dict()
            MAPE[comparison] = dict()
            avg_MAPE[comparison] = dict()
            MAPE_error[comparison] = dict()
            all_MAPE[comparison] = dict()
            for model in config_paths[comparison]:
                configs[comparison][model] = pd.read_csv(config_paths[comparison][model], index_col="property").transpose()
                MAPE[comparison][model] = float(configs[comparison][model]["smallest loss"][0])
                MAPE_error[comparison][model] = float(configs[comparison][model]["loss error"][0])
                avg_MAPE[comparison][model] = float(configs[comparison][model]["avg validation loss"][0])
                all_MAPE[comparison][model] = np.array(ast.literal_eval(configs[comparison][model]["total_losses"][0]))
                if comparison == "Units per Layer":
                    print(comparison, model)
                    print("mape error", MAPE_error[comparison][model])
                    print("avg mape", avg_MAPE[comparison][model])
                    print("all_mape",all_MAPE[comparison][model])
                while MAPE_error[comparison][model] >= 0.9 * avg_MAPE[comparison][model]:
                    all_MAPE[comparison][model] = all_MAPE[comparison][model][all_MAPE[comparison][model] < avg_MAPE[comparison][model] + MAPE_error[comparison][model]]
                    MAPE_error[comparison][model] = np.std(all_MAPE[comparison][model], ddof=1)
                    avg_MAPE[comparison][model] = np.mean(all_MAPE[comparison][model])
                    if comparison == "Units per Layer":
                        print("cut geschehen")
                        print(comparison, model)
                        print("mape error", MAPE_error[comparison][model])
                        print("avg mape", avg_MAPE[comparison][model])
                        print("all_mape", all_MAPE[comparison][model])

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
    all_MAPE_losses = dict()
    for comparison in model_paths:
        # TODO prints entfernen wenn alles läuft
        # print(MAPE_error[comparison])
        names = list(model_paths[comparison].keys())
        #MSE_losses = list(MSE.values())
        MAPE_losses[comparison] = list(MAPE[comparison].values())
        MAPE_errors[comparison] = list(MAPE_error[comparison].values()) # TODO errors mit 1/N-1 statt 1/N
        avg_MAPE_losses[comparison] = list(avg_MAPE[comparison].values())
        all_MAPE_losses[comparison] = list(all_MAPE[comparison].values())
        ml.make_comparison_plot(names=names, min_losses=MAPE_losses[comparison], all_losses=all_MAPE_losses[comparison],
                                avg_losses=avg_MAPE_losses[comparison], losses_errors=MAPE_errors[comparison], save_path=save_path,
                                comparison=comparison)

        plt.show()


if __name__ == "__main__":
    main()
