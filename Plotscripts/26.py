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
    testing_data_paths = dict()
    testing_data_paths["Important Range"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TestData10k_ep_0.163/all"
    testing_data_paths["Full Range"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TrainingData10k_ep_0.01/all"
    testing_data_paths["Full Range + IS"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TrainingData10k_ep_0.01_IS/all"
    project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/Models/PartonicTheta/"
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/"
    comparison = "Training Data"
    #Zu vergleichende models laden
    model_paths = dict()
    #model_paths["best model "] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/Loss_comparison/best_model"
    model_paths["10k"] = project_path + "theta_model_full_range_less_data"
    model_paths["IS\n10k"] = project_path + "theta_model_full_range_IS_less_data"
    model_paths["60k"] = project_path + "theta_model_full_range"
    model_paths["IS\n60k"] = project_path + "theta_model_full_range_IS"

    #festlegen ob modelle an Datenmengen getestet werden oder die Losses aus config ausgelesen werden
    from_config = False

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

    test_features = dict()
    test_labels = dict()
    for dataset in testing_data_paths:
        (_, test_features[dataset], test_labels[dataset], _, _, _) = \
            ml.data_handling(data_path=testing_data_paths[dataset], label_name=label_name,
                         return_pd=False)

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
        MAPE_loss_fn = keras.losses.MeanAbsolutePercentageError()

        for model in models:
            Predictions[model] = dict()
            MAPE[model] = dict()
            calc_times[model] = []

            for dataset in test_features:
                #Berechnung der Labels timen
                time_pre_calc = time.time()
                Predictions[model][dataset] = transformers[model].retransform(models[model].predict(test_features[dataset]))
                time_post_calc = time.time()
                #Zeit auf Berechnung von 1M Datenpunkten normieren
                time_per_million = ((time_post_calc - time_pre_calc)/(float(tf.size(test_labels[dataset])))) * 1e+6
                #In dicts abspeichern
                MAPE[model][dataset] = float((MAPE_loss_fn(y_true= test_labels[dataset], y_pred=Predictions[model][dataset])))
                calc_times[model].append(time_per_million)
            MAPE_error[model] = np.std([*MAPE[model].values()])
            avg_MAPE[model] = np.mean([*MAPE[model].values()])

        #Ergebnisse speichern:
        """
        Results = pd.DataFrame(
            {
                "model": list(models.keys()),
                "MSE": list(MSE.values()),
                "MAPE": list(MAPE.values()),
                "Time per 1M": list(calc_times.values())
            }
        )
        Results.to_csv(project_path + "/results")
        """


    #Vergleich plotten
    print(MAPE_error)
    print(MAPE)
    print(avg_MAPE)
    names = list(model_paths.keys())
    MAPE_losses = list(MAPE.values())
    MAPE_errors = list(MAPE_error.values())
    avg_MAPE_losses = list(avg_MAPE.values())
    calc_times = list(calc_times.values())
    training_times = list(training_time.values())
    print("avg_MAPE_losses", avg_MAPE_losses)
    print("MAPE_losses", MAPE_losses)
    print("MAPE_error", MAPE_errors)
    print(type(MAPE_losses[0]))
    print(type(MAPE_errors[0]))


    ml.make_comparison_plot(names=names, all_losses=MAPE_losses, avg_losses=avg_MAPE_losses,
                             save_path=save_path, comparison=comparison, autoscale=True)
    plt.show()
if __name__ == "__main__":
    main()
