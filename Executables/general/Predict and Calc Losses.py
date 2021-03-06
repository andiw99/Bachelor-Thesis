import os

from tensorflow import keras
import numpy as np
import pandas as pd
import MC
import ml
import time

def main():
    # save_path angeben wo Ergebnisse gespeichert werden sollen
    save_path = "//Results/transfer_speed_test/"
    file_name = "reweight_vs_transfer_rw_analytic"

    # Datensets angeben, für die Dinge predicted werden sollen
    data_paths = dict()
    #data_paths["MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Partonic/PartonicData/TrainingDataEta10k"
    #data_paths["hadronic"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/TrainingData500k/all"
    data_paths["MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Transfer/Data/MMHT TestData50k/all"

    # TODO: label_name angeben und anzeigen ob reweight oder nicht
    label_name = "WQ"
    reweight = True
    reweight_from_source_model = False
    use_x_cut = False
    repeat = 1000
    loss_function = keras.losses.MeanAbsolutePercentageError()

    if reweight:
        source_paths = dict()
        if not reweight_from_source_model:
            source_paths["MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Transfer/Data/CT14 TestData50k/all"
            source_features = dict()
            source_labels = dict()
    # Modelle einlesen die predicten sollen
    model_paths = dict()
    #model_paths["eta_model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Partonic/Models/PartonicEta/best_model"


    """
    model_paths[
        "fastest model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/final comparison/Architecture/(32, 10)"
    model_paths[
        "slowest model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/final comparison/Architecture/(1024, 2)"
    model_paths["transfer"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Transfer/Models/test"
    model_paths["source_model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/best_model"
    model_paths["best guess"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/best_guess_4M"
    model_paths["transfer no FT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Transfer/Models/test_without_fine_tuning"
    #model_paths["transfer plus layer"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Transfer/Models/transferred_search_2/best_model"
    """
    model_paths["reweight"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Reweight/Models/best_parameters_0"
    source_model_paths = dict()
    source_model_paths["source_model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Models/best_model"

    # Daten präparieren
    features = dict()
    labels = dict()
    for dataset in data_paths:
        (features[dataset], labels[dataset]) = MC.data_handling(data_path=data_paths[dataset], label_name=label_name, return_pd=False, return_as_tensor=True)
        print("features", features[dataset], features[dataset].shape)
        if use_x_cut:
            _, x_1_cut = MC.x_cut(features=features[dataset][:,0], lower_cut=0, upper_cut=0.8, return_cut=True)
            features[dataset] = features[dataset][x_1_cut]
            _, x_2_cut = MC.x_cut(features=features[dataset][:,1], lower_cut=0, upper_cut=0.8, return_cut=True)
            features[dataset] = features[dataset][x_2_cut]
            labels[dataset] = labels[dataset][x_1_cut]
            labels[dataset] = labels[dataset][x_2_cut]
        if reweight and not reweight_from_source_model:
            (source_features[dataset], source_labels[dataset]) = MC.data_handling(data_path=source_paths[dataset], label_name=label_name, return_pd=False, return_as_tensor=True)
            if use_x_cut:
                source_features[dataset] = source_features[dataset][x_1_cut]
                source_features[dataset] = source_features[dataset][x_2_cut]
                source_labels[dataset] = source_labels[dataset][x_1_cut]
                source_labels[dataset] = source_labels[dataset][x_2_cut]

    # Modelle laden
    models = dict()
    source_models = dict()
    transformers = dict()
    source_transformers = dict()
    for model in model_paths:
        models[model], transformers[model] = ml.import_model_transformer(model_path=model_paths[model])
    if reweight_from_source_model:
        for model in source_model_paths:
            source_models[model], source_transformers[model] = ml.import_model_transformer(model_path=source_model_paths[model])

    # Predictions berechnen
    predictions = dict()
    losses = dict()
    calc_times = dict()
    for model in models:
        predictions[model] = dict()
        losses[model] = dict()
        calc_times[model] = dict()
        for dataset in features:
            for i in range(repeat):
                if i == 10:
                    time_pre_calc = time.time()
                if reweight:
                    if not reweight_from_source_model:
                        predictions[model][dataset] = \
                            1/(transformers[model].retransform(models[model](features[dataset][:,:2], training=False))) * (
                                source_labels[dataset])
                    else:
                        predictions[model][dataset] = \
                            1/(transformers[model].retransform(models[model](features[dataset][:,:2], training=False))) * (
                                source_transformers["source_model"].retransform(source_models["source_model"](features[dataset], training=False)))
                else:
                    predictions[model][dataset] = transformers[model].retransform(models[model](features[dataset], training=False))
                    #predictions[model][dataset] = models[model](features[dataset], training=False)
            calc_times[model][dataset] = (time.time() - time_pre_calc)/((repeat-10) * len(labels[dataset])) * 1e+6
            # losses berechnen
            losses[model][dataset] = float(loss_function(y_pred=predictions[model][dataset], y_true=labels[dataset]))

    # results dataframe erstellen
    results = pd.DataFrame()
    model_row = list(models.keys()) + ["smallest loss", "loss error", "avg validation loss", "total_losses"]
    results["property"] = model_row
    for dataset in features:
        losses_dataset = list()
        times_dataset = list()
        for model_name in models:
            losses_dataset.append(losses[model_name][dataset])
            times_dataset.append(calc_times[model_name][dataset])
        # smallest, error, avg, total in der reihenfolgen appenden
        smallest_loss = np.min(losses_dataset)
        loss_error = np.std(losses_dataset, ddof=1)
        avg_validation_loss = np.mean(losses_dataset)
        total_losses = losses_dataset
        losses_dataset = losses_dataset + [smallest_loss, loss_error, avg_validation_loss, total_losses]
        times_dataset = times_dataset + [0, 0, 0, np.mean(times_dataset)]
        results[dataset] = losses_dataset
        results[dataset + " time"] = times_dataset

    print(results)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results.to_csv(save_path + file_name, index=False)
    predictions = pd.DataFrame()




if __name__ == "__main__":
    main()