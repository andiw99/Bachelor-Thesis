from tensorflow import keras
import numpy as np
import pandas as pd
import MC
import ml


def main():
    # save_path angeben wo Ergebnisse gespeichert werden sollen
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Results"
    file_name = "statistical deviation reweight model"

    # Datensets angeben, für die Dinge predicted werden sollen
    data_paths = dict()
    data_paths["MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/MMHT TestData50k/all"

    # TODO: label_name angeben und anzeigen ob reweight oder nicht
    label_name = "WQ"
    reweight = True
    reweight_from_source_model = True
    use_x_cut = True
    loss_function = keras.losses.MeanAbsolutePercentageError()

    if reweight:
        source_paths = dict()
        if not reweight_from_source_model:
            source_paths["MMHT"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/CT14 TestData50k/all"
            source_features = dict()
            source_labels = dict()
    # Modelle einlesen die predicten sollen
    model_paths = dict()
    source_model_paths = dict()
    model_paths["reweight_0"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/best_parameters_0"
    model_paths["reweight_1"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/best_parameters_1"
    model_paths["reweight_2"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/best_parameters_2"
    source_model_paths["source_model"] = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/LastRandomSearch/best_model"

    # Daten präparieren
    features = dict()
    labels = dict()
    for dataset in data_paths:
        (features[dataset], labels[dataset]) = MC.data_handling(data_path=data_paths[dataset], label_name=label_name, return_pd=False)
        if use_x_cut:
            _, x_1_cut = MC.x_cut(features=features[dataset][:,0], lower_cut=0, upper_cut=0.8, return_cut=True)
            features[dataset] = features[dataset][x_1_cut]
            _, x_2_cut = MC.x_cut(features=features[dataset][:,1], lower_cut=0, upper_cut=0.8, return_cut=True)
            features[dataset] = features[dataset][x_2_cut]
            labels[dataset] = labels[dataset][x_1_cut]
            labels[dataset] = labels[dataset][x_2_cut]
        if reweight and not reweight_from_source_model:
            (source_features[dataset], source_labels[dataset]) = MC.data_handling(data_path=source_model_paths[dataset], label_name=label_name, return_pd=False)
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
    for model in models:
        predictions[model] = dict()
        losses[model] = dict()
        for dataset in features:
            if reweight:
                if not reweight_from_source_model:
                    predictions[model][dataset] = \
                        1/(transformers[model].retransform(models[model].predict(features[dataset][:,:2]))) * (
                            source_labels[dataset])
                else:
                    source_prediction = source_transformers["source_model"].retransform(source_models["source_model"].predict(features[dataset]))
                    predictions[model][dataset] = \
                        1/(transformers[model].retransform(models[model].predict(features[dataset][:,:2]))) * (
                            source_transformers["source_model"].retransform(source_models["source_model"].predict(features[dataset])))
                    print(source_prediction)
                    print(predictions[model][dataset])
                    print(labels[dataset])
                    print("source loss", float(loss_function(y_pred=source_prediction, y_true=labels[dataset])))
                    print("reweighted loss", float(loss_function(y_pred=predictions[model][dataset], y_true=labels[dataset])))
            else:
                predictions[model][dataset] = transformers[model].retransform(models[model].predict(features[dataset]))
            # losses berechnen
            losses[model][dataset] = float(loss_function(y_pred=predictions[model][dataset], y_true=labels[dataset]))

    # results dataframe erstellen
    results = pd.DataFrame()
    model_row = list(models.keys()) + ["smallest_loss", "loss_error", "avg_validation_loss", "total_losses"]
    results["model"] = model_row
    for dataset in features:
        losses_dataset = list()
        for model_name in models:
            losses_dataset.append(losses[model_name][dataset])
        # smallest, error, avg, total in der reihenfolgen appenden
        smallest_loss = np.min(losses_dataset)
        loss_error = np.std(losses_dataset, ddof=1)
        avg_validation_loss = np.mean(losses_dataset)
        total_losses = [losses_dataset]
        losses_dataset = losses_dataset + [smallest_loss, loss_error, avg_validation_loss, total_losses]
        results[dataset] = losses_dataset

    print(results)

    results.to_csv(save_path + file_name)
    predictions = pd.DataFrame()




if __name__ == "__main__":
    main()