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
    testing_data_path = "//Files/Transfer/Data/TransferTestData50k/all"
    project_path = "//Files/Partonic/Models/PartonicTheta"
    save_path = project_path + "/RandomSearchTheta/enhanced results"


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

    config_paths = dict()
    for comparison in model_paths:
        config_paths[comparison] = dict()
        for model in model_paths[comparison]:
            config_paths[comparison][model] = model_paths[comparison][model] + "/" + "config"

    avg_MAPE = dict()
    configs = dict()
    for comparison in config_paths:
        configs[comparison] = dict()
        avg_MAPE = pd.DataFrame()
        avg_losses = []

        for i,model in enumerate(config_paths[comparison]):
            configs[comparison][model] = pd.read_csv(
                config_paths[comparison][model],
                index_col="property").transpose()
            avg_MAPE[model] = ["{:.5f}".format(float(
                configs[comparison][model]["avg validation loss"][0]))]
            avg_losses.append(float(
                configs[comparison][model]["avg validation loss"][0]))
    avg_MAPE = avg_MAPE.transpose()
    avg_MAPE.to_csv(save_path)
    for i in range(10):
        print("index mit dem " + str(i+1) + ". niedrigesten loss:", np.argmin(avg_losses)+2, "loss:", np.min(avg_losses))
        avg_losses.pop(np.argmin(avg_losses))

if __name__ == "__main__":
    main()