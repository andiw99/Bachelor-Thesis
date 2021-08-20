import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import time

#Daten einlesen
diff_WQ_theta_data_raw = pd.read_csv("ThetaData")


#Grid erstellen
batch_size_pool=[32, 64]
units_pool = [64, 128]
nr_hidden_layers_pool = [2, 3]
learning_rate_pool=[1e-3, 1e-4, 1e-5]
losses_pool = [tf.keras.losses.MeanSquaredError(), keras.losses.MeanAbsoluteError()]
optimizers_pool = [keras.optimizers.Adam(), keras.optimizers.SGD()]


#Grid unabhängige variablen initialisieren:
training_epochs = 30
total_data = len(diff_WQ_theta_data_raw["Theta"])
l2_kernel = 0
l2_bias = 0
train_frac = 0.8
buffer_size = int(total_data * train_frac)
print("Messwerte:", total_data)
total_loss_fn = keras.losses.MeanSquaredError()

#Daten vorbereiten muss eigetnlich vor dem grid passieren
dataset = diff_WQ_theta_data_raw.copy()
#In Training und Testdaten unterteilen
train_dataset = dataset.sample(frac=train_frac, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#aus den Daten einen Tensor machen
training_data_theta = tf.constant([train_dataset["Theta"]], shape=(len(train_dataset["Theta"]), 1), dtype="float32")
test_data_theta = tf.constant([test_dataset["Theta"]], shape=(len(test_dataset["Theta"]), 1), dtype="float32")

training_data = tf.data.Dataset.from_tensor_slices((training_data_theta, train_dataset["WQ"]))

#Grid punkte initialisieren:
for batch_size in batch_size_pool:
    for units in units_pool:
        for learning_rate in learning_rate_pool:
            for nr_hidden_layers in nr_hidden_layers_pool:
                for optimizers in optimizers_pool:
                    for lossfunction in losses_pool:
                        #Gridvariablen initialisieren
                        batch_size = batch_size
                        units = units
                        learning_rate = learning_rate
                        nr_hidden_layers = nr_hidden_layers
                        time0 = time.time()

                        #Initialisiere Model
                        theta_model = ml.DNN(nr_hidden_layers=nr_hidden_layers, units=units, outputs=1,
                                             kernel_regularization=keras.regularizers.l2(l2=l2_kernel),
                                             bias_regularization=keras.regularizers.l2(l2=l2_bias))
                        optimizer = optimizers
                        loss_fn = lossfunction
                        print("Wir initialisieren ein Model mit: nr_hidden_layers:", nr_hidden_layers, "batch_size:",
                              batch_size, "units:", units, "learning_rate:", learning_rate, "optimizer:", optimizer._name,
                              "Loss-Function:", loss_fn.name)

                        #Batchen muss nach grid passieren weil batch size grid variable ist
                        training_data_batched = training_data.batch(batch_size=batch_size)

                        #Training starten
                        losses = []
                        epochs = []

                        for epoch in range(training_epochs):
                            # losses aktualisieren in jeder epoche aktualisieren!
                            grid_losses_theta = pd.read_csv("grid_losses_theta")
                            best_total_loss = grid_losses_theta["best_total_loss"]
                            epochs.append(epoch)
                            loss = 0
                            for step, (x,y) in enumerate(training_data_batched):
                                loss = theta_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
                            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
                            training_data_batched.shuffle(buffer_size=buffer_size)
                            losses.append(loss)
                            time1 = time.time()
                            time_finished_training = time1 - time0

                            #model testen, aber in jeder epoche    Überprüfen, ob es besser war
                            results = theta_model(test_data_theta, training=False)
                            WQ_data_test = tf.constant([test_dataset["WQ"]], dtype="float32", shape=(len(test_dataset["WQ"]),1))
                            validation_loss = total_loss_fn(results, WQ_data_test)
                            print("validation loss:", float(validation_loss))
                            time2 = time.time()
                            time_finished_testing = time2 - time1


                            #Ergebnisse speichern, nur alle 5 epochen
                            if (epoch+1) % 10 == 0:
                                parameters = pd.DataFrame(
                                    {
                                        "train_frac": [train_frac],
                                        "nr_hidden_layers": [nr_hidden_layers],
                                        "batch_size": [batch_size],
                                        "buffer_size": [buffer_size],
                                        "training_epochs": [epoch],
                                        "learning_rate": [learning_rate],
                                        "units": [units],
                                        "epochs": [epoch],
                                        "validation loss": ["{:.2f}".format(float(validation_loss))],
                                        "training_time":  ["{:.2f}".format(time_finished_training)],
                                        "performance_time": ["{:.5f}".format(time_finished_testing)],
                                        "optimizer": [optimizer._name],
                                        "Loss-Function": [loss_fn.name]
                                    }
                                )
                                name = str()
                                for param in parameters:
                                    name += str(parameters[param][0]) + "__"
                                parameters.to_csv("GridSearchTheta/"+name , index=False)

                            if validation_loss <= best_total_loss:
                                print("Verbesserung erreicht!")
                                print("Verbesserung erreicht!")
                                print("Verbesserung erreicht!")
                                print("Wir haben ein besseres Model mit: nr_hidden_layers", nr_hidden_layers, "batch_size:",
                                      batch_size, "units:", units, "learning_rate:", learning_rate, "training_epochs:", epoch,
                                      "optimizer:", optimizer._name, "Loss-Funktion:", loss_fn.name)

                                theta_model.save(filepath="grid_theta_model")

                                best_losses = pd.DataFrame(
                                    {
                                        "best_total_loss": [float(validation_loss)],
                                    }
                                )
                                best_hyperparameters = pd.DataFrame(
                                    {
                                        "train_frac": [train_frac],
                                        "nr_hidden_layers": [nr_hidden_layers],
                                        "batch_size": [batch_size],
                                        "buffer_size": [buffer_size],
                                        "training_epochs": [epoch],
                                        "learning_rate": [learning_rate],
                                        "units": [units],
                                        "epochs": [epoch],
                                        "optimizer": [optimizer._name],
                                        "Loss-Function": [loss_fn.name]
                                    }
                                )

                                best_losses.to_csv("grid_losses_theta", index=False)
                                best_hyperparameters.to_csv("grid_hyperparameters_theta", index=False)

                                #plotten, falls Verbesserung erreicht
                                plt.plot(test_dataset["Theta"], results)
                                plt.ylabel("WQ")
                                plt.xlabel("pred. data")
                                plt.ylim([0, 50])
                                plt.xlim([0.01, 3.14])
                                plt.show()


#erstmal keine plots würde ich sagen
"""
plt.plot(test_dataset["Theta"], test_dataset["WQ"])
plt.ylabel("WQ")
plt.xlabel("raw data")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()

plt.plot(epochs, losses)
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.show()

plt.plot(test_dataset["Theta"], results)
plt.ylabel("WQ")
plt.xlabel("pred. data")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()
"""
