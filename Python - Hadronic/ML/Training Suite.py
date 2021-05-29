import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
import ml
import time
import os
import ast


def main():
    debugging = False
    transfer = False
    freeze = False
    if debugging:
        new_model = False
        transfer = True

    #Neues Modell oder weitertrainieren?
    else:
        new_model = ast.literal_eval(input("new_model= "))
        if not new_model:
            transfer = ast.literal_eval(input("transfer= "))
            if not transfer:
                freeze = ast.literal_eval((input("freeze= ")))


    time1 = time.time()
    #Daten einlesen
    data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/TrainingData4M/"
    data_name = "all"
    project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/"
    loss_name = "Source_loss"
    project_name = ""

    read_name = "scaling_theta_config_2" #"best_guess_important_range" #
    label_name = "WQ"
    read_path =project_path + project_name + read_name
    transferred_transformer = None
    add_layers = 0
    if transfer:
        save_name = "transferred_model_2M_new_layer"
        save_path = project_path + project_name + save_name
        add_layers = int(input("add layers: "))
    else:
        save_path = read_path
    if not new_model:
        (_, transferred_transformer) = ml.import_model_transformer(model_path=read_path)


    #Überprüfen, ob es für das vorliegende Problem schon losses gibt und ggf einlesen
    best_losses = None
    if os.path.exists(project_path+ project_name + loss_name):
        best_losses = pd.read_csv(project_path+ project_name + loss_name)


    #Variablen...
    train_frac = 0.95
    batch_size = 256
    training_epochs = 100
    repeat= 1
    nr_layers = 4
    units = 128
    learning_rate = 5e-3
    if not any([new_model, transfer, freeze]):
        learning_rate = 5e-6
        print("learning_rate für fine tuning reduziert!")
    rm_layers = 1
    loss_fn = keras.losses.MeanAbsoluteError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10)
    print("optimizer")
    hidden_activation = tf.nn.relu
    output_activation = ml.LinearActiavtion()
    kernel_initializer = tf.keras.initializers.HeNormal()
    bias_initializer = tf.keras.initializers.Zeros()
    l2_kernel = 0
    l2_bias = 0
    lr_patience = 1
    stopping_patience = 3 * lr_patience
    min_delta= 2e-10
    min_lr = 5e-8
    dropout = False
    dropout_rate = 0.1
    scaling_bool = True
    logarithm = False
    base10 = False
    shift = False
    label_normalization = False
    feature_normalization = False
    feature_rescaling= False

    #ggf Losses einlesen
    if best_losses is not None:
        best_percentage_loss = best_losses["best_percentage_loss"]


    #Custom Layers oder Keras Layers?
    custom = False

    #3D-plots zeigen?
    show_3D_plots = False

    #Loss für validation loss festlegen
    loss_function = keras.losses.MeanAbsolutePercentageError()

    #Verzeichnis erstellen
    if not os.path.exists(path=save_path):
        os.makedirs(save_path)

    #best_total_loss = best_losses
    time2= time.time()
    (training_data, train_features, train_labels, test_features, test_labels, transformer) =\
            ml.data_handling(data_path=data_path+data_name,
            train_frac=train_frac,batch_size=batch_size,
            label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm, base10=base10,
            shift=shift, label_normalization=label_normalization, feature_rescaling=feature_rescaling,
                             transformer=transferred_transformer)
    time3 = time.time()
    print("Zeit, um Daten vorzubereiten:", time3-time1)
    models = []
    training_time = 0
    total_losses = []
    for i in range(repeat):
        #initialisiere Model
        models.append(ml.initialize_model(nr_layers=nr_layers, units=units, loss_fn=loss_fn, optimizer=optimizer, learning_rate=learning_rate, hidden_activation=hidden_activation, output_activation=output_activation,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, l2_kernel=l2_kernel, l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                                    new_model=new_model, custom=custom, transfer=transfer, rm_layers=1, read_path=read_path, freeze=freeze, feature_normalization=feature_normalization, add_layers=add_layers)
                      )
    for i, model in enumerate(models):
        #callbacks initialisieren
        reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=lr_patience,
                                                                 min_delta=min_delta, min_lr=min_lr)
        early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * min_delta,
                                                       patience=stopping_patience)
        lr_schedule = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=0.05, min_lr=min_lr))
        callbacks = [reduce_lr_on_plateau, early_stopping, lr_schedule]
        #Training starten
        time4 = time.time()
        history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2,
                            callbacks=callbacks, shuffle=True)
        time5 = time.time()
        training_time += time5 - time4

        #Überprüfen wie gut es war
        results = model(test_features)
        loss = loss_function(y_true=transformer.retransform(test_labels), y_pred=transformer.retransform(results))
        total_losses.append(float(loss))
        print("total loss von Durchgang Nr. ", i, ":", float(loss))

        #Losses plotten
        ml.make_losses_plot(history=history)
        plt.savefig(save_path + "/training_losses")
        plt.show()

    print("Losses of the specific cycle:", total_losses)
    print("average Loss over", repeat, "cycles:", np.mean(total_losses))

    #Modell und config speichern
    print("Das beste Modell (Modell Nr.", np.argmin(total_losses), ") wird gespeichert")
    model = models[np.argmin(total_losses)]
    avg_total_loss = np.mean(total_losses)
    smallest_loss = np.min(total_losses)
    loss_error = np.std(total_losses)
    print("avg_total_loss", avg_total_loss, "smallest_loss:", smallest_loss, "loss_error:", loss_error, "total_losses", total_losses)
    training_time = 1/repeat * training_time
    model.save(filepath=save_path, save_format="tf")
    (config, index) = ml.save_config(new_model=new_model, model=model, learning_rate=learning_rate, training_epochs=training_epochs,
                                     batch_size=batch_size, avg_total_Loss=avg_total_loss, transformer=transformer,
                                     training_time=training_time, custom=custom, loss_fn=loss_fn, smallest_loss=smallest_loss,
                                     loss_error=loss_error, total_losses=total_losses, read_path=read_path, save_path=save_path)



    #Überprüfen, ob Fortschritt gemacht wurde
    ml.check_progress(model=model, transformer=transformer, test_features=test_features, test_labels=test_labels, best_losses=best_losses,
                      project_path=project_path, project_name=project_name, index=index, config=config, loss_name=loss_name)


if __name__ == "__main__":
    main()
