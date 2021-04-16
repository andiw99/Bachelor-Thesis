import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import time
import os


#Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
pools = dict()
pools["batch_size"] = [256]
pools["units"] = [512]
pools["nr_layers"] =  [4]
pools["learning_rate"]= [1e-3]
pools["l2_kernel"] = [0.0]
pools["l2_bias"] = [0.0]
pools["loss_fn"] = [keras.losses.MeanAbsoluteError(), keras.losses.MeanSquaredError(), keras.losses.MeanSquaredLogarithmicError()]
pools["optimizer"] = [keras.optimizers.Adam]
pools["momentum"] = [0.1]
pools["dropout"] = [False]
pools["dropout_rate"] = [0]
pools["kernel_initializer"] = [tf.keras.initializers.HeNormal()]
pools["bias_initializer"] = [tf.keras.initializers.Zeros()]
pools["hidden_activation"] = [tf.nn.relu]
pools["output_activation"] = [ml.LinearActiavtion()]
pools["feature_normalization"] = [None]
#Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:
names = {"loss_fn"  }

time1 = time.time()
#Daten einlesen
location = input("Auf welchem Rechner?")
root_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/"
if location == "Taurus" or location == "taurus":
    root_path = "/home/s1388135/Bachelor-Thesis/"
data_path = root_path + "/Files/Transfer/Data/NewRandom/"
data_name = "all"
project_path = root_path + "Files/Hadronic/HadronicModels/Loss_comparison/"
loss_name = "best_loss"
project_name = ""

label_name = "WQ"

#Variablen...
train_frac = 0.95
training_epochs = 70
size = 3
output_activation = ml.LinearActiavtion()
bias_initializer =tf.keras.initializers.Zeros()
l2_bias = 0
momentum = 0.1
min_lr = 1e-7
nesterov = True
loss_function = keras.losses.MeanAbsolutePercentageError()
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = True
base10 = True
shift = False
label_normalization = True
feature_normalization = False
feature_rescaling = False

custom = False
new_model=True

lr_patience = 1
stopping_patience = 3



#Menge mit bereits gesehen konfigurationen
checked_configs = ml.create_param_configs(pools=pools, size=size)
results_list = dict()

for config in checked_configs:
    #Schönere accessability
    params = dict()
    for i,param in enumerate(pools):
        params[param] = config[i]

    if params["feature_normalization"] == "rescaling":
        feature_rescaling = True
    elif params["feature_normalization"] == "normalization":
        feature_normalization = True

    #training_epochs = int(1/200 * params["batch_size"]) + 10

    #Callbacks initialisieren
    #min delta initialiseren
    #Fall dass msle benutzt werden soll behandeln, label normalization auf [-1,1] verhindern
    actual_label_normalization = label_normalization
    if params["loss_fn"].name == "mean_absolute_error":
        min_delta = 1e-4
        label_normalization = actual_label_normalization
    elif params["loss_fn"].name == "mean_squared_error":
        min_delta = 1e-5
        label_normalization = actual_label_normalization
    elif params["loss_fn"].name == "mean_squared_logarithmic_error":
        min_delta = 1e-6
        label_normalization = False
    reduce_lr = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=0.05, min_lr=min_lr))
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=lr_patience, min_delta=min_delta, min_lr=min_lr)
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * min_delta, patience=stopping_patience)
    callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

    # Daten einlsen
    # Daten einlesen:
    (training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(
        data_path=data_path + data_name, label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm, base10=base10,
        shift=shift, label_normalization=label_normalization, feature_rescaling=feature_rescaling,
        train_frac=train_frac)

    print(train_features)
    print(train_labels)
    exit()

    #Create path to save model
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

    #Modell initialisieren
    model = ml.initialize_model(nr_layers=params["nr_layers"], units=params["units"], loss_fn=params["loss_fn"], optimizer=params["optimizer"],
                                        hidden_activation=params["hidden_activation"], output_activation=output_activation,
                                        kernel_initializer=params["kernel_initializer"], bias_initializer=bias_initializer, l2_kernel=params["l2_kernel"],
                                        learning_rate=params["learning_rate"], momentum=momentum, nesterov=nesterov,
                                        l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                                        new_model=new_model, custom=custom, feature_normalization=feature_normalization)

    # Training starten
    time4 = time.time()
    history = model.fit(x=train_features, y=train_labels, batch_size=params["batch_size"], epochs=training_epochs,
                        callbacks = callbacks, verbose=2, shuffle=True)
    time5 = time.time()
    training_time = time5 - time4

    # Losses plotten
    ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
    plt.savefig(save_path + "/training_losses")
    plt.show()

    # Überprüfen wie gut es war
    results = model(test_features)
    total_loss = loss_function(y_pred=transformer.retransform(results), y_true=transformer.retransform(test_labels))
    print("total loss:", float(total_loss))

    # Modell und config speichern
    model.save(filepath=save_path, save_format="tf")
    (config, index) = ml.save_config(new_model=new_model, model=model, learning_rate=params["learning_rate"],
                                     training_epochs=training_epochs, batch_size=params["batch_size"],
                                     total_loss=total_loss, transformer=transformer, training_time=training_time,
                                     custom=custom, loss_fn=params["loss_fn"], save_path=save_path)

    #Überprüfen ob Fortschritt gemacht wurde
    ml.check_progress(model=model, transformer=transformer, test_features=test_features, test_labels=test_labels,
                      best_losses=best_losses, project_path=project_path, project_name=project_name,
                      index=index, config=config, loss_name=loss_name)

    #Ergebnis im dict festhalten
    results_list[model_name] = "{:.2f}".format(float(total_loss))

    #Ergebnisse speichern
    results_list_pd = pd.DataFrame(
        results_list,
        index = [0]
    )
    results_list_pd = results_list_pd.transpose()
    results_list_pd.to_csv(project_path + "results")

