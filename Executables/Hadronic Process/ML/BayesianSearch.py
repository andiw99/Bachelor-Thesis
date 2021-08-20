from tensorflow import keras
import tensorflow as tf
import kerastuner
import sys

# location feststellen um ml zu importieren
location = input("Auf welchem Rechner?")
root_path = "//"
if location == "Taurus" or location == "taurus":
    root_path = "/home/s1388135/Bachelor-Thesis/"
sys.path.insert(0, root_path)
import ml


def build_model(hp):
    # Grid erstellen, pool für jeden Hyperparameter, so kann man dynamische einstellen in welchen Dimensionen das Grid liegt
    # wichtig: Standardmodell hat index 0 in jedem pool
    pools = dict()
    pools["units"] = [32, 64, 128, 256, 512]
    pools["nr_layers"] = [1,2,3,4]
    pools["learning_rate"] = [1e-3, 1e-2, 1e-4, 5e-3, 1e-5]
    pools["l2_kernel"] = [0.0, 0.01, 0.0001]
    pools["l2_bias"] = [0.0, 0.01, 0.0001]
    pools["loss_fn"] = ["mean_absolute_error",
                        "mean_squared_error", "huber_loss"]
    pools["optimizer"] = ["Adam", "RMSprop",
                          "SGD"]
    pools["kernel_initializer"] = ["he_normal", "random_normal"]
    pools["bias_initializer"] = ["zeros"]
    pools["hidden_activation"] = ["relu", "elu",
                                  "tanh", "sigmoid"]
    pools["feature_normalization"] = [True, False]
    # Festlegen, welche Hyperparameter in der Bezeichnung stehen sollen:

    nesterov = True
    new_model = True
    custom = False

    params = dict()
    for pool in pools:
        params[pool] = hp.Choice(pool, pools[pool])

    model = ml.initialize_model(nr_layers=params["nr_layers"],
                                units=params["units"],
                                loss_fn=params["loss_fn"],
                                optimizer=params["optimizer"],
                                hidden_activation=params["hidden_activation"],
                                output_activation=ml.LinearActiavtion(),
                                kernel_initializer=params["kernel_initializer"],
                                bias_initializer=params["bias_initializer"],
                                l2_kernel=params["l2_kernel"],
                                learning_rate=params["learning_rate"],
                                momentum=0.1, nesterov=nesterov,
                                l2_bias=params["l2_bias"],
                                dropout=False,
                                dropout_rate=0,
                                new_model=new_model, custom=custom,
                                feature_normalization=params[
                                    "feature_normalization"],
                                metrics=["mean_absolute_percentage_error"])
    return model

def main():

    #Parameter für tuner und search festlegen
    max_trials = 100
    epochs = 100
    batch_size = 512
    executions_per_trial = 2
    min_lr = 1e-7
    min_delta = 3e-6
    lr_reduction = 0.05
    lr_factor = 0.5
    lr_patience = 1
    stopping_patience = 3
    save_path = root_path + "Files/Hadronic/Models/"
    data_path = root_path + "/Files/Hadronic/Data/TrainingData4M"
    label_name = "WQ"
    scaling_bool = True
    logarithm = True
    base10 = True
    label_normalization = True

    validation_total = 15000

    tuner = kerastuner.BayesianOptimization(build_model, objective="mean_absolute_percentage_error", max_trials=max_trials,
                                            executions_per_trial=executions_per_trial, directory=save_path, project_name="BayesianSearch4")

    # Daten einlesen
    (training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(
        data_path=data_path + "/all", label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm,
        base10=base10, label_normalization=label_normalization, validation_total=validation_total)

    #Callbacks initialisieren
    #min delta initialiseren
    reduce_lr = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=lr_reduction, min_lr=min_lr))
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience, min_delta=min_delta, min_lr=min_lr)
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * min_delta, patience=stopping_patience)
    callbacks = [reduce_lr_on_plateau, early_stopping, reduce_lr]

    # Search beginnen
    tuner.search(train_features, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=(test_features, test_labels))


if __name__ == "__main__":
    main()
