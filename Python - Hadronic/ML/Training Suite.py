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
        input("auf die config geguckt?")


time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/NewRandom/"
data_name = "all"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Models/"
loss_name = "Source_loss"
project_name = ""

read_name = "Feature-Normalization+MAE+SmallBatch"
label_name = "WQ"
read_path =project_path + project_name + read_name
if transfer:
    save_name = "transferred_model"
    save_path = project_path + project_name + save_name
else:
    save_path = read_path



#Überprüfen, ob es für das vorliegende Problem schon losses gibt und ggf einlesen
best_losses = None
if os.path.exists(project_path+ project_name + loss_name):
    best_losses = pd.read_csv(project_path+ project_name + loss_name)


#Variablen...
train_frac = 0.95
batch_size = 64
training_epochs = 100
nr_layers = 2
units = 512
learning_rate = 1e-2
rm_layers = 1
loss_fn = keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
hidden_activation = tf.nn.relu
output_activation = ml.LinearActiavtion()
kernel_initializer = tf.keras.initializers.HeNormal()
bias_initializer = tf.keras.initializers.Zeros()
l2_kernel = 0
l2_bias = 0
lr_patience = 1
stopping_patience = 3 * lr_patience
min_delta= 3e-6
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=lr_patience,
                                                         min_delta=min_delta, min_lr=1e-7)
early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-1 * min_delta, patience=stopping_patience)
lr_schedule = keras.callbacks.LearningRateScheduler(ml.class_scheduler(reduction=0.05, min_lr=1e-7))
callbacks = [reduce_lr_on_plateau, early_stopping, lr_schedule]
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = True
base10 = True
shift = False
label_normalization = True
feature_normalization = True
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
loss_function_name = loss_function.name
loss_fn_name = loss_fn.name

#Verzeichnis erstellen
if not os.path.exists(path=save_path):
    os.makedirs(save_path)

#best_total_loss = best_losses
time2= time.time()
(training_data, train_features, train_labels, test_features, test_labels, transformer) =\
        ml.data_handling(data_path=data_path+data_name,
        train_frac=train_frac,batch_size=batch_size,
        label_name=label_name, scaling_bool=scaling_bool, logarithm=logarithm,
        shift=shift, label_normalization=label_normalization, feature_rescaling=feature_rescaling)
time3 = time.time()

print("min ", tf.reduce_min(train_labels))
print("max ", tf.reduce_max(train_labels))
print("mean ", tf.reduce_mean(train_labels))
print("stddev ", tf.math.reduce_std(train_labels))


for i in range(train_features.shape[1]):
    print("min ", tf.reduce_min(train_features[:,i]))
    print("max ", tf.reduce_max(train_features[:,i]))
    print("mean ", tf.reduce_mean(train_features[:,i]))
    print("stddev ", tf.math.reduce_std(train_features[:,i]))



print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
model = ml.initialize_model(nr_layers=nr_layers, units=units, loss_fn=loss_fn, optimizer=optimizer, hidden_activation=hidden_activation, output_activation=output_activation,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, l2_kernel=l2_kernel, l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                            new_model=new_model, custom=custom, transfer=transfer, rm_layers=1, read_path=read_path, freeze=freeze, feature_normalization=feature_normalization)

#Training starten
time4 = time.time()
history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2,
                    callbacks=callbacks, shuffle=True)
time5 = time.time()
training_time = time5 - time4

#Überprüfen wie gut es war
results = model(test_features)
total_loss = loss_function(y_true=transformer.retransform(test_labels),y_pred=transformer.retransform(results) )
print("total loss:", float(total_loss))

#Losses plotten
ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
plt.savefig(save_path + "/training_losses")
plt.show()

#Modell und config speichern
model.save(filepath=save_path, save_format="tf")
(config, index) = ml.save_config(new_model=new_model, model=model, learning_rate=learning_rate, training_epochs=training_epochs,
               batch_size=batch_size, total_loss=total_loss, transformer=transformer,
               training_time=training_time, custom=custom, loss_fn=loss_fn, feature_rescaling=feature_rescaling, read_path=read_path, save_path=save_path)



#Überprüfen, ob Fortschritt gemacht wurde
ml.check_progress(model=model, transformer=transformer, test_features=test_features, test_labels=test_labels, best_losses=best_losses,
                  project_path=project_path, project_name=project_name, index=index, config=config, loss_name=loss_name)

