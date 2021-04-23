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
    new_model =False
    transfer= True
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
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Uniform+Strange+middlex/"
data_name = "all"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/"
loss_name = "reweight_loss"
project_name = ""

read_name = "test_model"
label_name = "reweight"
read_path =project_path + project_name + read_name
if transfer:
    save_name = "transferred_model"
    save_path = project_path + project_name + save_name
else:
    save_path = read_path


best_losses = None
if os.path.exists(project_path+ project_name + loss_name):
    best_losses = pd.read_csv(project_path+ project_name + loss_name)


#Variablen...
train_frac = 0.85
batch_size = 32
training_epochs = 2
nr_layers = 3
units = 128
learning_rate = 3e-6
rm_layers = 1
loss_fn = keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1)
hidden_activation = tf.nn.leaky_relu
output_activation = ml.LinearActiavtion()
kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.1)
bias_initializer =tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.05)
l2_kernel = 0
l2_bias = 0
dropout = False
dropout_rate = 0.1
scaling_bool = False
logarithm = False
shift = False
label_normalization = False

#ggf Losses einlesen
if best_losses is not None:
    best_percentage_loss = best_losses["best_percentage_loss"]


#Custom Layers oder Keras Layers?
custom = False


#Loss für validation loss festlegen
loss_function = keras.losses.MeanAbsolutePercentageError()
loss_function_name = loss_function.name
loss_fn_name = loss_fn.name

#Verzeichnis erstellen
if not os.path.exists(path=read_path):
    os.makedirs(read_path)

#best_total_loss = best_losses
time2= time.time()
(training_data, train_features, train_labels, test_features, test_labels, transformer) = ml.data_handling(data_path=data_path+data_name,
                                                                                                          train_frac=train_frac,batch_size=batch_size,
                                                                                                          label_name=label_name)
time3 = time.time()

print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
model = ml.initialize_model(nr_layers=nr_layers, units=units, loss_fn=loss_fn, optimizer=optimizer, hidden_activation=hidden_activation, output_activation=output_activation,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, l2_kernel=l2_kernel, l2_bias=l2_bias, dropout=dropout, dropout_rate=dropout_rate,
                            new_model=new_model, custom=custom, transfer=transfer, rm_layers=1, read_path=read_path, freeze=freeze)

#Training starten
time4 = time.time()
history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2, shuffle=True)
time5 = time.time()
training_time = time5 - time4

#Überprüfen wie gut es war
results = model(test_features)
total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
print("total loss:", float(total_loss))

#Losses plotten
ml.make_losses_plot(history=history, custom=custom, new_model=new_model)
plt.savefig(save_path + "/training_losses")
plt.show()

#Modell und config speichern
model.save(filepath=save_path, save_format="tf")
(config, index) = ml.save_config(new_model=new_model, model=model, learning_rate=learning_rate, training_epochs=training_epochs,
                                 batch_size=batch_size, avg_total_Loss=total_loss, scaling_bool=scaling_bool, logarithm=logarithm, transformer=transformer,
                                 training_time=training_time, custom=custom, loss_fn=loss_fn, read_path=read_path, save_path=save_path)



#Überprüfen, ob Fortschritt gemacht wurde
ml.check_progress(model=model, transformer=transformer, test_features=test_features, test_labels=test_labels, best_losses=best_losses,
                  project_path=project_path, project_name=project_name, index=index, config=config, loss_name=loss_name)
