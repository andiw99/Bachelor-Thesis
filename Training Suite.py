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
#Neues Modell oder weitertrainieren?
new_model = ast.literal_eval(input("new_model= "))

time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/Uniform+Strange+middlex/"
data_name = "all"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Models/"
loss_name = "reweight_loss"
project_name = ""

model_name = "MSE+MoreNodes+MoreLayers"
label_name = "reweight"
path =project_path + project_name + model_name



#data_name_x_constant = "log_neg_x12/x_constant"
#data_name_eta_x_2_constant = "log_neg_x12/eta_x_2_constant"
#data_name_eta_x_1_constant = "log_neg_x12/eta_x_1_constant"
#data_name_x_2_constant = "log_neg_3D/x_2_constant__3D"
data_raw = pd.read_csv(data_path+data_name)
#Überprüfen, ob es für das vorliegende Problem schon losses gibt und ggf einlesen
best_losses = None
if os.path.exists(project_path+ project_name + loss_name):
    best_losses = pd.read_csv(project_path+ project_name + loss_name)


#Variablen...
total_data = len(data_raw[label_name])
train_frac = 0.85
batch_size = 32
buffer_size = int(total_data * train_frac)
training_epochs = 120
nr_layers = 3
units = 256
learning_rate = 0.8e-8
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

#3D-plots zeigen?
show_3D_plots = False

#Loss für validation loss festlegen
loss_function = keras.losses.MeanAbsolutePercentageError()
loss_function_name = loss_function.name
loss_fn_name = loss_fn.name

#Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

#best_total_loss = best_losses
time2= time.time()
print("Zeit nur zum Einlesen von ", total_data, "Punkten:", time2-time1,"s")

#Daten vorbereiten
train_dataset = data_raw.sample(frac=train_frac)
test_dataset = data_raw.drop(train_dataset.index)

#In Features und Labels unterteilen
train_features_pd = train_dataset.copy()
test_features_pd = test_dataset.copy()

train_labels_pd = train_features_pd.pop(label_name)
test_labels_pd = test_features_pd.pop(label_name)

#Aus den Pandas Dataframes tf-Tensoren machen
for i,key in enumerate(train_features_pd):
    print(i)
    if i == 0:
        train_features = tf.constant([train_features_pd[key]], dtype="float32")
    else:
        more_features = tf.constant([train_features_pd[key]], dtype="float32")
        train_features = tf.experimental.numpy.append(train_features, more_features, axis=0)

for i,key in enumerate(test_features_pd):
    if i == 0:
        test_features = tf.constant([test_features_pd[key]], dtype="float32")
    else:
        more_features = tf.constant([test_features_pd[key]], dtype="float32")
        test_features = tf.experimental.numpy.append(test_features, more_features, axis=0)

#Dimensionen arrangieren
train_features = tf.transpose(train_features)
test_features = tf.transpose(test_features)

train_labels = tf.math.abs(tf.transpose(tf.constant([train_labels_pd], dtype="float32")))
test_labels = tf.math.abs(tf.transpose(tf.constant([test_labels_pd], dtype="float32")))

transformer = ml.label_transformation(train_labels, scaling=scaling_bool, logarithm=logarithm, shift=shift, label_normalization=label_normalization)
train_labels = transformer.transform(train_labels)
test_labels = transformer.transform(test_labels)

training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

training_data = training_data.batch(batch_size=batch_size)
time3 = time.time()

print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
if new_model:
    if custom:
        model = ml.DNN2(
             nr_hidden_layers=nr_layers, units=units, outputs=1,
             loss_fn=loss_fn, optimizer=optimizer,
             hidden_activation=hidden_activation, output_activation=output_activation,
             kernel_initializer=kernel_initializer,
             bias_initializer=bias_initializer,
             kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias),
             dropout=dropout, dropout_rate=dropout_rate
        )
    if not custom:
        model = keras.Sequential()
        #Architektur aufbauen
        for i in range(nr_layers):
            if dropout:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.Dense(units=units, activation=hidden_activation, name="Layer_" + str(i),
                                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                                  kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                                  bias_regularizer=keras.regularizers.l2(l2=l2_bias)))
        #Output layer nicht vergessen
        model.add(keras.layers.Dense(units=1, activation=output_activation, name="Output_layer",
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                     kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                     bias_regularizer=keras.regularizers.l2(l2=l2_bias)
                                     ))
        #Model compilen
        model.compile(optimizer=optimizer, loss=loss_fn)

else:
    model = keras.models.load_model(filepath=path)
    model.compile(optimizer=optimizer, loss=loss_fn)


#Training starten
losses = []
steps =[]
epochs = []
total_steps = 0
time4 = time.time()
if new_model and custom:
    for epoch in range(training_epochs):
        epochs.append(epoch)
        results = []
        loss = 0
        #Evaluation in every epoch
        results = tf.math.abs(model(test_features, training=False))
        total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
        print("total loss:", float(total_loss))
        for step, (x,y) in enumerate(training_data.as_numpy_iterator()):
            loss = model.train_on_batch(x=x, y=y)
            total_steps += 1
            if step % (int(total_data/(8*batch_size))) == 0:
                print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
                steps.append(total_steps)
        losses.append(loss)
else:
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2, shuffle=True)
time5 = time.time()
training_time = time5 - time4

#Überprüfen wie gut es war
results = model(test_features)
total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
print("total loss:", float(total_loss))

#Losses plotten
if new_model and custom:
    plt.plot(epochs, losses)
    plt.yscale("log")
    plt.ylabel("Losses")
    plt.xlabel("Epoch")

else:
    plt.plot(history.history["loss"], label="loss")
    plt.ylabel("Losses")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.legend()
plt.savefig(path + "/training_losses")
plt.show()
#Modell und config speichern
model.save(filepath=path, save_format="tf")

if new_model:
    config = pd.DataFrame(model.get_config())
    training_parameters = pd.DataFrame(
        {
            "learning_rate": learning_rate,
            "epochs": training_epochs,
            "batch_size": batch_size,
            "validation loss": "{:.2f}".format(float(total_loss)),
            "scaling": scaling_bool,
            "logarithm": logarithm,
            "transformer_config": str(transformer.get_config()),
            "training time:": "{:.2f}".format(training_time),
            "Custom": custom
        },
        index=[0]
    )
    #config.append(training_parameters)
    config = pd.concat([config, training_parameters], axis=1)
    index=True
if not new_model:
    config = pd.read_csv(path + "/config")
    config = config.transpose()
    i = 0
    while pd.notna(config[2][i]):
        i += 1
    config[2][i] = learning_rate
    index= False

config = config.transpose()
print(config)
config.to_csv(path + "/config", index=index)



#Überprüfen, ob Fortschritt gemacht wurde
results = transformer.retransform(model(test_features))
percentage_loss = keras.losses.MeanAbsolutePercentageError()
validation_loss = percentage_loss(y_true=transformer.retransform(test_labels), y_pred=results)
print("percentage loss:", float(validation_loss))

losses_data = pd.DataFrame(
    {
        "best_percentage_loss": [float(validation_loss)]
    }
)


if best_losses is not None:
    if float(validation_loss) < float(best_losses["best_percentage_loss"]):
        model.save(filepath=project_path + project_name + "best_model", save_format="tf")
        losses_data.to_csv(project_path + project_name + loss_name, index=False)
        config.to_csv(project_path + project_name + "best_config", index = index)
        config.to_csv(project_path + project_name + "best_model/config", index = index)
        print("VERBESSERUNG ERREICHT!")

elif best_losses == None:
    model.save(filepath=project_path + project_name + "best_model", save_format="tf")
    losses_data.to_csv(project_path + project_name + loss_name, index=False)
    config.to_csv(project_path + project_name + "best_model/config", index = index)
    config.to_csv(project_path + project_name + "best_config", index= index)
    print("VERBESSERUNG ERREICHT!")
