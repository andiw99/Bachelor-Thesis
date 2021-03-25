import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import Layers
import time

time1 = time.time()
#Daten einlesen
hadronic_WQ_data_raw = pd.read_csv("less_hadronic_WQ_data")
#best_losses = pd.read_csv("hadronic_best_losses")


#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.999
batch_size = 32
buffer_size = int(total_data * train_frac)
training_epochs = 5
nr_layers = 4
units = 64
learning_rate = 3e-5
l2_kernel = 0.005
l2_bias = 0.005
#best_total_loss = best_losses
time2= time.time()
print("Zeit nur zum Einlesen von ", total_data, "Punkten:", time2-time1,"s")

#Daten vorbereiten
train_dataset = hadronic_WQ_data_raw.sample(frac=train_frac, random_state=0)
test_dataset = hadronic_WQ_data_raw.drop(train_dataset.index)

#In Features und Labels unterteilen
train_features = train_dataset.copy()
test_features = test_dataset.copy()
print(train_features)

train_labels = train_features.pop("WQ")
test_labels = test_features.pop("WQ")

#Aus den Pandas Dataframes tf-Tensoren machen
train_features = tf.constant([train_features["x_1"], train_features["x_2"], train_features["eta"]],  dtype="float32")
test_features = tf.constant([test_features["x_1"], test_features["x_2"], test_features["eta"]],  dtype="float32")
#Dimensionen arrangieren
train_features = tf.transpose(train_features)
test_features = tf.transpose(test_features)

training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
training_data = training_data.batch(batch_size=batch_size)
#testing_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
#testing_data = testing_data.batch(batch_size=batch_size)
time3 = time.time()
print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
hadronic_model = Layers.DNN(nr_hidden_layers=nr_layers, units=units, outputs=1, kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias))
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
print("regularizer:", keras.regularizers.l2(l2=0.01))

#Training starten
losses = []
steps =[]
epochs = []
total_steps = 0
for epoch in range(training_epochs):
    epochs.append(epoch)
    loss = 0
    #Evaluation in every epoch
    results = hadronic_model(test_features)
    total_loss = loss_fn(results, test_labels)
    print("total loss:", float(total_loss))
    for step, (x,y) in enumerate(training_data):
        loss = hadronic_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
        total_steps += 1
        if step % (int(total_data/(8*batch_size))) == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(total_steps)
            losses.append(loss)
    training_data.shuffle(buffer_size=total_data)

#Überprüfen wie gut es war
results = hadronic_model(test_features)
total_loss = loss_fn(results, test_labels)
print("total loss:", float(total_loss))

#Losses plotten
plt.plot(steps, losses)
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.show()

#plot für eta plotten
#plotte Graphen für eta, für x_1=x_2=0.2
x_1 = 0.2
x_2 = x_1
WQ_list = []
eta_list = []
step = 0
for i in range(-200, 200):
    eta = 3*i/200
    x = tf.constant([[x_1, x_2, eta]])
    step += 1
    if step % 50 == 0:
        print("Prediction für", eta, "ist", float(hadronic_model(x)))
    WQ_list.append(float(hadronic_model(x)))
    eta_list.append(eta)

plt.plot(eta_list, WQ_list)
plt.xlabel(r"$\eta$")
plt.ylabel("WQ")
plt.show()

