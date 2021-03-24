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
hadronic_WQ_data_raw = pd.read_csv("hadronic_WQ_data")
#best_losses = pd.read_csv("hadronic_best_losses")
time2= time.time()
print("Zeit nur zum Einlesen von 4000000 Punkten:", time2-time1,"s")

#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.999
batch_size = 256
buffer_size = int(total_data * train_frac)
training_epochs = 5
units = 128
learning_rate = 3e-3
#best_total_loss = best_losses

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
hadronic_model = Layers.DNN(nr_hidden_layers=5, units=units, outputs=1)
loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Training starten
losses = []
steps =[]
epochs = []

for epoch in range(training_epochs):
    epochs.append(epoch)
    loss = 0
    for step, (x,y) in enumerate(training_data):
        loss += hadronic_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
        if step % 1000 == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(step)
    losses.append(loss)

#Überprüfen wie gut es war
results = hadronic_model(test_features)
total_loss = loss_fn(results, test_labels)
print("total loss:", float(total_loss))

#Losses plotten
plt.plot(epochs, losses)
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.show()


