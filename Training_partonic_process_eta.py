import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import Layers

#Daten einlesen
diff_WQ_eta_data_raw = pd.read_csv("diff_WQ_eta_data")
best_losses_eta = pd.read_csv("best_losses_eta")

#Variablen
total_data = len(diff_WQ_eta_data_raw["Eta"])
train_frac = 0.8
batch_size = 64
buffer_size = total_data * train_frac
training_epochs = 10
units = 64
learning_rate  = 3e-5
best_total_loss = best_losses_eta["best_total_loss"]
best_total_loss_squared = best_losses_eta["best_total_loss_squared"]

#Daten vorbereiten
dataset = diff_WQ_eta_data_raw.copy()
#In Training und Testdaten unterteilen
train_dataset = dataset.sample(frac=train_frac, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#aus den Daten einen Tensor machen
training_data_eta = tf.constant([train_dataset["Eta"]], shape=(len(train_dataset["Eta"]), 1), dtype="float32")
test_data_eta = tf.constant([test_dataset["Eta"]], shape=(len(test_dataset["Eta"]), 1), dtype="float32")

training_data = tf.data.Dataset.from_tensor_slices((training_data_eta, train_dataset["WQ"]))
training_data = training_data.batch(batch_size=batch_size)

#initialisiere Model
eta_model = Layers.DNN(nr_hidden_layers=3, units=units, outputs=1)
loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Training starten
losses = []
steps = []
epochs = []

for epoch in range(training_epochs):
    epochs.append(epoch)
    loss=0
    for step, (x,y) in enumerate(training_data):
        loss += eta_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
        if step % 100 == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(step)
    losses.append(loss)

#tanh^2+1 plotten
plt.plot(test_dataset["Eta"], test_dataset["WQ"])
plt.ylabel("WQ")
plt.xlabel("raw data")
plt.show()

#losses plotten
plt.plot(epochs, losses)
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.show()

print(training_data_eta)
print(training_data)
print(test_data_eta)



