import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import ml

#Daten einlesen
diff_WQ_theta_data_raw = pd.read_csv("ThetaData")
best_losses_theta = pd.read_csv("best_losses_theta")

#Variablen
total_data = len(diff_WQ_theta_data_raw["Theta"])
print("Messwerte:", total_data)
train_frac = 0.8
batch_size = 64
buffer_size = int(total_data * train_frac)
training_epochs = 100
units = 64
learning_rate  = 3e-4
nr_hidden_layers = 3
l2_kernel = 0.01
l2_bias = 0.01
best_total_loss = best_losses_theta["best_total_loss"]
best_total_loss_squared = best_losses_theta["best_total_loss_squared"]

#Daten vorbereiten
dataset = diff_WQ_theta_data_raw.copy()
#In Training und Testdaten unterteilen
train_dataset = dataset.sample(frac=train_frac, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#aus den Daten einen Tensor machen
training_data_theta = tf.constant([train_dataset["Theta"]], shape=(len(train_dataset["Theta"]), 1), dtype="float32")
test_data_theta = tf.constant([test_dataset["Theta"]], shape=(len(test_dataset["Theta"]), 1), dtype="float32")

training_data = tf.data.Dataset.from_tensor_slices((training_data_theta, train_dataset["WQ"]))
training_data = training_data.batch(batch_size=batch_size)

#initialisiere Model
theta_model = ml.DNN(nr_hidden_layers=nr_hidden_layers, units=units, outputs=1, kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias))
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Training starten
losses = []
steps = []
epochs = []

for epoch in range(training_epochs):
    epochs.append(epoch)
    loss = 0
    for step, (x,y) in enumerate(training_data):
        loss += theta_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
        if step % 100 == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(step)
    training_data.shuffle(buffer_size=buffer_size)
    losses.append(loss)


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

#model testen
results = theta_model(test_data_theta)
print(results)
plt.plot(test_dataset["Theta"], results)
plt.ylabel("WQ")
plt.xlabel("pred. data")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()

#Überprüfen, ob es besser war
WQ_data_test = tf.constant([test_dataset["WQ"]], dtype="float32", shape=(len(test_dataset["WQ"]),1))
print("WQ_data_test:", WQ_data_test)
total_loss = loss_fn(results, WQ_data_test)
total_loss_squared = tf.reduce_sum(tf.square(tf.add(results, -WQ_data_test)))
print("total loss:", float(total_loss))
print("total square loss:", float(total_loss_squared))

if total_loss <= best_total_loss and total_loss_squared <= best_total_loss_squared:
    print("Verbesserung erreicht!")
    print("Verbesserung erreicht!")
    print("Verbesserung erreicht!")

    theta_model.save(filepath="theta_model")

    best_losses = pd.DataFrame(
        {
            "best_total_loss": [float(total_loss)],
            "best_total_loss_squared": [float(total_loss_squared)]
        }
    )
    best_parameters = pd.DataFrame(
        {
            "train_frac": [train_frac],
            "batch_size": [batch_size],
            "buffer_size": [buffer_size],
            "training_epochs": [training_epochs],
            "learning_rate": [learning_rate],
            "units": [units]
        }
    )

    best_losses.to_csv("best_losses_theta", index=False)
    best_parameters.to_csv("best_parameters_theta", index=False)




