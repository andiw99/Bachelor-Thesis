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
hadronic_WQ_data_raw = pd.read_csv("HadronicData/lower_hadronic_WQ_data")
#best_losses = pd.read_csv("hadronic_best_losses")


#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.99
batch_size = 64
buffer_size = int(total_data * train_frac)
training_epochs = 5
nr_layers = 8
units = 128
learning_rate = 1e-6
l2_kernel = 0.00001
l2_bias = 0.00001
dropout = False
dropout_rate = 0.1

loss_function = keras.losses.MeanSquaredError()

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

train_labels = tf.transpose(tf.constant([train_labels], dtype="float32"))
test_labels = tf.transpose(tf.constant([test_labels], dtype="float32"))
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))
#Das mit dem Logarithmus probieren, weil das ja irgendwie geholfen hatte
#train_labels = train_labels+1
#test_labels = test_labels+1
scaling = 1/tf.math.reduce_min(train_labels)
print("scaling:", scaling)
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))
train_labels = tf.math.log(scaling * train_labels)
test_labels = tf.math.log(scaling * test_labels)
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))

training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

"""
print("training_data:", training_data)
for step, (x,y) in enumerate(training_data):
    if step % 150000 == 0:
        print("step:", step, "x:", x , "y:", y)
        if step % 200000 == 0 and step != 0:
            break
"""
training_data = training_data.batch(batch_size=batch_size)
#testing_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
#testing_data = testing_data.batch(batch_size=batch_size)
"""
for step, (x,y) in enumerate(training_data):
    if step % 1000 == 0:
        print("step:", step, "x:", x , "y:", y)
"""

time3 = time.time()
print("train_features:", train_features)
print("train_labels:", train_labels)
print("training_data:", training_data)

print("Zeit, um Daten vorzubereiten:", time3-time1)

"""
#Plotten
eta_list  = []
WQ_list_eta = []
for i, features in enumerate(train_features):
    if features[0] == 0.2 and features[1] == 0.2:
        eta_list.append(float(features[2]))
        WQ_list_eta.append(float(train_labels[i]))

print("eta_list:", eta_list)
print("WQ_list_eta:", WQ_list_eta)
plt.plot(eta_list, WQ_list_eta)
plt.show()
exit()
"""

#initialisiere Model
hadronic_model = Layers.DNN(nr_hidden_layers=nr_layers, units=units, outputs=1,
                            kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias),
                            dropout=dropout, dropout_rate=dropout_rate)
loss_fn = keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


#Training starten
losses = []
steps =[]
epochs = []
total_steps = 0
for epoch in range(training_epochs):
    epochs.append(epoch)
    results = []
    loss = 0
    #Evaluation in every epoch
    results = hadronic_model(test_features, training=False)
    total_loss = loss_function(results, test_labels)
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
total_loss = loss_function(results, test_labels)
print("total loss:", float(total_loss))

#Losses plotten
plt.plot(steps, losses)
plt.yscale("log")
plt.ylabel("Losses")
plt.xlabel("Step")
plt.show()

#plot für eta plotten
#plotte Graphen für eta, für x_1=x_2=0.2
"""
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
    WQ_list.append(np.exp(float(hadronic_model(x, training=False))))
    eta_list.append(eta)
"""

#Daten einlesen
hadronic_data_x_constant = pd.read_csv("HadronicData/hadronic__x_constant__0.2")
hadronic_data_eta_constant = pd.read_csv("HadronicData/hadronic__eta_constant__1.0133779264214047")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
predictions_x_constant =(1/scaling)* np.exp(hadronic_model(pred_feature_x_constant))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["x_2"], hadronic_data_eta_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
predictions_eta_constant = (1/scaling) *  np.exp(hadronic_model(pred_feature_eta_constant))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    losses_x_constant.append(float(loss_function(1/scaling * np.exp(hadronic_model(feature)), hadronic_data_x_constant["WQ"][i])))

for i, feature in enumerate(pred_feature_eta_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(1/scaling * np.exp(hadronic_model(feature)), hadronic_data_eta_constant["WQ"][i]))
    losses_eta_constant.append(error)
    if (i+1) % 10 == 0:
        print("feature:", feature)
        print("error:", error)


#Plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
plt.plot(hadronic_data_x_constant["eta"], predictions_x_constant)
plt.tight_layout()
plt.xlabel(r"$\eta$")
plt.ylabel("WQ")
plt.show()

#Losses plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], losses_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta,x_2
plt.plot(hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["WQ"])
plt.plot(hadronic_data_eta_constant["x_1"], predictions_eta_constant)
plt.xlim(0.0075, 0.2)
plt.ylim(0, 0.08)
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
plt.plot(hadronic_data_eta_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.xlim(0, 0.2)
plt.tight_layout()
plt.show()

