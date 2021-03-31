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
hadronic_WQ_data_raw = pd.read_csv("HadronicData/logarithmic_hadronic_data_no_negative")
#best_losses = pd.read_csv("hadronic_best_losses")


#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.90
batch_size = 64
buffer_size = int(total_data * train_frac)
training_epochs = 5
nr_layers = 3
units = 128
learning_rate = 5e-3
l2_kernel = 0
l2_bias = 0
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = False

loss_function = keras.losses.MeanAbsoluteError()

#best_total_loss = best_losses
time2= time.time()
print("Zeit nur zum Einlesen von ", total_data, "Punkten:", time2-time1,"s")

#Daten vorbereiten
train_dataset = hadronic_WQ_data_raw.sample(frac=train_frac)
print(train_dataset)
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

train_labels = tf.math.abs(tf.transpose(tf.constant([train_labels], dtype="float32")))
test_labels = tf.math.abs(tf.transpose(tf.constant([test_labels], dtype="float32")))

scaling = 1
if scaling_bool:
    print("minimum:", tf.math.reduce_min(train_labels))
    scaling = 1/tf.math.reduce_min(train_labels)
    print("scaling by:", scaling)
    train_labels = scaling * train_labels
    test_labels = scaling * test_labels

if logarithm:
    train_labels = tf.math.log(train_labels)
    test_labels = tf.math.log(test_labels)

print("min train_labels", tf.math.reduce_min(train_labels))
print("max train_labels", tf.math.reduce_max(train_labels))

"""
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))
#Das mit dem Logarithmus probieren, weil das ja irgendwie geholfen hatte
#train_labels = train_labels+1
#test_labels = test_labels+1
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))
train_labels = tf.math.log(train_labels)
test_labels = tf.math.log(test_labels)
print("test_labels:", test_labels, "test_labels[-1]:", test_labels[-1], "len(test_labels):", len(test_labels))
"""

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
loss_fn = Layers.MeanAbsoluteLogarithmicError()
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
    for step, (x,y) in enumerate(training_data.as_numpy_iterator()):
        loss = hadronic_model.train_on_batch(x=x, y=y, loss_fn=loss_fn, optimizer=optimizer)
        total_steps += 1
        if step % (int(total_data/(8*batch_size))) == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(total_steps)
            losses.append(loss)
    """
    training_data = training_data.unbatch()
    training_data.shuffle(buffer_size=1000000, reshuffle_each_iteration=True)
    training_data = training_data.batch(batch_size=batch_size)
    """

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
    WQ_list.append((float(hadronic_model(x, training=False))))
    eta_list.append(eta)
"""

#Daten einlesen
hadronic_data_x_constant = pd.read_csv("HadronicData/logarithmic_hadronic_data_no_negative__x_constant__0.11")
hadronic_data_eta_constant = pd.read_csv("HadronicData/logarithmic_hadronic_data_no_negative__eta_constant__0.45")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
if logarithm:
    predictions_x_constant = ((1 / scaling) * np.exp(hadronic_model(pred_feature_x_constant)))
else:
    predictions_x_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_x_constant)))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["x_2"], hadronic_data_eta_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
if logarithm:
    predictions_eta_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))
else:
    predictions_eta_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    losses_x_constant.append(float(loss_function(((1/scaling)*hadronic_model(feature)), hadronic_data_x_constant["WQ"][i])))

for i, feature in enumerate(pred_feature_eta_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(((1/scaling)*hadronic_model(feature)), hadronic_data_eta_constant["WQ"][i]))
    losses_eta_constant.append(error)
    if (i+1) % 10 == 0:
        print("feature:", feature)
        print("error:", error)


#Plot mit konstantem x_1, x_2
plt.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
plt.plot(hadronic_data_x_constant["eta"], predictions_x_constant)
plt.xlabel(r"$\eta$")
plt.ylabel("WQ")
plt.legend(r"$ x_1$, $x_2$ constant")
plt.tight_layout()
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
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
plt.plot(hadronic_data_eta_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.show()

