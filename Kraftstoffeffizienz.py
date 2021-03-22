import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
#tf.debugging.set_log_device_placement(True)


url = "auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
sns.set_theme()
sns.set()

dataset = raw_dataset.copy()

#unvollständige Daten verwerfen
dataset = dataset.dropna()
#Origin von Zahlen zu Orten konvertieren
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
#ich weiß nicht was das macht
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
#Datensets aufteilen
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)



#plt.show()
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

horsepower = np.array(train_features["Horsepower"])
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)

def build_and_compile_model(norm):
    """
    Diese Funktion baut Deep neural network models für gegebene Normalization
    :param norm: Normalization
    :return: DNN Model
    """
    model = keras.Sequential([
        norm,
        #two hidden layers mit 64 neuronen
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        #ein output layer mit 1 neuron
        layers.Dense(1)
    ])

    model.compile(loss="mean_absolute_error",
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#model mit einer variable
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
#model mit mehreren variablen
linear_model = tf.keras.Sequential([
    #normalizer legt dim der Eingabe fest
    normalizer,
    #wie viele outputs
    layers.Dense(units=1)
])
#DNN model mit einer variable
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

#DNN model mit mehreren variablen
dnn_model = build_and_compile_model(normalizer)

#gespeichertes und reloaded model
reloaded = tf.keras.models.load_model("dnn_model")

#Trainingsmethoden festlegen
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss="mean_absolute_error")

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss = "mean_absolute_error"
)

#Training
history_horsepower = horsepower_model.fit(
    #Dim von train_labels muss mit outputs des Models übereinstimmen
    train_features["Horsepower"], train_labels,
    epochs = 80,
    #supress logging
    verbose = 0,
    #calculate validation results on 20% of the training data
    validation_split=0.2)

history_linear  = linear_model.fit(
    train_features, train_labels,
    epochs = 80,
    verbose=0,
    validation_split=0.2
    )

history_horsepower_dnn = dnn_horsepower_model.fit(
    train_features["Horsepower"], train_labels,
    epochs = 80,
    verbose = 0,
    validation_split=0.2
    )

history_dnn = dnn_model.fit(
    train_features, train_labels,
    epochs = 80,
    verbose = 0,
    validation_split=0.2
    )

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 23])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

#plt.show()

hist = pd.DataFrame(history_horsepower.history)
hist['epoch'] = history_horsepower.epoch
#print(hist)

#test results für später speichern
test_results = {}
test_results["horsepower_model"] = horsepower_model.evaluate(
    test_features["Horsepower"],
    test_labels, verbose=0
)
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)
test_results["horsepower_dnn_model"] = horsepower_model.evaluate(
    test_features["Horsepower"],
    test_labels, verbose=0
)
test_results["dnn_model"] = dnn_model.evaluate(
    test_features, test_labels, verbose=0
)
test_results["reloaded"] = reloaded.evaluate(test_features, test_labels, verbose=0)

#prediction des horspower models grafisch
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)
z = dnn_horsepower_model.predict(x)

def plot_horsepower(x,y):
    plt.scatter(train_features["Horsepower"], train_labels, label="Data")
    plt.plot(x, y, color = "k", label = "Predictions")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()

#plot_loss(history_linear)
#plot_loss(history_dnn)
#plt.show()
#plot_horsepower(x,z)

#make predictions
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=30)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()


print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)
print(tf.config.list_physical_devices("GPU"))