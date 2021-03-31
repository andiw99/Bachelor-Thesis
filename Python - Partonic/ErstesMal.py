import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


#prepare Data
diff_WQ_eta_data = pd.read_csv("diff_WQ_eta_data")

train_dataset_eta = diff_WQ_eta_data.sample(frac=0.8, random_state=0)
test_dataset_eta = diff_WQ_eta_data.drop(train_dataset_eta.index)


print(np.array(train_dataset_eta))

#Was suchen wir uns jetzt für ein Modell aus?
#DNN mit 2 Hidden Layer a 64 Neuronen?

def build_and_compile_model():
    """
    Diese Funktion baut Deep neural network models für gegebene Normalization
    :param norm: Normalization
    :return: DNN Model
    """
    model = keras.Sequential([
        #two hidden layers mit 64 neuronen
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        #ein output layer mit 1 neuron
        layers.Dense(1)
    ])

    model.compile(loss="mean_absolute_error",
                  optimizer=tf.keras.optimizers.Adam(1e-3))
    return model

#for step, (x,y) in enumerate(train_dataset_eta):


eta_model = build_and_compile_model()
history_eta_model = eta_model.fit(
    np.array(train_dataset_eta["Eta"]), np.array(train_dataset_eta["d(sigma)/d(eta)"]),
    epochs = 10,
    verbose = 1,
    validation_split = 0.2)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

plot_loss(history_eta_model)
plt.show()

plt.plot(test_dataset_eta["Eta"], test_dataset_eta["d(sigma)/d(eta)"])
plt.ylabel("d(sigma)/d(eta)")
plt.show()


results_eta = eta_model.evaluate(x=test_dataset_eta["Eta"], y=test_dataset_eta["d(sigma)/d(eta)"])
print("results evaluation:", results_eta)
#mal sehen wie das DNN performed, liste mit funktionswerten, berechnet durch DNN erstellen
results = []
results = eta_model.predict(x=np.array(test_dataset_eta["Eta"]))

plt.plot(test_dataset_eta["Eta"], results)
plt.ylabel("d(sigma)/d(eta)")
plt.show()
print("prediction hoffentlich:", eta_model(tf.constant([[1]])))
print("test_dataset_eta[Eta]", np.array(test_dataset_eta["Eta"]))
print("test_dataset_eta[WQ]", test_dataset_eta["d(sigma)/d(eta)"])
