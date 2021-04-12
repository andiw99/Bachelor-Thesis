import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

#Datenset einlesen
diff_WQ_theta_data_raw = pd.read_csv("ThetaData")
best_losses_theta = pd.read_csv("best_losses_theta")

#Initialisiere Variablen
total_data = len(diff_WQ_theta_data_raw["Theta"])
train_frac = 0.8
batch_size = 64
buffer_size = total_data * train_frac
training_epochs = 5
units = 64
learning_rate  = 3e-5
best_total_loss = best_losses_theta["best_total_loss"]
best_total_loss_squared = best_losses_theta["best_total_loss_squared"]

#prepare dataset
diff_WQ_theta_data_raw = pd.read_csv("ThetaData")
dataset = diff_WQ_theta_data_raw.copy()
train_dataset = dataset.sample(frac=train_frac, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

data_theta = tf.constant(np.array(train_dataset["Theta"]))
print(data_theta)
print(tf.shape(data_theta).numpy()[0])
data_theta = tf.reshape(tensor=data_theta, shape=[tf.shape(data_theta).numpy()[0],1])

data_theta_test = tf.constant(np.array(test_dataset["Theta"]))
data_theta_test = tf.reshape(tensor=data_theta_test, shape=[tf.shape(data_theta_test).numpy()[0],1])
print(data_theta)
#ThetaData = tf.data.Dataset.from_tensor_slices((diff_WQ_theta_data_raw["Theta"], diff_WQ_theta_data_raw["WQ"]))
#ThetaData = tf.data.Dataset.from_tensor_slices(np.array(dict(diff_WQ_theta_data_raw)))
#ThetaData = tf.data.Dataset.from_tensor_slices(np.array(diff_WQ_theta_data_raw))
#ThetaData = tf.data.Dataset.zip()
#ThetaData = tf.data.Dataset.from_tensor_slices((data_theta, tf.reshape(tensor=tf.constant(diff_WQ_theta_data_raw["WQ"]), shape=[6000,1])))
diff_WQ_theta_data = tf.data.Dataset.from_tensor_slices((data_theta, train_dataset["WQ"]))
diff_WQ_theta_data = diff_WQ_theta_data.shuffle(buffer_size=6000)
"""
for step, (x,y) in enumerate(ThetaData):
    print("Step:", step, "theta:", x, "WQ:", y)
    if step >= 10:
        break
"""
diff_WQ_theta_data = diff_WQ_theta_data.batch(batch_size=batch_size)

#diff_WQ_theta_data_batches = tf.data.experimental.make_csv_dataset("ThetaData", batch_size = 10, label_name="WQ")
"""
print(ThetaData)
for step, (x,y) in enumerate(ThetaData):
    print("Step:", step, "theta:", x, "WQ:", y)
    if step >= 10:
        break
"""
print(diff_WQ_theta_data)
"""
    for feature_batch, label_batch in diff_WQ_theta_data_batches.take(2):
        print("WQ: {}".format(label_batch))
        print("features:")
        for key, value in feature_batch.items():
            print("{!r:20s}: {}".format(key, value))
"""

#print(diff_WQ_theta_data_batches)

"""
    for feature_batch in ThetaData.take(2):
        for key, value in feature_batch.items():
            print(" {!r:20s}: {}".format(key, value))
"""

#train_dataset_theta = diff_WQ_theta_data_batches.sample(frac=0.8, random_state=0)
#test_dataset_theta = diff_WQ_theta_data_batches.drop(train_dataset_theta.index)

#Layer klasse
class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, name="Linear_layer"):
        super(Linear, self).__init__()
        self.units = units
        self.weight_name = name

    def build(self, input_shape):
        print("input shape:", input_shape)
        print("input shape[-1]:", input_shape[-1])

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name= self.weight_name,
        )
        print("weights:", self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True,
            name = self.weight_name,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



#DNN Klasse
class MLP(keras.Model):

    def __init__(self, units=64):
        super(MLP, self).__init__()
        self.units = units
        self.linear_1 = Linear(self.units, name="linear_1")  #wird hier die Klasse von oben verwendet? muss eigentlich oder?
        self.linear_2 = Linear(self.units, name="linear_2")
        self.linear_3 = Linear(self.units, name="linear_3")
        self.linear_4 = Linear(1, name="linear_4") #1 Output

    def call(self, inputs):
        x = self.linear_1(inputs)   #hier macht der erste Layer seine Arbeit
        x = tf.nn.relu(x)   #ich muss meine Ausgabe normieren, soweit ich weiß
        x = self.linear_2(x) #Layer 2 macht seine Arbeit
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        x = tf.nn.relu(x)
        return self.linear_4(x)

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

print("serialize Linear:", tf.keras.layers.serialize(layer=Linear()))
print("serialize MLP:", tf.keras.layers.serialize(layer=MLP()))

#Model erstellen und loss und optimizer festlegen
theta_model = MLP(units=units)
loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Training Loop?
@tf.function
def train_on_batch(x,y):
    with tf.GradientTape() as tape:
        print("x:", x)
        logits = theta_model(x)
        print("logits:", logits)
        print("y:", y)
        loss = loss_fn(logits, y)
        gradients = tape.gradient(loss, theta_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, theta_model.trainable_weights))
    return loss

#und feuer, losses tracken
losses = []
steps = []
epochs = []

for epoch in range(training_epochs):
    epochs.append(epoch)
    for step, (x,y) in enumerate(diff_WQ_theta_data):
        loss = train_on_batch(x, y)
        if step % 100 == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(step)
    losses.append(loss)

x= tf.constant([[1], [2]], dtype="float32")
print(x)
print("model prediction für x=1:", theta_model(x))

#tanh^2+1 plotten
plt.plot(test_dataset["Theta"], test_dataset["WQ"])
plt.ylabel("WQ")
plt.xlabel("raw data")
plt.show()

#losses plotten
plt.plot(epochs, losses)
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.show()

#model testen
results = theta_model(data_theta_test)
print(results)
plt.plot(test_dataset["Theta"], results)
plt.ylabel("WQ")
plt.xlabel("pred. data")
plt.show()

WQ_data_test = tf.reshape(tensor=tf.constant(np.array(test_dataset["WQ"]), dtype="float32"), shape=(12000,1) )
print("WQ_data_test:", WQ_data_test)
total_loss = loss_fn(results, WQ_data_test)
total_loss_squared = tf.reduce_sum(tf.square(tf.add(results, -WQ_data_test)))
print("total loss:", float(total_loss))
print("total square loss:", float(total_loss_squared))


theta_model_realoaded = keras.models.load_model(filepath="theta_model")

results_reloaded = theta_model_realoaded(data_theta_test)
total_loss_reloaded = loss_fn(results_reloaded, WQ_data_test)
total_loss_squared_reloaded = tf.reduce_sum(tf.square(tf.add(results_reloaded, -WQ_data_test)))
print("total loss reloaded:", float(total_loss_reloaded))
print("total square loss reloaded:", float(total_loss_squared_reloaded))


if total_loss <= best_total_loss and total_loss_squared <= best_total_loss_squared:
    print("Verbesserung erreicht!")
    print("Verbesserung erreicht!")
    print("Verbesserung erreicht!")

    theta_model.save(filepath="theta_model")

    best_losses_theta = pd.DataFrame(
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

    best_losses_theta.to_csv("best_losses_theta", index=False)
    best_parameters.to_csv("best_parameters", index=False)
else:
    print("keine Verbesserung")