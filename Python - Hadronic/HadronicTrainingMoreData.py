import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import Layers
import time
import os

time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_x12/"
data_name = "logarithmic_hadronic_data_no_negative"
data_name_x_constant = "logarithmic_hadronic_data_no_negative__x_2_constant__0.11"
data_name_eta_x_2_constant = "logarithmic_hadronic_data_no_negative__eta_x_2_constant__0.45"
data_name_eta_x_1_constant = "logarithmic_hadronic_data_no_negative__eta_x_1_constant__0.45"
hadronic_WQ_data_raw = pd.read_csv(data_path+data_name)
#best_losses = pd.read_csv("hadronic_best_losses")


#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.90
batch_size = 32
buffer_size = int(total_data * train_frac)
training_epochs = 5
nr_layers = 2
units = 256
learning_rate = 7e-3
l2_kernel = 0
l2_bias = 0
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = False

model_name = "LessDeep/"
path ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicModels/" + model_name


loss_function = keras.losses.MeanAbsoluteError()
loss_function_name = "MeanAbsoluteError()"
loss_fn_name = "Layers.MeanSquaredLogarithmicError()"

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


training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

training_data = training_data.batch(batch_size=batch_size)
#testing_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
#testing_data = testing_data.batch(batch_size=batch_size)


time3 = time.time()
print("train_features:", train_features)
print("train_labels:", train_labels)
print("training_data:", training_data)

print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
loss_fn = Layers.MeanSquaredLogarithmicError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
hadronic_model = Layers.DNN2(nr_hidden_layers=nr_layers, units=units, outputs=1,
                             loss_fn=loss_fn, optimizer=optimizer,
                             kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias),
                             dropout=dropout, dropout_rate=dropout_rate)

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
    results = tf.math.abs(hadronic_model(test_features, training=False))
    total_loss = loss_function(results, test_labels)
    print("total loss:", float(total_loss))
    for step, (x,y) in enumerate(training_data.as_numpy_iterator()):
        loss = hadronic_model.train_on_batch(x=x, y=y)
        total_steps += 1
        if step % (int(total_data/(8*batch_size))) == 0:
            print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
            steps.append(total_steps)
    losses.append(loss)

#Überprüfen wie gut es war
results = hadronic_model(test_features)
total_loss = loss_function(results, test_labels)
print("total loss:", float(total_loss))

#Losses plotten
plt.plot(epochs, losses)
plt.yscale("log")
plt.ylabel("Losses")
plt.xlabel("Epoch")
plt.savefig(path + "training_losses")
plt.show()


#Daten einlesen
hadronic_data_x_constant = pd.read_csv(data_path + data_name + "__x_2_constant__0.11")
hadronic_data_eta_x_2_constant = pd.read_csv(data_path + data_name + "__eta_x_2_constant__0.45")
hadronic_data_eta_x_1_constant = pd.read_csv(data_path + data_name + "__eta_x_1_constant__0.45")

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
if logarithm:
    predictions_x_constant = ((1 / scaling) * np.exp(hadronic_model(pred_feature_x_constant)))
else:
    predictions_x_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_x_constant)))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["x_2"], hadronic_data_eta_x_2_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
if logarithm:
    predictions_eta_x_2_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))
else:
    predictions_eta_x_2_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))

pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_1_constant["x_1"], hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["eta"]], dtype="float32")
pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)
if logarithm:
    predictions_eta_x_1_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))
else:
    predictions_eta_x_1_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    losses_x_constant.append(float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_x_constant["WQ"][i])))

for i, feature in enumerate(pred_feature_eta_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_eta_x_2_constant["WQ"][i]))
    losses_eta_constant.append(error)

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
plt.plot(hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["WQ"])
plt.plot(hadronic_data_eta_x_2_constant["x_1"], predictions_eta_x_2_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

#Plot mit konstantem eta, x_1
plt.plot(hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["WQ"])
plt.plot(hadronic_data_eta_x_1_constant["x_2"], predictions_eta_x_1_constant)
plt.xlabel(r"$x_2$")
plt.ylabel("WQ")
plt.yscale("log")
plt.legend(r"$x_2$, $\eta$ constant$")
plt.tight_layout()
plt.show()

#Losses plotten mit konstantem eta, x_2
#print("hadronic_data_eta_constant[x_1]:", hadronic_data_eta_constant["x_1"])
#print("losses_eta_constant",losses_eta_constant)
plt.plot(hadronic_data_eta_x_2_constant["x_1"], losses_eta_constant)
plt.xlabel(r"$x_1$")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.show()

#Modell und config speichern
if not os.path.exists(path=path):
    os.makedirs(path)
if not os.path.exists(path=path+model_name):
    os.makedirs(path+model_name)

config = pd.DataFrame(hadronic_model.get_config(), index=[0])
training_parameters = pd.DataFrame(
    {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "scaling": scaling_bool,
        "logarithm": logarithm
    },
    index=[0]
)
print(config)
#config.append(training_parameters)
config = pd.concat([config, training_parameters], axis=1)
print(config)
config = config.transpose()
config.to_csv(path + "config")

hadronic_model.save(filepath=path + model_name)


#script zum plotten erstellen
plot_script = open("plot_scirpt.py", "w")
plot_script.write(  "import tensorflow as tf\n"
                    "import pandas as pd\n"
                    "import numpy as np\n"
                    "from tensorflow import keras\n"
                    "from matplotlib import pyplot as plt \n"
                    "import Layers\n"
                    "#Modell laden \n"
                    "hadronic_model = keras.models.load_model(filepath=" + "\"" + model_name + "\")\n"
                    "logarithm=" + str(logarithm) + "\n"
                    "scaling=" + str(float(scaling)) + "\n"
                    "loss_function=keras.losses." + loss_function_name + "\n"
                    "loss_fn=" + loss_fn_name + "\n"
                    "#Daten einlesen\n"
                    "hadronic_data_x_constant = pd.read_csv(\"" + data_path + data_name_x_constant+ "\")\n"
                    "hadronic_data_eta_x_2_constant = pd.read_csv(\"" + data_path + data_name_eta_x_2_constant + "\")\n"
                    "hadronic_data_eta_x_1_constant = pd.read_csv(\"" + data_path + data_name_eta_x_1_constant + "\")\n"
                    "\n"
                    "#predictions berechnen\n"
                    "pred_feature_x_constant = tf.constant([hadronic_data_x_constant[\"x_1\"], hadronic_data_x_constant[\"x_2\"], hadronic_data_x_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_x_constant = tf.transpose(pred_feature_x_constant)\n"
                    "if logarithm:\n"
                    "    predictions_x_constant = ((1 / scaling) * np.exp(hadronic_model(pred_feature_x_constant)))\n"
                    "else:\n"
                    "    predictions_x_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_x_constant)))\n"
                    "\n"
                    "pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_2_constant[\"x_1\"], hadronic_data_eta_x_2_constant[\"x_2\"], hadronic_data_eta_x_2_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)\n"
                    "if logarithm:\n"
                    "    predictions_eta_x_2_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))\n"
                    "else:\n"
                    "    predictions_eta_x_2_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))\n"
                    "\n"
                    "pred_feature_eta_constant = tf.constant([hadronic_data_eta_x_1_constant[\"x_1\"], hadronic_data_eta_x_1_constant[\"x_2\"], hadronic_data_eta_x_1_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_eta_constant = tf.transpose(pred_feature_eta_constant)\n"
                    "if logarithm:\n"
                    "    predictions_eta_x_1_constant = ((1 / scaling) * tf.math.exp(hadronic_model(pred_feature_eta_constant)))\n"
                    "else:\n"
                    "    predictions_eta_x_1_constant = (tf.math.abs((1/scaling) * hadronic_model(pred_feature_eta_constant)))\n"
                    "\n"
                    "#Loss pro Punkt berechnen\n"
                    "losses_eta_constant = []\n"
                    "losses_x_constant = []\n"
                    "train_losses_eta_constant=[]\n"
                    "train_losses_x_constant=[]\n"
                    "for i, feature in enumerate(pred_feature_x_constant):\n"
                    "    feature = tf.reshape(feature, shape=(1, 3))\n"
                    "    losses_x_constant.append(float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_x_constant[\"WQ\"][i])))\n"
                    "    train_losses_x_constant.append(float(loss_fn((1/scaling)*hadronic_model(feature),hadronic_data_x_constant[\"WQ\"][i])))\n"
                    "\n"
                    "for i, feature in enumerate(pred_feature_eta_constant):\n"
                    "    feature = tf.reshape(feature, shape=(1, 3))\n"
                    "    error = float(loss_function((tf.math.abs((1/scaling)*hadronic_model(feature))), hadronic_data_eta_x_2_constant[\"WQ\"][i]))\n"
                    "    train_losses_eta_constant.append(float(loss_fn((1/scaling)*hadronic_model(feature),hadronic_data_eta_x_2_constant[\"WQ\"][i])))\n"
                    "    losses_eta_constant.append(error)\n"
                    "\n"
                    "\n"
                    "#Plot mit konstantem x_1, x_2 und losses im subplot\n"
                    "fig, (ax0, ax1) = plt.subplots(2)\n"
                    "ax0.plot(hadronic_data_x_constant[\"eta\"], hadronic_data_x_constant[\"WQ\"])\n"
                    "ax0.plot(hadronic_data_x_constant[\"eta\"], predictions_x_constant)\n"
                    "ax0.set(xlabel=r\"$\eta$\", ylabel=\"WQ\")\n"                                                                                                           
                    "ax0.set_title(\"x_1, x_2 constant\")\n"
                    "ax1.plot(hadronic_data_x_constant[\"eta\"], train_losses_x_constant)\n"
                    "ax1.set(xlabel=r\"$\eta$\", ylabel=\"Loss\")\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "\n"
                    "#Losses plot mit konstantem x_1, x_2\n"
                    "plt.plot(hadronic_data_x_constant[\"eta\"], losses_x_constant)\n"
                    "plt.xlabel(r\"$\eta$\")\n"
                    "plt.ylabel(\"Loss\")\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "\n"
                    "#Plot mit konstantem eta,x_2\n"
                    "plt.plot(hadronic_data_eta_x_2_constant[\"x_1\"], hadronic_data_eta_x_2_constant[\"WQ\"])\n"
                    "plt.plot(hadronic_data_eta_x_2_constant[\"x_1\"], predictions_eta_x_2_constant)\n"
                    "plt.xlabel(r\"$x_1$\")\n"
                    "plt.ylabel(\"WQ\")\n"
                    "plt.yscale(\"log\")\n"
                    "plt.legend(r\"$x_2$, $\eta$ constant$\")\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "\n"
                    "#Plot mit konstantem eta, x_1\n"
                    "plt.plot(hadronic_data_eta_x_1_constant[\"x_2\"], hadronic_data_eta_x_1_constant[\"WQ\"])\n"
                    "plt.plot(hadronic_data_eta_x_1_constant[\"x_2\"], predictions_eta_x_1_constant)\n"
                    "plt.xlabel(r\"$x_2$\")\n"
                    "plt.ylabel(\"WQ\")\n"
                    "plt.yscale(\"log\")\n"
                    "plt.legend(r\"$x_2$, $\eta$ constant$\")\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "\n"
                    "#Losses plotten mit konstantem eta, x_2\n"
                    "#print(\"hadronic_data_eta_constant[x_1]:\", hadronic_data_eta_constant[\"x_1\"])\n"
                    "#print(\"losses_eta_constant\",losses_eta_constant)\n"
                    "plt.plot(hadronic_data_eta_x_2_constant[\"x_1\"], losses_eta_constant)\n"
                    "plt.xlabel(r\"$x_1$\")\n"
                    "plt.ylabel(\"Loss\")\n"
                    "plt.yscale(\"log\")\n"
                    "plt.tight_layout()\n"
                    "plt.show()")
#Skript zum plotten erstellen:
plot_script = open(path + "plot_script.py", "w")
plot_script.close()