import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
import ml
import time
import os


time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/"
data_name = "log_neg_x12/all"
data_name_x_constant = "log_neg_x12/x_constant"
data_name_eta_x_2_constant = "log_neg_x12/eta_x_2_constant"
data_name_eta_x_1_constant = "log_neg_x12/eta_x_1_constant"
data_name_x_2_constant = "log_neg_3D/x_2_constant__3D"
hadronic_WQ_data_raw = pd.read_csv(data_path+data_name)
#best_losses = pd.read_csv("hadronic_best_losses")


#Variablen...
total_data = len(hadronic_WQ_data_raw["eta"])
train_frac = 0.90
batch_size = 32
buffer_size = int(total_data * train_frac)
training_epochs = 20
nr_layers = 2
units = 512
learning_rate = 2e-5
loss_fn = keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10)
hidden_activation = tf.nn.leaky_relu
output_activation = ml.LinearActiavtion()
kernel_initializer = tf.keras.initializers.RandomNormal(mean=1, stddev=0.2)
bias_initializer =tf.keras.initializers.RandomNormal(mean=1, stddev=0.2)
l2_kernel = 0
l2_bias = 0
dropout = False
dropout_rate = 0.1
scaling_bool = True
logarithm = True
shift = False
label_normalization = False

#Neues Modell oder weitertrainieren?
new_model = False


show_3D_plots = False

model_name = "Scaling+logarithm+MSE+leaky_relu/"
path ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicModels/" + model_name


loss_function = keras.losses.MeanAbsolutePercentageError()
loss_function_name = loss_function.name
loss_fn_name = loss_fn.name


#Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)


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

print("Vor Transformation:")
print("minimum bei: ", float(tf.math.reduce_min(train_labels)), "maximum bei", float(tf.math.reduce_max(train_labels)))


scaling = 1
"""
if scaling_bool:
    print("minimum:", tf.math.reduce_min(train_labels))
    scaling = 1/tf.math.reduce_min(train_labels)
    print("scaling by:", scaling)
    train_labels = scaling * train_labels
    test_labels = scaling * test_labels

if logarithm:
    train_labels = tf.math.log(train_labels)
    test_labels = tf.math.log(test_labels)
    print("Minimum bei: ", tf.math.reduce_min(train_labels))

if shift:
    shift_value = tf.math.reduce_min(train_labels)
    train_labels = train_labels - shift_value
    test_labels = test_labels - shift_value
    print("minimum bei", tf.math.reduce_min(train_labels), "maximum bei", tf.math.reduce_max(train_labels))

if label_normalization:
    normalization_value = tf.math.reduce_max(train_labels)
    print(normalization_value)
    train_labels = train_labels/normalization_value
    test_labels = test_labels/normalization_value
    print("Minimum bei:", float(tf.math.reduce_min(train_labels)), "maximum bei:", float(tf.math.reduce_max(train_labels)))

#Labels transformieren
"""
transformer = ml.label_transformation(train_labels, scaling=scaling_bool, logarithm=logarithm, shift=shift, label_normalization=label_normalization)
train_labels = transformer.transform(train_labels)
test_labels = transformer.transform(test_labels)

print("min train_labels", float(tf.math.reduce_min(train_labels)))
print("max train_labels", float(tf.math.reduce_max(train_labels)))
print("min test_labels", float(tf.math.reduce_min(test_labels)))
print("max test_labels", float(tf.math.reduce_max(test_labels)))

#Rücktrafo testen
backtrafo_labels = transformer.retransform(train_labels)
print("min rücktrafo train_labels:", float(tf.math.reduce_min(backtrafo_labels)))
print("max rücktrafo train_lables:", float(tf.math.reduce_max(backtrafo_labels)))

#get config methode überprüfen
config = transformer.get_config()
print(config)

transformer2 = ml.label_transformation(config=config)
config2 = transformer2.get_config()
print(config2)

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
if new_model:
    hadronic_model = ml.DNN2(
         nr_hidden_layers=nr_layers, units=units, outputs=1,
         loss_fn=loss_fn, optimizer=optimizer,
         hidden_activation=hidden_activation, output_activation=output_activation,
         kernel_initializer=kernel_initializer,
         bias_initializer=bias_initializer,
         kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias),
         dropout=dropout, dropout_rate=dropout_rate
    )
else:
    hadronic_model = keras.models.load_model(filepath=path + "model")
    hadronic_model.compile(optimizer=optimizer, loss=loss_fn)


#Training starten
losses = []
steps =[]
epochs = []
total_steps = 0
if new_model:
    for epoch in range(training_epochs):
        epochs.append(epoch)
        results = []
        loss = 0
        #Evaluation in every epoch
        results = tf.math.abs(hadronic_model(test_features, training=False))
        total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
        print("total loss:", float(total_loss))
        for step, (x,y) in enumerate(training_data.as_numpy_iterator()):
            loss = hadronic_model.train_on_batch(x=x, y=y)
            total_steps += 1
            if step % (int(total_data/(8*batch_size))) == 0:
                print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
                steps.append(total_steps)
        losses.append(loss)
else:
    history = hadronic_model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2, shuffle=True)


#Überprüfen wie gut es war
results = hadronic_model(test_features)
total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
print("total loss:", float(total_loss))

#Losses plotten
if new_model:
    plt.plot(epochs, losses)
    plt.yscale("log")
    plt.ylabel("Losses")
    plt.xlabel("Epoch")
    plt.savefig(path + "training_losses")
    plt.show()

if not new_model:
    plt.plot(history.history["loss"], label="loss")
    plt.legend()
    plt.show()


#Daten einlesen
hadronic_data_x_constant = pd.read_csv(data_path + data_name + "__x_2_constant__0.11")
hadronic_data_eta_x_2_constant = pd.read_csv(data_path + data_name + "__eta_x_2_constant__0.45")
hadronic_data_eta_x_1_constant = pd.read_csv(data_path + data_name + "__eta_x_1_constant__0.45")
hadronic_data_x_2_constant = pd.read_csv(data_path + data_name_x_2_constant)

#predictions berechnen
pred_feature_x_constant = tf.constant([hadronic_data_x_constant["x_1"], hadronic_data_x_constant["x_2"], hadronic_data_x_constant["eta"]], dtype="float32")
pred_feature_x_constant = tf.transpose(pred_feature_x_constant)
predictions_x_constant = transformer.retransform(hadronic_model(pred_feature_x_constant))


pred_feature_eta_x_2_constant = tf.constant([hadronic_data_eta_x_2_constant["x_1"], hadronic_data_eta_x_2_constant["x_2"], hadronic_data_eta_x_2_constant["eta"]], dtype="float32")
pred_feature_eta_x_2_constant = tf.transpose(pred_feature_eta_x_2_constant)
predictions_eta_x_2_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_2_constant))

pred_feature_eta_x_1_constant = tf.constant([hadronic_data_eta_x_1_constant["x_1"], hadronic_data_eta_x_1_constant["x_2"], hadronic_data_eta_x_1_constant["eta"]], dtype="float32")
pred_feature_eta_x_1_constant = tf.transpose(pred_feature_eta_x_1_constant)
predictions_eta_x_1_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_1_constant))

pred_feature_x_2_constant = tf.constant([hadronic_data_x_2_constant["x_1"], hadronic_data_x_2_constant["x_2"], hadronic_data_x_2_constant["eta"]], dtype="float32")
pred_feature_x_2_constant = tf.transpose(pred_feature_x_2_constant)
predictions_x_2_constant = transformer.retransform(hadronic_model(pred_feature_x_2_constant))

#Loss pro Punkt berechnen
losses_eta_constant = []
losses_x_constant = []
losses_x_2_constant = []
for i, feature in enumerate(pred_feature_x_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function((tf.math.abs((1/scaling)*tf.math.exp(hadronic_model(feature)))), hadronic_data_x_constant["WQ"][i]))
    losses_x_constant.append(error)

for i, feature in enumerate(pred_feature_eta_x_2_constant):
    feature = tf.reshape(feature, shape=(1, 3))
    error = float(loss_function(transformer.retransform(hadronic_model(feature)),
                                    hadronic_data_eta_x_2_constant["WQ"][i]))
    losses_eta_constant.append(error)

print("dauert..")
if show_3D_plots:
    for i, feature in enumerate(pred_feature_x_2_constant):
        feature = tf.reshape(feature, shape=(1,3))
        error = float(loss_function(transformer.retransform(hadronic_model(feature)), hadronic_data_x_2_constant["WQ"][i]))
        losses_x_2_constant.append(error)

print("das hier so lange?")
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

#3D Plot mit konstantem x_2
if show_3D_plots:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(hadronic_data_x_2_constant["x_1"], hadronic_data_x_2_constant["eta"], predictions_x_2_constant, cmap=cm.coolwarm)
    ax.set_xlabel("x_1")
    ax.set_ylabel("eta")
    ax.set_zlabel("WQ")
    ax.set_zscale("log")
    plt.tight_layout()
    plt.show()

#3D losses Plot mit konstantem x_2
if show_3D_plots:
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_trisurf(X=hadronic_data_x_2_constant["x_1"], Y=hadronic_data_x_2_constant["eta"], Z=losses_x_2_constant, cmap=cm.coolwarm)
    ax.set_xlabel("x_1")
    ax.set_ylabel("eta")
    ax.set_zlabel("Losses")
    ax.set_zscale("log")
    plt.tight_layout()
    plt.show()

config = pd.DataFrame(hadronic_model.get_config(), index=[0])
training_parameters = pd.DataFrame(
    {
        "learning_rate": learning_rate,
        "epochs": training_epochs,
        "batch_size": batch_size,
        "validation loss": total_loss,
        "scaling": scaling_bool,
        "logarithm": logarithm
    },
    index=[0]
)

#config.append(training_parameters)
config = pd.concat([config, training_parameters], axis=1)

config = config.transpose()
config.to_csv(path + "config")

hadronic_model.save(filepath=path + "model", save_format="tf")


#Skript zum plotten erstellen:
plot_script = open(path + "plot_script.py", "w")
plot_script.write(  "import tensorflow as tf\n"
                    "import pandas as pd\n"
                    "import numpy as np\n"
                    "from tensorflow import keras\n"
                    "from matplotlib import pyplot as plt \n"
                    "import Layers\n"
                    "#Modell laden \n"
                    "hadronic_model = keras.models.load_model(filepath=" + "\"" + "model" + "\")\n"
                    "logarithm=" + str(logarithm) + "\n"
                    "scaling=" + str(float(scaling)) + "\n"
                    "show_3D_plots =" + str(show_3D_plots) + "\n"
                    "config =" + str(transformer.get_config()) + "\n"
                    "transformer = Layers.LabelTransformation(config=config)\n"
                    "loss_function=keras.losses." + loss_function_name + "\n"
                    "loss_fn=" + loss_fn_name + "\n"
                    "#Daten einlesen\n"
                    "hadronic_data_x_constant = pd.read_csv(\"" + data_path + data_name_x_constant+ "\")\n"
                    "hadronic_data_eta_x_2_constant = pd.read_csv(\"" + data_path + data_name_eta_x_2_constant + "\")\n"
                    "hadronic_data_eta_x_1_constant = pd.read_csv(\"" + data_path + data_name_eta_x_1_constant + "\")\n"
                    "hadronic_data_x_2_constant = pd.read_csv(\"" + data_path + data_name_x_2_constant + "\")\n"
                    "\n"
                    "#predictions berechnen\n"
                    "pred_feature_x_constant = tf.constant([hadronic_data_x_constant[\"x_1\"], hadronic_data_x_constant[\"x_2\"], hadronic_data_x_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_x_constant = tf.transpose(pred_feature_x_constant)\n"
                    "predictions_x_constant = transformer.retransform(hadronic_model(pred_feature_x_constant))\n"
                    "\n"
                    "\n"
                    "pred_feature_eta_x_2_constant = tf.constant([hadronic_data_eta_x_2_constant[\"x_1\"], hadronic_data_eta_x_2_constant[\"x_2\"], hadronic_data_eta_x_2_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_eta_x_2_constant = tf.transpose(pred_feature_eta_x_2_constant)\n"
                    "predictions_eta_x_2_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_2_constant))\n"
                    "\n"
                    "pred_feature_eta_x_1_constant = tf.constant([hadronic_data_eta_x_1_constant[\"x_1\"], hadronic_data_eta_x_1_constant[\"x_2\"], hadronic_data_eta_x_1_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_eta_x_1_constant = tf.transpose(pred_feature_eta_x_1_constant)\n"
                    "predictions_eta_x_1_constant = transformer.retransform(hadronic_model(pred_feature_eta_x_1_constant))\n"
                    "\n"
                    "pred_feature_x_2_constant = tf.constant([hadronic_data_x_2_constant[\"x_1\"], hadronic_data_x_2_constant[\"x_2\"], hadronic_data_x_2_constant[\"eta\"]], dtype=\"float32\")\n"
                    "pred_feature_x_2_constant = tf.transpose(pred_feature_x_2_constant)\n"
                    "predictions_x_2_constant = transformer.retransform(hadronic_model(pred_feature_x_2_constant))\n"
                    "\n"
                    "#Loss pro Punkt berechnen\n"
                    "losses_eta_constant = []\n"
                    "losses_x_constant = []\n"
                    "losses_x_2_constant = []\n"
                    "for i, feature in enumerate(pred_feature_x_constant):\n"
                    "    feature = tf.reshape(feature, shape=(1, 3))\n"
                    "    error = float(loss_function(transformer.retransform(tf.math.exp(hadronic_model(feature)), hadronic_data_x_constant[\"WQ\"][i]))\n"
                    "    losses_x_constant.append(error)\n"
                    "    \n"
                    "for i, feature in enumerate(pred_feature_eta_x_2_constant):\n"
                    "    feature = tf.reshape(feature, shape=(1, 3))\n"
                    "    error = float(loss_function(transformer.retransform(hadronic_model(feature)),\n"
                    "                                    hadronic_data_eta_x_2_constant[\"WQ\"][i]))\n"
                    "    losses_eta_constant.append(error)\n"
                    "\n"
                    "print(\"dauert..\")\n"
                    "if show_3D_plots:\n"
                    "    for i, feature in enumerate(pred_feature_x_2_constant):\n"
                    "        feature = tf.reshape(feature, shape=(1,3))\n"
                    "        error = float(loss_function(transformer.retransform(hadronic_model(feature)), hadronic_data_x_2_constant[\"WQ\"][i]))\n"
                    "        losses_x_2_constant.append(error)"
                    "\n"
                    "#Plot mit konstantem x_1, x_2 und losses im subplot\n"
                    "fig, (ax0, ax1) = plt.subplots(2)\n"
                    "ax0.plot(hadronic_data_x_constant[\"eta\"], hadronic_data_x_constant[\"WQ\"])\n"
                    "ax0.plot(hadronic_data_x_constant[\"eta\"], predictions_x_constant)\n"
                    "ax0.set(xlabel=r\"$\eta$\", ylabel=\"WQ\")\n"                                                                                                           
                    "ax0.set_title(\"x_1, x_2 constant\")\n"
                    "ax1.plot(hadronic_data_x_constant[\"eta\"], losses_x_constant)\n"
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
plot_script.close()