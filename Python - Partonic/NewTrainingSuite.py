import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
import Layers
import time
import os


time1 = time.time()
#Daten einlesen
data_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/"
data_name = "diff_WQ_theta_data"
project_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/"
loss_name = "partonic_loss_theta"
project_name = "PartonicTheta/"

model_name = "NewTrainingSuitetransformerconfigtest/"
path ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicModels/" + project_name + model_name

#data_name_x_constant = "log_neg_x12/logarithmic_hadronic_data_no_negative__x_2_constant__0.11"
#data_name_eta_x_2_constant = "log_neg_x12/logarithmic_hadronic_data_no_negative__eta_x_2_constant__0.45"
#data_name_eta_x_1_constant = "log_neg_x12/logarithmic_hadronic_data_no_negative__eta_x_1_constant__0.45"
#data_name_x_2_constant = "log_neg_3D/logarithmic_hadronic_data_no_negative__x_2_constant__3D"
data_raw = pd.read_csv(data_path+data_name)
#Überprüfen, ob es für das vorliegende Problem schon losses gibt und ggf einlesen
best_losses = None
if os.path.exists(project_path+ project_name + loss_name):
    best_losses = pd.read_csv(project_path+ project_name + loss_name)


#Variablen...
total_data = len(data_raw["WQ"])
train_frac = 0.85
batch_size = 32
buffer_size = int(total_data * train_frac)
training_epochs = 50
nr_layers = 2
units = 128
learning_rate = 1e-5
loss_fn = keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10)
hidden_activation = tf.nn.leaky_relu
output_activation = Layers.LinearActiavtion()
kernel_initializer = tf.keras.initializers.RandomNormal()
bias_initializer =tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.05)
l2_kernel = 0
l2_bias = 0
dropout = False
dropout_rate = 0.1
scaling_bool = False
logarithm = False
shift = False
label_normalization = False

#ggf Losses einlesen
if best_losses is not None:
    best_percentage_loss = best_losses["best_percentage_loss"]

#Neues Modell oder weitertrainieren?
new_model = False
#Custom Layers oder Keras Layers?
custom = False

#3D-plots zeigen?
show_3D_plots = False

#Loss für validation loss festlegen
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
train_dataset = data_raw.sample(frac=train_frac)
test_dataset = data_raw.drop(train_dataset.index)

#In Features und Labels unterteilen
train_features_pd = train_dataset.copy()
test_features_pd = test_dataset.copy()

train_labels_pd = train_features_pd.pop("WQ")
test_labels_pd = test_features_pd.pop("WQ")

#Aus den Pandas Dataframes tf-Tensoren machen
for i,key in enumerate(train_features_pd):
    print(i)
    if i == 0:
        train_features = tf.constant([train_features_pd[key]], dtype="float32")
    else:
        more_features = tf.constant([train_features_pd[key]], dtype="float32")
        train_features = tf.experimental.numpy.append(train_features, more_features, axis=0)

for i,key in enumerate(test_features_pd):
    if i == 0:
        test_features = tf.constant([test_features_pd[key]], dtype="float32")
    else:
        more_features = tf.constant([test_features_pd[key]], dtype="float32")
        test_features = tf.experimental.numpy.append(test_features, more_features, axis=0)

#Dimensionen arrangieren
train_features = tf.transpose(train_features)
test_features = tf.transpose(test_features)

train_labels = tf.math.abs(tf.transpose(tf.constant([train_labels_pd], dtype="float32")))
test_labels = tf.math.abs(tf.transpose(tf.constant([test_labels_pd], dtype="float32")))

transformer = Layers.LabelTransformation(train_labels, scaling=scaling_bool, logarithm=logarithm, shift=shift, label_normalization=label_normalization)
train_labels = transformer.transform(train_labels)
test_labels = transformer.transform(test_labels)

training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

training_data = training_data.batch(batch_size=batch_size)
time3 = time.time()

print("Zeit, um Daten vorzubereiten:", time3-time1)

#initialisiere Model
if new_model:
    if custom:
        model = Layers.DNN2(
             nr_hidden_layers=nr_layers, units=units, outputs=1,
             loss_fn=loss_fn, optimizer=optimizer,
             hidden_activation=hidden_activation, output_activation=output_activation,
             kernel_initializer=kernel_initializer,
             bias_initializer=bias_initializer,
             kernel_regularization=keras.regularizers.l2(l2=l2_kernel), bias_regularization=keras.regularizers.l2(l2=l2_bias),
             dropout=dropout, dropout_rate=dropout_rate
        )
    if not custom:
        model = keras.Sequential()
        #Architektur aufbauen
        for i in range(nr_layers):
            if dropout:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.Dense(units=units, activation=hidden_activation, name="Layer_" + str(i),
                                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                                  kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                                  bias_regularizer=keras.regularizers.l2(l2=l2_bias)))
        #Output layer nicht vergessen
        model.add(keras.layers.Dense(units=1, activation=output_activation, name="Output_layer",
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                     kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                     bias_regularizer=keras.regularizers.l2(l2=l2_bias)
                                     ))
        #Model compilen
        model.compile(optimizer=optimizer, loss=loss_fn)

else:
    model = keras.models.load_model(filepath=path + "model")
    model.compile(optimizer=optimizer, loss=loss_fn)


#Training starten
losses = []
steps =[]
epochs = []
total_steps = 0
if new_model and custom:
    for epoch in range(training_epochs):
        epochs.append(epoch)
        results = []
        loss = 0
        #Evaluation in every epoch
        results = tf.math.abs(model(test_features, training=False))
        total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
        print("total loss:", float(total_loss))
        for step, (x,y) in enumerate(training_data.as_numpy_iterator()):
            loss = model.train_on_batch(x=x, y=y)
            total_steps += 1
            if step % (int(total_data/(8*batch_size))) == 0:
                print("Epoch:", epoch+1, "Step:", step, "Loss:", float(loss))
                steps.append(total_steps)
        losses.append(loss)
else:
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=training_epochs, verbose=2, shuffle=True)


#Überprüfen wie gut es war
results = model(test_features)
total_loss = loss_function(transformer.retransform(results), transformer.retransform(test_labels))
print("total loss:", float(total_loss))

#Losses plotten
if new_model and custom:
    plt.plot(epochs, losses)
    plt.yscale("log")
    plt.ylabel("Losses")
    plt.xlabel("Epoch")
    plt.savefig(path + "training_losses")
    plt.show()

else:
    plt.plot(history.history["loss"], label="loss")
    plt.ylabel("Losses")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.legend()
    plt.show()

#Modell und config speichern
config = pd.DataFrame(model.get_config())
training_parameters = pd.DataFrame(
    {
        "learning_rate": learning_rate,
        "epochs": training_epochs,
        "batch_size": batch_size,
        "validation loss": total_loss,
        "scaling": scaling_bool,
        "logarithm": logarithm,
        "transformer config": str(transformer.get_config())
    },
    index=[0]
)

#config.append(training_parameters)
config = pd.concat([config, training_parameters], axis=1)

config = config.transpose()
config.to_csv(path + "config")

model.save(filepath=path + "model", save_format="tf")

#Überprüfen, ob Fortschritt gemacht wurde
results = transformer.retransform(model(test_features))
percentage_loss = keras.losses.MeanAbsolutePercentageError()
validation_loss = percentage_loss(y_true=transformer.retransform(test_labels), y_pred=results)
print("percentage loss:", float(validation_loss))

losses_data = pd.DataFrame(
    {
        "best_percentage_loss": [float(validation_loss)]
    }
)


if best_losses is not None:
    if float(validation_loss) < float(best_losses["best_percentage_loss"]):
        model.save(filepath=project_path + project_name + "best_model", save_format="tf")
        losses_data.to_csv(project_path + project_name + loss_name, index=False)
        if new_model:
            config.to_csv(project_path + project_name + "best_config")
        print("VERBESSERUNG ERREICHT!")

elif best_losses == None:
    model.save(filepath=project_path + project_name + "best_model", save_format="tf")
    losses_data.to_csv(project_path + project_name + loss_name, index=False)
    config.to_csv(project_path + project_name + "best_config")
    print("VERBESSERUNG ERREICHT!")

#weitere Daten einlesen für höherdimensionale plots allgemein um die plots kümmern, gar keine plots mehr direkt nach dem Training?
"""
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
"""