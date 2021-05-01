import ast

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import MC
import os

#Bis auf weiters output auskommentiert, da es nicht benutzt wird und auf taurus nicht läuft
"""
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import control_flow_util
"""


class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, name="Linear_layer", kernel_initializer=keras.initializers.HeNormal(), bias_initializer=keras.initializers.HeNormal(),
                 kernel_regularization=None, bias_regularization=None):
        super(Linear, self).__init__()
        self.units = units
        self.weight_name = name
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        #print("input shape in Layer:", self.weight_name, input_shape)
        #print("input shape[-1]:", input_shape[-1])

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name= self.weight_name,
        )
        #print("weights von:", self.weight_name, self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer=self.bias_initializer, trainable=True,
            name = self.weight_name,
        )
        #print("weights von", self.weight_name, ":", self.w)

    def call(self, inputs, training=True):
        output = tf.math.add(tf.linalg.matmul(inputs, self.w), self.b)
        #print("output von Layer:", self.weight_name, output)
        return output

    def get_regularization(self):
        if self.kernel_regularization == None:
            return self.bias_regularization(self.b)
        if self.bias_regularization == None:
            return self.kernel_regularization(self.w)
        else:
            return self.kernel_regularization(self.w) + self.bias_regularization(self.b)

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MLP(keras.Model):

    def __init__(self, units=64):
        super(MLP, self).__init__()
        self.units = units
        self.linear_1 = Linear(self.units, name="linear_1")
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

    @tf.function
    def train_on_batch(self,
                            x,
                            y,
                            loss_fn,
                            optimizer):
        with tf.GradientTape() as tape:
            logits = self.call(x)
            loss = loss_fn(logits, y)
            gradients = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return loss

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DNN(keras.Model):
    def __init__(self, nr_hidden_layers=3, units=64, outputs=1, kernel_regularization=None, bias_regularization=None, dropout=False, dropout_rate=0):
        super(DNN, self).__init__()
        self.units = units
        self.nr_hidden_layers = nr_hidden_layers
        self.hidden_layers = []
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization
        self.dropout_rate = dropout_rate
        self.names = set()
        self.dropout = dropout

        for i in range(self.nr_hidden_layers):
            name = "Layer_" + str(i)
            dropout_name = "Dropout_" + str(i)
            self.names.add(name)
            if dropout:
                self.hidden_layers.append(Dropout(rate=self.dropout_rate, name=dropout_name))
            self.hidden_layers.append(Linear(units=self.units, name=name, kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization))
        #self.hidden_layers = [Linear(units=self.units, name="hidden_layer") in range(self.nr_hidden_layers)]
        self.linear_output = Linear(units=outputs, name="output_layer", kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization)
        #print("names:", self.names)


    def call(self, inputs, training=True):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            if self.dropout:
                if layer.weight_name in self.names:
                    x = tf.nn.relu(x)
            else:
                x = tf.nn.relu(x)
        x = self.linear_output(x, training=training)
        return tf.nn.relu(x)

    def get_regularization(self):
        penalty = 0
        if self.kernel_regularization != None or self.bias_regularization != None:
            for layer in self.hidden_layers:
                penalty += layer.get_regularization()
            penalty += self.linear_output.get_regularization()
        return penalty


    @tf.function
    def train_on_batch(self,
                            x,
                            y,
                            loss_fn,
                            optimizer):
        with tf.GradientTape() as tape:
            logits = self.call(x)
            loss = float(loss_fn(logits, y)) + self.get_regularization()
            gradients = tape.gradient(loss, self.trainable_weights)
            #print("gradienten:", gradients)
        optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return loss


    def get_config(self):
        return {"units": self.units,
                "nr_hidden_layers": self.nr_hidden_layers,
                "kernel_regularization": self.kernel_regularization,
                "bias_regularization": self.bias_regularization,
                "dropout": self.dropout,
                "dropout_rate": self.dropout_rate
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DNN2(keras.Model):
    def __init__(self, nr_hidden_layers=3, units=64, loss_fn=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(),
                 hidden_activation=tf.nn.sigmoid, output_activation=tf.nn.relu,  outputs=1, kernel_initializer=keras.initializers.HeNormal(), bias_initializer = keras.initializers.HeNormal(),
                 kernel_regularization=None, bias_regularization=None, dropout=False, dropout_rate=0):
        super(DNN2, self).__init__()
        self.units = units
        self.nr_hidden_layers = nr_hidden_layers
        self.hidden_layers = []
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization
        self.dropout_rate = dropout_rate
        self.names = set()
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.hidden_actviation = hidden_activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        for i in range(self.nr_hidden_layers):
            name = "Layer_" + str(i)
            dropout_name = "Dropout_" + str(i)
            self.names.add(name)
            if dropout:
                self.hidden_layers.append(Dropout(rate=self.dropout_rate, name=dropout_name))
            self.hidden_layers.append(Linear(units=self.units, name=name, kernel_initializer = self.kernel_initializer, bias_initializer=self.bias_initializer,
                                             kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization))
        #self.hidden_layers = [Linear(units=self.units, name="hidden_layer") in range(self.nr_hidden_layers)]
        self.linear_output = Linear(units=outputs, name="output_layer", kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization)
        #print("names:", self.names)


    def call(self, inputs, training=True):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            if self.dropout:
                if layer.weight_name in self.names:
                    x = self.hidden_actviation(x)
            else:
                x = self.hidden_actviation(x)
        x = self.linear_output(x, training=training)
        return self.output_activation(x)

    def get_regularization(self):
        penalty = 0
        if self.kernel_regularization != None or self.bias_regularization != None:
            for layer in self.hidden_layers:
                penalty += layer.get_regularization()
            penalty += self.linear_output.get_regularization()
        return penalty

    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
        self.loss_fn = loss
        self.optimizer = optimizer

    @tf.function
    def train_on_batch(self,
                            x,
                            y
                       ):
        with tf.GradientTape() as tape:
            logits = self.call(x)
            loss = float(self.loss_fn(logits, y)) + self.get_regularization()
            gradients = tape.gradient(loss, self.trainable_weights)
            #print("gradienten:", gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return loss

    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):

        if shuffle:
            print("In Custom models hast du shuffle noch nicht implementiert.")

        training_data = tf.data.Dataset.from_tensor_slices((x, y))
        training_data = training_data.batch(batch_size=batch_size)
        total = tf.size(y)
        history = []
        for epoch in range(epochs):
            print("Epoch ",epoch,"/",epochs)
            loss = 0
            for step, (x, y) in enumerate(training_data.as_numpy_iterator()):
                loss += self.train_on_batch(x=x, y=y)
            loss = loss/total
            if verbose:
                print("loss: ", loss)
            history.append(loss)
        return history


    def get_config(self):
        return {"units": self.units,
                "nr_hidden_layers": self.nr_hidden_layers,
                "loss_fn": self.loss_fn.name,
                "optimizer": self.optimizer._name,
                "hidden_activation": self.hidden_actviation.__name__,
                "output_activation": self.output_activation.__name__,

                "kernel_regularization": self.kernel_regularization,
                "bias_regularization": self.bias_regularization,
                "dropout": self.dropout,
                "dropout_rate": self.dropout_rate
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MeanSquaredLogarithmicError1p(tf.keras.losses.Loss):
    def __init__(self, name="mean_squared_logarithmic_error_1p"):
        super(MeanSquaredLogarithmicError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        ln_y_pred = tf.math.log1p(y_pred)
        ln_y_true = tf.math.log1p(y_true)
        diff = tf.math.subtract(x=ln_y_pred, y=ln_y_true)
        loss = tf.reduce_mean(tf.square(diff))
        return loss

class MeanSquaredLogarithmicError(tf.keras.losses.Loss):
    def __init__(self, name="mean_squared_logarithmic_error"):
        super(MeanSquaredLogarithmicError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        ln_y_pred = tf.math.log(y_pred)
        ln_y_true = tf.math.log(y_true)
        diff = tf.math.subtract(x=ln_y_pred, y=ln_y_true)
        loss = tf.reduce_mean(tf.square(diff))
        return loss

class MeanAbsoluteLogarithmicError(tf.keras.losses.Loss):
    def __init__(self, name="mean_absolute_logarithmic_error"):
        super(MeanAbsoluteLogarithmicError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        ln_y_pred = tf.math.log(tf.math.abs(y_pred))
        ln_y_true = tf.math.log(y_true)
        diff = tf.math.subtract(x=ln_y_pred, y=ln_y_true)
        loss = tf.reduce_mean(tf.abs(diff))
        return loss

class MeanLogarithmOfAbsoluteError(tf.keras.losses.Loss):
    def __init__(self, name="mean_logarithm_of_absolute_error"):
        super(MeanLogarithmOfAbsoluteError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        diff = tf.math.subtract(x=y_pred, y=y_true)
        loss = tf.reduce_mean((tf.abs(diff)))
        return tf.math.log(1+loss)

class CustomError(tf.keras.losses.Loss):
    def __init__(self, scaling = 1, name="Custom_error"):
        super(CustomError, self).__init__()
        self.name = name
        self.scaling = scaling

    def __call__(self, y_pred, y_true, sample_weight=None):
        #Anstatt zwischen 1 und 1e-16 arbeiten wir jetzt zwischen 1e+16 und 1
        scaled_y_true = self.scaling * y_true
        scaled_y_pred = tf.math.abs(self.scaling * y_pred)
        #Wir berechnen den Logarithmus der Werte und verringern somit das intervall auf [0,40]
        log_y_true_scaled = tf.math.log(scaled_y_true)
        log_y_pred_scaled = tf.math.log(scaled_y_pred)
        diff = tf.math.subtract(x=log_y_pred_scaled, y=log_y_true_scaled)
        loss = tf.reduce_mean(tf.abs(diff))
        #print("loss:", loss)
        return loss

class MeanAbsoluteError(tf.keras.losses.Loss):
    def __init__(self, name="mean_absolute_error"):
        super(MeanAbsoluteError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        diff = tf.math.subtract(x=y_pred, y=y_true)
        loss = tf.reduce_mean(tf.abs(diff))
        return loss

class MeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, name="mean_squared_error"):
        super(MeanSquaredError, self).__init__()
        self.name = name

    def __call__(self, y_pred, y_true, sample_weight=None):
        diff = tf.math.subtract(x=y_pred, y=y_true)
        loss = tf.reduce_mean(tf.math.square(diff))
        return loss

class LinearActiavtion():
    def __init__(self, name="linear"):
        self.__name__ = name

    def __call__(self, x):
        return x

class label_transformation():
    def __init__(self, train_labels=None, scaling=False, logarithm=False, shift=False, label_normalization=False, config=None, base10=False, feature_rescaling=False):
        self.scaling = scaling
        self.logarithm = logarithm
        self.shift = shift
        self.label_normalization = label_normalization
        self.values = dict()
        self.base10 = base10
        self.feature_rescaling = feature_rescaling
        if train_labels is not None:
            if self.scaling:
                self.scale = 1/np.min(train_labels)
                self.values["scale"] = self.scale
                train_labels = train_labels * self.scale
            if logarithm:
                if self.base10:
                    train_labels = np.log10(train_labels)
                else:
                    train_labels = np.log(train_labels)
            if self.shift:
                self.shift_value = np.min(train_labels)
                self.values["shift_value"] = self.shift_value
                train_labels = train_labels - self.shift_value
            if self.label_normalization:
                self.normalization_value = np.max(train_labels)
                self.values["normalization_value"] = self.normalization_value

        if config:
            if config["scaling"]:
                self.scale = config["scale"]
                self.values["scale"] = self.scale
                self.scaling = config["scaling"]
            if config["logarithm"]:
                self.logarithm = config["logarithm"]
            if config["shift"]:
                self.shift = config["shift"]
                self.shift_value = config["shift_value"]
                self.values["shift_value"] = self.shift_value
            if config["label_normalization"]:
                self.label_normalization = config["label_normalization"]
                self.normalization_value = config["normalization_value"]
                self.values["normalization_value"] = self.normalization_value
            if config["feature_rescaling"]:
                self.feature_rescaling = config["feature_rescaling"]
            if config["base10"]:
                self.base10 = config["base10"]


    def transform(self, x):
        if self.scaling:
            x = self.scale * x
        if self.logarithm:
            if self.base10:
                x = np.log10(x)
            else:
                x = np.log(x)
        if self.shift:
            x = x - self.shift_value
        if self.label_normalization:
            x = x/(1/2* self.normalization_value) - 1
        return x

    def retransform(self, x):
        if self.label_normalization:
            x = (x + 1) * 1/2 * self.normalization_value
        if self.shift:
            x = x + self.shift_value
        if self.logarithm:
            if self.base10:
                x = 10 ** x
            else:
                x = np.exp(x)
        if self.scaling:
            x = (1/self.scale) * x
        return x

    def rescale(self, x):
        x = np.array(x)
        if self.feature_rescaling:
            for i in range(x.shape[1]):
                print(np.max(x[:,i]))
                x[:, i] = x[:, i] - np.min(x[:, i])
                x[:, i] = x[:, i] / ((1 / 2) * np.max(x[:, i])) - 1
        x = tf.constant(x)
        return x

    def get_config(self):
        config = {
            "scaling": self.scaling,
            "logarithm": self.logarithm,
            "base10": self.base10,
            "shift": self.shift,
            "label_normalization": self.label_normalization,
            "feature_rescaling": self.feature_rescaling
        }
        for transformation, value in self.values.items():
            config[transformation] = float(value)
        return config


#Dropout class stammt aus der tensorflow doku
Dropout = LinearActiavtion
#Bis auf weiters output auskommentiert, da es nicht benutzt wird und auf taurus nicht läuft
"""
class Dropout(keras.layers.Layer):
  def __init__(self, rate, name=None, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True
    self.weight_name = name

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = control_flow_util.smart_cond(training, dropped_inputs,
                                          lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_regularization(self):
      return 0
"""

def data_handling(data_path, label_name, scaling_bool=False, logarithm=False, base10=False, shift=False,
                  label_normalization=False, feature_rescaling=False, train_frac=1 , validation_total=0, batch_size=64, return_pd=False,
                  lower_cutoff=1e-20, upper_cutoff=1e+3, label_cutoff=True, transformer=None, return_as_tensor=True):
    #Daten einlesen
    data = pd.read_csv(data_path)
    #in test und trainingsdaten untertaeilen
    if (train_frac != 1):
        train_dataset = data.sample(frac=train_frac)
    elif (validation_total != 0):
        train_frac = (1 - validation_total/len(data[label_name]))
        train_dataset = data.sample(frac=train_frac)
    else:
        train_dataset = data.copy()
    test_dataset = data.drop(train_dataset.index)
    #In features und labels unterteilen
    train_features_pd = train_dataset.copy()
    test_features_pd = test_dataset.copy()

    train_labels_pd = train_features_pd.pop(label_name)
    test_labels_pd = test_features_pd.pop(label_name)

    # Aus den Pandas Dataframes np-arrays machen
    for i, key in enumerate(train_features_pd):
        if i == 0:
            train_features = np.array([train_features_pd[key]], dtype="float32")
        else:
            more_features = np.array([train_features_pd[key]], dtype="float32")
            train_features = np.append(train_features, more_features, axis=0)

    for i, key in enumerate(test_features_pd):
        if i == 0:
            test_features = np.array([test_features_pd[key]], dtype="float32")
        else:
            more_features = np.array([test_features_pd[key]], dtype="float32")
            test_features = np.append(test_features, more_features, axis=0)

    # Dimensionen arrangieren
    train_features = np.transpose(train_features)
    test_features = np.transpose(test_features)

    train_labels =np.transpose(np.array([train_labels_pd], dtype="float32"))
    test_labels = np.transpose(np.array([test_labels_pd], dtype="float32"))

    #Ggf. Punkte mit WQ 0 entfernen
    if label_cutoff:
        print("size train_labels bevor WQ cutoff:", train_labels.size)
        train_features = train_features[(train_labels[:,0] > lower_cutoff) & (train_labels[:,0] < upper_cutoff)]
        test_features = test_features[(test_labels[:,0] > lower_cutoff) & (test_labels[:,0] < upper_cutoff)]

        train_labels = train_labels[(train_labels[:,0] > lower_cutoff) & (train_labels[:,0] < upper_cutoff)]
        test_labels = test_labels[(test_labels[:,0] > lower_cutoff) & (test_labels[:,0] < upper_cutoff)]
        print("size train_lables nach WQ cutoff:", train_labels.size)

    # Labels transformieren
    # Transformer initialisieren
    if transformer:
        transformer = transformer
    else:
        transformer = label_transformation(train_labels, scaling=scaling_bool, logarithm=logarithm, base10=base10, shift=shift,
                                       label_normalization=label_normalization, feature_rescaling=feature_rescaling)
    train_labels = transformer.transform(train_labels)
    test_labels = transformer.transform(test_labels)

    #Gegebenfalls die Features auf [-1.1] rescalen
    train_features = transformer.rescale(train_features)
    test_features = transformer.rescale(test_features)

    #numpy arrays zu tensorflow-tensoren machen, wegen schneller verarbeitung
    training_data = None
    if return_as_tensor:
        train_features = tf.constant(train_features, dtype="float32")
        test_features = tf.constant(test_features, dtype="float32")
        train_labels = tf.constant(train_labels, dtype="float32")
        test_labels = tf.constant(test_labels, dtype="float32")


        training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        training_data = training_data.batch(batch_size=batch_size)

    if return_pd:
        re = (training_data, train_features, train_labels, test_features, test_labels, train_features_pd, train_labels_pd, transformer)
    else:
        re = (training_data, train_features, train_labels, test_features, test_labels, transformer)
    return re


def initialize_model(nr_layers=3, units=512, loss_fn=keras.losses.MeanAbsoluteError(),
                     optimizer=keras.optimizers.Adam, hidden_activation=tf.nn.leaky_relu,
                     output_activation=LinearActiavtion, kernel_initializer=keras.initializers.HeNormal,
                     bias_initializer=keras.initializers.Zeros, l2_kernel=0, l2_bias=0,
                     dropout=False, dropout_rate=0, learning_rate=5e-3, momentum = 0,
                     nesterov=False, feature_normalization = False, new_model=True, custom=False,
                     read_path=None, freeze=False, metrics=None,
                     transfer=False, source_model=None, rm_layers=1, add_layers=0):
    if new_model:
        if custom:
            model = DNN2(
                nr_hidden_layers=nr_layers, units=units, outputs=1,
                loss_fn=loss_fn, optimizer=optimizer,
                hidden_activation=hidden_activation, output_activation=output_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularization=keras.regularizers.l2(l2=l2_kernel),
                bias_regularization=keras.regularizers.l2(l2=l2_bias),
                dropout=dropout, dropout_rate=dropout_rate
            )
        if not custom:
            model = keras.Sequential()
            # Architektur aufbauen
            if feature_normalization:
                model.add(keras.layers.experimental.preprocessing.Normalization())
            for i in range(nr_layers):
                if dropout:
                    model.add(keras.layers.Dropout(rate=dropout_rate))
                model.add(keras.layers.Dense(units=units, activation=hidden_activation, name="Layer_" + str(i),
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                             kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                             bias_regularizer=keras.regularizers.l2(l2=l2_bias)))
            # Output layer nicht vergessen
            model.add(keras.layers.Dense(units=1, activation=output_activation, name="Output_layer",
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                         bias_regularizer=keras.regularizers.l2(l2=l2_bias)
                                         ))

    else:
        if transfer:
            #TODO ist das hier valid oder zerschieße ich mir das source model zwischen den Trainings?
            print("model summary vor umbau:")
            print(source_model.summary())
            prev_model = source_model
            for i in range(rm_layers):
                prev_model.pop()
            prev_model.trainable = False

            model = prev_model
            for i in range(add_layers + 1):
                inp = model.input
                if i == (add_layers):
                    new_layer = keras.layers.Dense(units=1, activation=output_activation, name="New_Output_layer",
                                                   kernel_initializer=kernel_initializer,
                                                   bias_initializer=bias_initializer,
                                                   kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                                   bias_regularizer=keras.regularizers.l2(l2=l2_bias))

                else:
                    new_layer = keras.layers.Dense(units=units, activation=hidden_activation,
                                                   name="New_Layer_" + str(i),
                                                   kernel_initializer=kernel_initializer,
                                                   bias_initializer=bias_initializer,
                                                   kernel_regularizer=keras.regularizers.l2(l2=l2_kernel),
                                                   bias_regularizer=keras.regularizers.l2(l2=l2_bias))

                out = new_layer(model.output)
                model = keras.Model(inp, out)
            print("model summary nach umbau:")
            print(model.summary())

        else:
            model = keras.models.load_model(filepath=read_path)
            if freeze:
                for layer in model.layers:
                    layer.trainable = False
                model.layers[-1].trainable = True
            print(model.summary())
    #optimizer instanz erstellen
    optimizer = construct_optimizer(optimizer=optimizer, learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model

def make_losses_plot(history):
    fig, ax = plt.subplots()
    if not type(history) == keras.callbacks.History:
        ax.plot(history, label="loss")
    else:
        ax.plot(history.history["loss"], label="loss")
    ax.set_ylabel("Losses")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def save_config(new_model, save_path, model, learning_rate, training_epochs, batch_size, avg_total_Loss=0,
                transformer=None, training_time=None, loss_fn=None, custom=False, feature_handling=None, min_delta=None, offset=None,
                nr_hidden_layers=None, lr_reduction=None, lr_factor=None, total_losses=[], smallest_loss=0.0,
                loss_error=0.0, read_path=None, fine_tuning=None, source_model=None, units=None):
    if new_model:
        config = pd.DataFrame([model.get_config()])
        training_parameters = pd.DataFrame(
            {
                "learning_rate": [learning_rate],
                "epochs": [training_epochs],
                "batch_size": [batch_size],
                "total_losses": [total_losses],
                "avg validation loss": ["{:.5f}".format(float(avg_total_Loss))],
                "smallest loss": ["{:.5f}".format(float(smallest_loss))],
                "loss error": ["{:.5f}".format(float(loss_error))],
                "transformer_config": [str(transformer.get_config())],
                "training time": ["{:.2f}".format(training_time)],
                "Custom": [custom],
                "loss_fn": [loss_fn.name],
                "feature_rescaling": [feature_handling],
                "min_delta": [min_delta],
                "nr_hidden_layers": [nr_hidden_layers],
                "units": [units],
                "lr_reduction": [lr_reduction],
                "lr_factor": [lr_factor],
                "fine_tuning": [fine_tuning],
                "source_model": [source_model],
                "offset": [offset]
            }
        )
        # config.append(training_parameters)
        config = pd.concat([config, training_parameters], axis=1)
        index = True
    if not new_model:
        config = pd.read_csv(read_path + "/config", index_col="property")
        config = config.transpose()
        i = 0
        while pd.notna(config["learning_rate"][i]):
            i += 1
        config["learning_rate"][i] = learning_rate
        index = True

    config = config.transpose()
    config.to_csv(save_path + "/config", index=index, index_label="property")

    return (config, index)

def check_progress(model, transformer, test_features, test_labels, best_losses, project_path, project_name, index, config, loss_name):
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
            config.to_csv(project_path + project_name + "best_config", index=index, index_label="property")
            config.to_csv(project_path + project_name + "best_model/config", index=index, index_label="property")
            print("VERBESSERUNG ERREICHT!")

    elif best_losses == None:
        model.save(filepath=project_path + project_name + "best_model", save_format="tf")
        losses_data.to_csv(project_path + project_name + loss_name, index=False)
        config.to_csv(project_path + project_name + "best_model/config", index=index, index_label="property")
        config.to_csv(project_path + project_name + "best_config", index=index, index_label="property")
        print("VERBESSERUNG ERREICHT!")


def calc_reweight(PDF1, PDF2, quarks, x_1, x_2, E):
    Q2 = 2 * x_1 * x_2 * (E ** 2)
    for i, q in enumerate(quarks["quark"]):
        if i == 0:
            sum_PDF1 = ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, np.array(PDF1.xfxQ2(q, x_1, Q2)) * np.array(PDF1.xfxQ2(-q, x_2, Q2))) + \
                    np.maximum(0, np.array(PDF1.xfxQ2(-q, x_1, Q2)) * np.array(PDF1.xfxQ2(q, x_2, Q2)))) / (x_1 * x_2))

            sum_PDF2 = ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, np.array(PDF2.xfxQ2(q, x_1, Q2)) * np.array(PDF2.xfxQ2(-q, x_2, Q2))) + \
                    np.maximum(0, np.array(PDF2.xfxQ2(-q, x_1, Q2)) * np.array(PDF2.xfxQ2(q, x_2, Q2)))) / (x_1 * x_2))
        else:
            sum_PDF1 += ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, np.array(PDF1.xfxQ2(q, x_1, Q2)) * np.array(PDF1.xfxQ2(-q, x_2, Q2))) + \
                    np.maximum(0, np.array(PDF1.xfxQ2(-q, x_1, Q2)) * np.array(PDF1.xfxQ2(q, x_2, Q2)))) / (x_1 * x_2))

            sum_PDF2 += ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, np.array(PDF2.xfxQ2(q, x_1, Q2)) * np.array(PDF2.xfxQ2(-q, x_2, Q2))) + \
                    np.maximum(0, np.array(PDF2.xfxQ2(-q, x_1, Q2)) * np.array(PDF2.xfxQ2(q, x_2, Q2)))) / (x_1 * x_2))

    reweight = sum_PDF1 / sum_PDF2

    return reweight

def calc_diff_WQ(PDF, quarks, x_1, x_2, eta, E):
    e = 0.30282212
    for i, q in enumerate(quarks["quark"]):
        if i==0:
            diff_WQ = (((quarks["charge"][q - 1]) ** 4 * e**4) / (192 * np.pi * x_1 * x_2 * E ** 2)) * \
                   ((np.maximum(np.array(PDF.xfxQ2(q, x_1, 2 * x_1 * x_2 * (E ** 2))) * np.array(PDF.xfxQ2(-q, x_2, 2 * x_1 * x_2 * (E ** 2))), 0) + np.maximum(
                       np.array(PDF.xfxQ2(-q, x_1, 2 * x_1 * x_2 * (E ** 2))) * np.array(PDF.xfxQ2(q, x_2, 2 * x_1 * x_2 * (E ** 2))), 0)) / (x_1 * x_2)) * \
                   (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)
        else:
            diff_WQ += (((quarks["charge"][q - 1]) ** 4 * e**4) / (192 * np.pi * x_1 * x_2 * E ** 2)) * \
                   ((np.maximum(np.array(PDF.xfxQ2(q, x_1, 2 * x_1 * x_2 * (E ** 2))) * np.array(PDF.xfxQ2(-q, x_2, 2 * x_1 * x_2 * (E ** 2))), 0) + np.maximum(
                       np.array(PDF.xfxQ2(-q, x_1, 2 * x_1 * x_2 * (E ** 2))) * np.array(PDF.xfxQ2(q, x_2, 2 * x_1 * x_2 * (E ** 2))), 0)) / (x_1 * x_2)) * \
                   (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)

    if diff_WQ.size == 1:
        diff_WQ = float(diff_WQ)
    return diff_WQ


def import_model_transformer(model_path):
    model = keras.models.load_model(filepath=model_path)
    config = pd.read_csv(model_path + "/config", index_col="property")
    config = config.transpose()
    transformer_config = ast.literal_eval(config["transformer_config"][0])
    transformer = label_transformation(config=transformer_config)

    return (model, transformer)

def pd_dataframe_to_tf_tensor(value):
    tensor = value.to_numpy()
    tensor = tf.convert_to_tensor(value=tensor)
    return tensor

def create_param_configs(pools, size, vary_multiple_parameters=True):
    checked_configs = list()
    if vary_multiple_parameters:
        while len(checked_configs) < size:
            config = []
            for param in pools:
                config.append(np.random.choice(pools[param]))
            config = tuple(config)
            checked_configs.append(config)
            checked_configs = list(set(checked_configs))
    else:
        #herausfinden, welche pools mehr als einen Parameter haben
        varying_params = []
        for param in pools:
            if len(pools[param]) > 1:
                varying_params.append(param)
        #Für jeden value in jedem variierenden pool eine config erstellen
        for varying_param in varying_params:
            for value in pools[varying_param]:
                config = []
                for param in pools:
                    #Wenn es der zu variierende param ist den value nehmen, ansonsten den standardwert 0
                    if param == varying_param:
                        config.append(value)
                    else:
                        config.append(pools[param][0])
                config.append(param)
                config = tuple(config)
                checked_configs.append(config)
                checked_configs = list(set(checked_configs))
    return checked_configs

def construct_name(config_as_dict, names_set):
    save_path = str()
    for param in config_as_dict:
        if param in names_set:
            if type(config_as_dict[param]) in {np.float64, np.int64, float, int, str, np.str_, np.bool_, bool, tuple, None}:
                save_path += str(param) + "_" + str(config_as_dict[param]) + "_"
            else:
                try:
                    save_path += config_as_dict[param].name + "_"
                except AttributeError:
                    try:
                        save_path += config_as_dict[param].__name__ + "_"
                    except AttributeError:
                        try:
                            save_path += config_as_dict[param]._name + "_"
                        except AttributeError:
                            pass

    return save_path

def construct_optimizer(optimizer, learning_rate=None, momentum=None, nesterov=None):
    if optimizer == str:
        optimizer = keras.optimizers.get({"class_name": optimizer, "config": {"lr": learning_rate}})
    else:
        try:
            optimizer = optimizer(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, clipvalue=2)
        except:
            try:
                optimizer = optimizer(learning_rate=learning_rate, momentum=momentum)
            except:
                try:
                    optimizer = optimizer(learning_rate=learning_rate, clipvalue=2)
                except:
                    optimizer = optimizer
    return optimizer

def load_model_and_transormer(model_path):
    model = keras.models.load_model(filepath=model_path)

    config = pd.read_csv(model_path + "/config", index_col="property")
    config = config.transpose()
    transformer_config = ast.literal_eval(config["transformer_config"][0])
    transformer = label_transformation(config=transformer_config)

    return (model, transformer)

def get_varying_value(features_pd):
    #überprüfen, ob es sich um 3d-data handelt
    plotting_data = 0
    keys = []
    for key in features_pd:
        value = features_pd[key][0]
        if not all(values == value for values in features_pd[key]):
            plotting_data += 1
            keys.append(key)
    return keys

def plot_model(features_pd, labels, predictions,  keys, save_path=None, losses=None,
               plot_losses=False, trans_to_pb=True, set_ylabel=None, set_ratio_yscale=None, autoscale_ratio=False):
    colors = ["C0", "C2", "C9", "C6", "deeppink"]
    linestyles = ["dashed", "dashdot", "dashed", "dashdot"]
    facecolors = ["C0", "None", "None", "None"]
    if len(keys) == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot the surface
        plot_labels = labels[:, 0]
        surf = ax.plot_trisurf(features_pd[keys[0]], features_pd[keys[1]], plot_labels,
                               cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel("rewait")
        ax.set_zscale("linear")
        plt.tight_layout()
        ax.view_init(10, 50)
        plt.show()

        # losses plotten
        plot_losses = losses
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # plot the surface
        surf = ax.plot_trisurf(features_pd[keys[0]], features_pd[keys[1]], losses,
                               cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel("reweight")
        ax.set_zscale("linear")
        plt.tight_layout()
        ax.view_init(10, 50)
        if save_path:
            plt.savefig(save_path + "_" + str(keys[0]) + "_" + str(keys[1]) + "_3d")
        plt.show()

        # Überprüfen, ob das feature konstant ist:
    if len(keys) == 1:
        for key in features_pd:
            value = features_pd[key][0]
            if not all(values == value for values in features_pd[key]):
                # Fkt plotten
                order = np.argsort(np.array(features_pd[key]), axis=0)
                print(order)
                print(features_pd[key])
                print(predictions)
                plot_features = np.array(features_pd[key])[order]
                plot_predictions = dict()
                for model_name in predictions:
                    print(predictions[model_name])
                    plot_predictions[model_name] = np.array(predictions[model_name])[order]
                    if trans_to_pb:
                        plot_predictions[model_name] = MC.gev_to_pb(plot_predictions[model_name])
                plot_labels = np.array(labels)[order]
                if trans_to_pb:
                    plot_labels = MC.gev_to_pb(plot_labels)
                fig, (ax_fct, ax_ratio) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [2.5,1]}, figsize=(6.4, 7.2))
                ax_fct.plot(plot_features, plot_labels, label="Analytic", linestyle="solid", color="C1")
                for i,model_name in enumerate(plot_predictions):
                    ax_fct.plot(plot_features, plot_predictions[model_name], label=model_name, linewidth=2, color=colors[i], linestyle=linestyles[i])
                s = ""
                log_factor = 1      #skalierungsfaktor falls logarithmische skala
                if features_pd.shape[1] == 3:
                    ylabel = r"$\frac{d^3\sigma}{d x_1 d x_2 d \eta} [pb]$"
                    if key == "x_1" or key == "x_2":
                        ax_fct.set_yscale("log")
                        log_factor=2
                        xlabel = "$" + key + "$"
                        s = (r"$x_1$ = " + "{:.2f}".format(features_pd["x_1"][0])) * (key != "x_1")\
                            + (r"$x_2$ = " + "{:.2f}".format(features_pd["x_2"][0])) * (key != "x_2")\
                            + "\n$\eta$ = " + "{:.2f}".format(features_pd["eta"][1])
                    elif key == "eta":
                        xlabel = "$\eta$"
                        s = r"$x_1$ = " + "{:.2f}".format(features_pd["x_1"][0]) + "\n$x_2$ = " + "{:.2f}".format(features_pd["x_2"][1])
                    elif key in {"theta", "Theta"}:
                        xlabel = r"$\theta$"
                else:
                    if key in {"theta", "Theta"}:
                        ylabel = r"$\frac{d \sigma}{d \theta}[pb]$"
                        xlabel = r"$\theta$"
                        pi_ticks = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                        pi_names = np.array(["0", r"$\frac{1}{4} \pi$", r"$\frac{1}{2} \pi$", r"$\frac{3}{4} \pi$", r"$\pi$"])
                        ax_fct.set_xticks(pi_ticks)
                        ax_fct.set_xticklabels(pi_names, fontdict={'fontsize': 20})
                    if key == "eta":
                        ylabel = r"$\frac{d \sigma}{d \eta}[pb]$"
                ax_fct.grid(True)
                if set_ylabel:
                    ylabel = set_ylabel
                ax_fct.set_ylabel(ylabel, loc="center", fontsize=15)
                #ax_fct.set_xlabel(xlabel)
                print(np.ptp(plot_labels))
                print(np.min(plot_labels))
                print(np.max(plot_labels))
                ax_fct.set_ylim((np.min(plot_labels)-0.05 * np.ptp(plot_labels), np.max(plot_labels) * 1.1 * log_factor)) # ylim so setzen dass Legende hereinpasst
                ax_fct.text(x=0.6, y=0.91, s=s, bbox=dict(boxstyle="round", facecolor="white", alpha=1, edgecolor="gainsboro"), transform=ax_fct.transAxes)
                ax_fct.legend(loc=(0.75, 0.89))
                # TODO plots checken ob das so passt mit der legendenpostion
                plt.tight_layout()
                #plt.show()

                #Ratios plotten
                ratios = dict()
                ratios_std = dict()
                for model_name in plot_predictions:
                    ratios[model_name] = plot_labels/plot_predictions[model_name]
                    #stddev der ratios berechnen, für skala
                    ratios_std[model_name] = np.std(ratios[model_name])
                print(ratios_std, np.array([*ratios_std.values()]))
                ratios_std = np.max(np.array([*ratios_std.values()]))
                print(ratios_std)

                for i,model_name in enumerate(ratios):
                    ax_ratio.scatter(plot_features, ratios[model_name], marker=".", s=20, linewidths=0.15, facecolors=facecolors[i], edgecolors=colors[i])
                s = "text fehlt"
                if key == "x_1" or key == "x_2":
                    xlabel = "$" + key + "$"
                    s = (r"$x_1$ = " + "{:.2f}".format(
                        features_pd["x_1"][0])) * (key != "x_1") \
                        + (r"$x_2$ = " + "{:.2f}".format(
                        features_pd["x_2"][0])) * (key != "x_2") \
                        + "\n$\eta$ = " + "{:.2f}".format(
                        features_pd["eta"][1])
                if key == "eta":
                    xlabel = "$\eta$"
                    if features_pd.shape[1] == 3:
                        s = r"$x_1$ = " + \
                            "{:.2f}".format(features_pd["x_1"][0]) + "\n$x_2$ = " + \
                            "{:.2f}".format(features_pd["x_2"][1])
                ax_ratio.yaxis.set_label_coords(-0.125,0.5)
                ax_ratio.grid(True)
                ax_ratio.set_ylabel(r"ratio", loc="center", rotation=90, fontsize=15)
                if ratios_std > 0.05:
                    ax_ratio.set_yscale("log")
                    ax_ratio.set_ylim(1-0.2, 1+0.2)
                    ax_ratio.set_yticks(np.array([0.96, 0.98, 1.00, 1.02, 1.04]))
                    ax_ratio.set_yticklabels(np.array([0.960, 0.980, 1.000, 1.020, 1.040]))
                else:
                    try:
                        ax_ratio.set_ylim(np.mean(ratios)-4*ratios_std, np.mean(ratios) + 4*ratios_std)
                    except:
                        ax_ratio.set_ylim(1 - 4 * ratios_std,
                                          1 + 4 * ratios_std)
                ax_ratio.set_xlabel(xlabel, fontsize=15)
                if set_ratio_yscale:
                    ax_ratio.set_yscale(set_ratio_yscale)
                if autoscale_ratio:
                    ax_ratio.autoscale()
                """
                ax_ratio.text(x=0.035, y=0.175, s=s,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=1,
                                  edgecolor="gainsboro"),
                        transform=ax_ratio.transAxes)                
                """
                plt.tight_layout()
                if save_path:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    plt.savefig(save_path + "_" + str(key) + "_ratio")
                plt.show()

                # losses plotten
                if plot_losses:
                    plot_losses = np.array(losses)[order]
                    plt.plot(plot_features, plot_losses)
                    plt.ylabel("Loss")
                    plt.xlabel(str(key))
                    plt.yscale("Log")
                    if save_path:
                        plt.savefig(save_path + "_" + str(key) + "_loss")
                    plt.show()

def scheduler(epoch, learning_rate, reduction = 0.1):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * (1-reduction)


def make_comparison_plot(names, losses, all_losses, avg_losses=None, losses_errors=None, save_path=None, comparison=None):
    x = list(np.arange(len(names)))
    print(x, len(x))
    print(all_losses, len(x))

    fig, ax = plt.subplots(figsize=(len(x) * 1.5, 4.8))
    for xe, ye in zip(x, all_losses):
        ax.scatter([xe] * len(ye), ye, marker=".", facecolors="None", edgecolors="C0")  # alle punkte plotten
    ax.scatter(x, losses, marker="o", color="orange", label="Min")  # niedrigster punkt hervorheben
    ax.errorbar(x, avg_losses, yerr=losses_errors,
                linewidth=0, elinewidth=1, color="C0",
                capsize=30 /np.sqrt(len(x)))  # errobars zeigen

    ax.set_ylabel("MAPE")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="lightgray")
    ax.set_xlim(x[0]-0.5, x[-1]+0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel(comparison)
    fig.legend()
    fig.tight_layout()
    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + str(comparison) + "_comparison")


def make_MC_plot(x, analytic_integral, ml_integral, xlabel=None, ylabel=None, save_path=None, name="", analytic_errors=None, ml_errors=None, scale="linear"):
    fig, (ax, ax_ratio) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [3,1], "hspace": 0.05}, figsize=(6.4, 7.2))
    # Integration plotten
    ax.step(x=x, y=analytic_integral,
                label="analytic", where="mid")
    ax.step(x=x, y=ml_integral, label="ML", where="mid", linestyle="dashed")
    if analytic_errors is not None:
        ax.errorbar(x=x, y=analytic_integral, yerr=analytic_errors, linewidth=0, elinewidth=2, capsize=100/len(x), ecolor="C0")
    if ml_errors is not None:
        ax.errorbar(x=x, y=ml_integral, yerr=ml_errors, linewidth=0, elinewidth=2, capsize=100/len(x), ecolor="orange")

    ax.set_ylabel(ylabel)
    ax.set_yscale(scale)
    ax.grid(True, color="lightgray")
    ax.legend()
    # ratio plotten
    ratio = analytic_integral/ml_integral
    ax_ratio.step(x=x, y=ratio, where="mid")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(1-0.05, 1+0.05)
    ax_ratio.grid(True, color="lightgray")
    ax_ratio.set_xlabel(xlabel)
    plt.tight_layout()
    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + name + "mc")


class class_scheduler():
    def __init__(self, reduction, offset=10, min_lr=0):
        self.reduction = reduction
        self.offset = offset
        self.min_lr = min_lr

    def __call__(self, epoch, learning_rate):
        print("learning rate: ", learning_rate)
        if epoch < self.offset:
            return learning_rate
        elif learning_rate <= self.min_lr:
            return self.min_lr
        else:
            return learning_rate * (1-self.reduction)

if __name__ == "__main__":
    print("Achtung, dieses Skript ist geschrieben um importiert zu werden")

"""
test_model = DNN()
x = tf.constant([[1]], dtype="float32")
print(float(test_model(x)))
"""
