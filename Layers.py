from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.eager import monitoring
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import control_flow_util

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

class LabelTransformation():
    def __init__(self, train_labels=None, scaling=False, logarithm=False, shift=False, label_normalization=False, config=None):
        self.scaling = scaling
        self.logarithm = logarithm
        self.shift = shift
        self.label_normalization = label_normalization
        self.values = dict()
        if train_labels is not None:
            if self.scaling:
                self.scale = 1/tf.math.reduce_min(train_labels)
                self.values["scale"] = self.scale
                train_labels = train_labels * self.scale
            if logarithm:
                train_labels = tf.math.log(train_labels)
            if self.shift:
                self.shift_value = tf.math.reduce_min(train_labels)
                self.values["shift_value"] = self.shift_value
                train_labels = train_labels - self.shift_value
            if self.label_normalization:
                self.normalization_value = tf.math.reduce_max(train_labels)
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



    def transform(self, x):
        if self.scaling:
            x = self.scale * x
        if self.logarithm:
            x = tf.math.log(x)
        if self.shift:
            x = x - self.shift_value
        if self.label_normalization:
            x = x/self.normalization_value
        return x

    def retransform(self, x):
        if self.label_normalization:
            x = x * self.normalization_value
        if self.shift:
            x = x + self.shift_value
        if self.logarithm:
            x = tf.math.exp(x)
        if self.scaling:
            x = (1/self.scale) * x
        return x

    def get_config(self):
        config = {
            "scaling": self.scaling,
            "logarithm": self.logarithm,
            "shift": self.shift,
            "label_normalization": self.label_normalization,

        }
        for transformation, value in self.values.items():
            config[transformation] = float(value)
        return config


#Dropout class stammt aus der tensorflow doku
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




if __name__ == "__main__":
    print("Achtung, dieses Skript ist geschrieben um importiert zu werden")

"""
test_model = DNN()
x = tf.constant([[1]], dtype="float32")
print(float(test_model(x)))
"""