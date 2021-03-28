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

    def __init__(self, units=32, name="Linear_layer", kernel_regularization=None, bias_regularization=None):
        super(Linear, self).__init__()
        self.units = units
        self.weight_name = name
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization

    def build(self, input_shape):
        #print("input shape in Layer:", self.weight_name, input_shape)
        #print("input shape[-1]:", input_shape[-1])

        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="HeNormal",
            trainable=True,
            name= self.weight_name,
        )
        #print("weights von:", self.weight_name, self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True,
            name = self.weight_name,
        )

    def call(self, inputs, training=True):
        output = tf.matmul(inputs, self.w) + self.b
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

        for i in range(self.nr_hidden_layers):
            name = "Layer_" + str(i)
            dropout_name = "Dropout_" + str(i)
            self.names.add(name)
            if dropout:
                self.hidden_layers.append(Dropout(rate=self.dropout_rate, name=dropout_name))
            self.hidden_layers.append(Linear(units=self.units, name=name, kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization))
        #self.hidden_layers = [Linear(units=self.units, name="hidden_layer") in range(self.nr_hidden_layers)]
        self.linear_output = Linear(units=outputs, name="output_layer", kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization)
        print("names:", self.names)


    def call(self, inputs, training=True):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            if layer.weight_name in self.names:
                x = tf.nn.relu(x)
        return self.linear_output(x, training=training)

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
                "bias_regularization": self.bias_regularization
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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