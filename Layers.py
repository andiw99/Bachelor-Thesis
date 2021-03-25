from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras



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

    def call(self, inputs):
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
    #im def oder im call läuft noch irgendetwas falsch, sollte eigentlich gleich sein wie MLP
    def __init__(self, nr_hidden_layers=3, units=64, outputs=1, kernel_regularization=None, bias_regularization=None):
        super(DNN, self).__init__()
        self.units = units
        self.nr_hidden_layers = nr_hidden_layers
        self.hidden_layers = []
        self.kernel_regularization = kernel_regularization
        self.bias_regularization = bias_regularization
        for i in range(self.nr_hidden_layers):
            name = "Layer_" + str(i)
            self.hidden_layers.append(Linear(units=self.units, name=name, kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization))
        #self.hidden_layers = [Linear(units=self.units, name="hidden_layer") in range(self.nr_hidden_layers)]
        self.linear_output = Linear(units=outputs, name="output_layer", kernel_regularization=self.kernel_regularization, bias_regularization=self.bias_regularization)


    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = tf.nn.relu(x)
        return self.linear_output(x)

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




if __name__ == "__main__":
    print("Achtung, dieses Skript ist geschrieben um importiert zu werden")

"""
test_model = DNN()
x = tf.constant([[1]], dtype="float32")
print(float(test_model(x)))
"""