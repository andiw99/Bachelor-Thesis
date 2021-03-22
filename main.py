import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        #print(input_shape)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        #print("weights:", self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

print(dataset)
for step, (x,y) in enumerate(dataset):
    print("step", step, "x:", x, "y:", y)
    if step >= 100:
        break

# Instantiate our layer.
linear_layer = Linear(units=10)
#loss function


# The layer can be treated as a function.
# Here we call it on some data.

class ComputeSum(keras.layers.Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        #Dieses Gewicht ist non-Trainable
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

#kreation klassinstanz der Klasse computesum
my_sum = ComputeSum(input_dim=2)      #Warum 2? Input_dim!

x = tf.ones((2,2))

y = my_sum(x) #hier addiere ich auf das inital_value, eine 0 Matrix meinen x Tensor.
#print(y.numpy())

y = my_sum(x) #ich addiere nochmal x Tensor, alter Wert ist warum auch immer in der Klassinstanz gespeichert?
#print(y.numpy())

#self.total ist Variable der bei jeder addition der neue Tensor addiert wird.
#Gewicht des Layers ist die Summe?

class MLP(keras.layers.Layer):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)  #wird hier die Klasse von oben verwendet? muss eigentlich oder?
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)   #hier macht der erste Layer seine Arbeit
        x = tf.nn.relu(x)   #ich muss meine Ausgabe normieren, soweit ich weiß
        x = self.linear_2(x) #Layer 2 macht seine Arbeit
        x = tf.nn.relu(x)
        return self.linear_3(x)

#Zum losses Tracken eigene klasse

class ActivityRegularization(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))    #Was genau macht add_loss und reduce_sum? reduce sum addiert alle einträge des tensors
        return inputs

class SparseMLP(keras.layers.Layer):
    def __init__(self):
        super(SparseMLP, self).__init__()
        self.linear_1 = Linear(32)
        self.regularization = ActivityRegularization(1e-2)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.regularization(x)
        return self.linear_3(x)

"""
    for step, (x,y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            #ergebnisse berechnen, gleiche dimension wie units?
            logits = linear_layer(x)
            #loss berechnen
            loss = loss_fn(y, logits)
    
            loss += sum(mlp.losses)
    
            gradients = tape.gradient(loss, linear_layer.trainable_weights)
        #update weigths
        optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))
        if step % 50 == 0:
            print("Step:", step, "Loss:", float(loss))
"""
#metric
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_on_batch(x,y):
    with tf.GradientTape() as tape:
        logits = model(x)
        print("logits:", logits)
        print("y:", y)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    #update accuracy
    accuracy.update_state(y, logits)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss

#Training?
"""
    for epoch in range(2):
        for step, (x,y) in enumerate(dataset):
            loss = train_on_batch(x,y)
            if step % 100 == 0:
                print("Epoch", epoch, "Step:", step, "Loss:", float(loss))
                print("Total running accuracy so far: %.3f" % accuracy.result())
    
        accuracy.reset_states()
"""

class Dropout(keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

class MLPWithDropout(keras.layers.Layer):
    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training = None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)

mlp = MLPWithDropout()
y_train = mlp(tf.ones((2,2)), training=True)
y_test = mlp(tf.ones((2,2)), training = False)

print(y_train)
print(y_test)

x = tf.ones(shape=(1,64))
#print("x:", x)
print("y:", y)

print(mlp.losses)

