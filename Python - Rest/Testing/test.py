import tensorflow as tf
from tensorflow import keras
#prepare Dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(16)
#ich möchte die Struktur meines Datensets untersuchen
print(dataset)
for step, (x,y) in enumerate(dataset):
    print("step", step, "x:", x, "y:", y)
    if step >= 10:
        break

class Linear(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        print("input_shape:", input_shape)
        print("input_shape[-1]", input_shape[-1])
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        print("weights:", self.w)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MLP(keras.layers.Layer):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(64)  #wird hier die Klasse von oben verwendet? muss eigentlich oder?
        self.linear_2 = Linear(64)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)   #hier macht der erste Layer seine Arbeit
        x = tf.nn.relu(x)   #ich muss meine Ausgabe normieren, soweit ich weiß
        x = self.linear_2(x) #Layer 2 macht seine Arbeit
        x = tf.nn.relu(x)
        return self.linear_3(x)

model = MLP()
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
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss

for step, (x,y) in enumerate(dataset):
    loss = train_on_batch(x,y)
    if step % 100 == 0:
        print("Step", step, "Loss:", loss)