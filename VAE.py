from tensorflow.keras import layers
import tensorflow as tf

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    """weist einer MNIST Zahl ein triplet (z_mean, z_log_var, z) zu
    Decodiert prinzipiell die große Dimension an Anfangsinformationen auf das wesentliche"""
    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    """
    entschlüsselt den im Encoder verschlüsselten Vektor und weißt diesem wieder eine Zahl zu
    """
    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu) #Für was steht proj? Einfach der erste Layer? project? auf was?
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x=self.dense_proj(inputs)
        return self.dense_output(x)
    
#Klassenbausteine werden zum VariationalAutoEncoder zusammengebaut und divergence regularization loss hinzugefügt
class VariationalAutoEncoder(layers.Layer):
    """end to end model"""
    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruced = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1
        )
        self.add_loss((kl_loss))
        return reconstruced

#Training-Loop
#Instanz der Klasse
vae = VariationalAutoEncoder(original_dim=784, intermediate_dim=64, latent_dim=32)

#loss and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#fucking dataset preparen, versteh ich gar nicht
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.shuffle(buffer_size=1024).batch(32)

#@tf.function für mehr power im Trainingsloop
@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        reconstructed = vae(x)

        loss = loss_fn(x, reconstructed)
        loss += sum(vae.losses)

    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return loss

losses=[]
for epoch in range(5):
    for step, x in enumerate(dataset):
        loss = training_step(x)

        losses.append(float(loss))

        if step%50 == 0:
            print("Epoch:", epoch, "Step:", step, "Loss:", sum(losses)/len(losses))

        if step >= 10000:
            break


