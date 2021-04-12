import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
tf.debugging.set_log_device_placement(True)

a = tf.constant([[2, 3], [4, 5]])
print("a:", a)

b = -a
print("b:", b)

c = tf.add(a, b)
print("c:", c)

d = tf.square(b)
print("d:", d)
print(d[0][0])
"""
diff_WQ_eta_data_raw = pd.read_csv("diff_WQ_eta_data")
data = tf.constant([diff_WQ_eta_data_raw["Eta"]], shape=(len(diff_WQ_eta_data_raw["Eta"]), 1))
print("data:", data)
"""

print(tf.config.experimental.list_physical_devices('GPU'))
print("Anzahl GPUs erkannt:", len(tf.config.experimental.list_physical_devices('GPU')))


# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
test = "auch mit variablen?"

plot_script = open("plotscript.py", "w")
plot_script.write("#Hier jetzt einfach der Code zum plotten rein?\n"
                  "print(\"test\")\n"
                  "print(\"" + test + "\")")
plot_script.close()
