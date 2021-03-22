import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


a = tf.constant([[2, 3], [4, 5]])
print("a:", a)

b = -a
print("b:", b)

c = tf.add(a, b)
print("c:", c)

d = tf.square(b)
print("d:", d)
print(d[0][0])

diff_WQ_eta_data_raw = pd.read_csv("diff_WQ_eta_data")
data = tf.constant([diff_WQ_eta_data_raw["Eta"]], shape=(len(diff_WQ_eta_data_raw["Eta"]), 1))
print("data:", data)