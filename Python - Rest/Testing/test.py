import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
import ml
import time
import os
import ast


set_name = "NewRandom/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/" + set_name
(data, features, labels, _, _, _) = ml.data_handling(data_path=path + "all", label_name="WQ")
print("min ", tf.reduce_min(labels))
print("max ", tf.reduce_max(labels))
print("mean ", tf.reduce_mean(labels))
print("stddev ", tf.math.reduce_std(labels))

(data, features, labels, _, _, transformer) = ml.data_handling(data_path=path + "all", label_name="WQ", shift=True, logarithm=True)

print("min ", tf.reduce_min(labels))
print("max ", tf.reduce_max(labels))
print("mean ", tf.reduce_mean(labels))
print("stddev ", tf.math.reduce_std(labels))

labels = transformer.retransform(labels)

print("min ", tf.reduce_min(labels))
print("max ", tf.reduce_max(labels))
print("mean ", tf.reduce_mean(labels))
print("stddev ", tf.math.reduce_std(labels))

(data, features, labels, _, _, transformer) = ml.data_handling(data_path=path + "all", label_name="WQ", shift=True, logarithm=True, base10 = True)

print("min ", tf.reduce_min(labels))
print("max ", tf.reduce_max(labels))
print("mean ", tf.reduce_mean(labels))
print("stddev ", tf.math.reduce_std(labels))