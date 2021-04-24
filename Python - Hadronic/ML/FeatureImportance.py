import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import ml
import ast

#Pfade einlesen
data_path = "/Files/Transfer/Data/WithoutCutWrongeValue/CT14nnlo/all"
model_path = "/Files/Transfer/Models/Old/best_model"

#model und transformer laden
(model, transformer) = ml.import_model_transformer(model_path=model_path)
loss_fn = keras.losses.MeanAbsoluteError()

#Data einlesen
(training_data, train_features, train_labels, test_features, test_labels, train_features_pd, train_labels_pd,
 data_transformer) = ml.data_handling(data_path=data_path, label_name="WQ", return_pd=True, train_frac=0.02)

#Error für Predictions berechnen:
predictions = transformer.retransform(model(train_features, training=False))
true_loss = loss_fn(y_true=train_labels, y_pred=predictions)

#Permutation erschafen
perm_features = dict()
for key in train_features_pd:
    perm_features[key] = train_features_pd.copy()
    perm_features[key].loc[:,key] = list(train_features_pd[key].sample(frac=1))
print(train_features_pd)
print(perm_features)

print(float(true_loss))
#perm-losses für jedes Feature berechnen
perm_predictions = dict()
perm_losses = dict()
for key in perm_features:
    perm_features[key] = ml.pd_dataframe_to_tf_tensor(perm_features[key])
    perm_predictions[key] = transformer.retransform(model(perm_features[key]))
    perm_losses[key] = float(loss_fn(y_true=train_labels, y_pred=perm_predictions[key]))

#Faktoren berechnen
frac_perm_loss = perm_losses.copy()
for key in frac_perm_loss:
    print(frac_perm_loss[key])
    frac_perm_loss[key] = frac_perm_loss[key]/float(true_loss)

#Plotten
names = list(frac_perm_loss.keys())
frac_losses = list(frac_perm_loss.values())
abs_losses = list(perm_losses.values())

x=np.arange(len(names))
width=0.30

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
rects1 = ax1.bar(x-width/2, frac_losses, width, label="frac losses")
rects2 = ax2.bar(x+width/2, abs_losses, width, label="abs losses")

ax1.set_title("Influence of feature permutation on loss")
ax1.set_ylabel("frac losses")
ax2.set_ylabel("abs losses")
ax1.set_xticks(x)
ax2.set_xticks(x)
ax1.set_xticklabels(names)
#fig.legend()
fig.tight_layout()
plt.show()




