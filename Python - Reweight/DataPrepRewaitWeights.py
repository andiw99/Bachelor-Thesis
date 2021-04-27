import random
import lhapdf as pdf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats

#quarks = {"quark": [1, 2, 3, 4],
#          "charge": [-1/3,2/3,-1/3,2/3]}
#PID = {1: "d", 2: "u", 3: "s", 4: "c"}
import MC
import ml

quarks = {"quark": [1, 2, 3, 4],
          "charge": [-1/3,2/3, -1/3, 2/3]}
#quarks = {"quark": [1, 2],
 #         "charge": [-1/3,2/3]}

PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF_CT14 = pdf.mkPDF("CT14nnlo", 0)
PDF_MMHT = pdf.mkPDF("MMHT2014nnlo68cl", 0)

#Variablen
E = 6500 #Strahlenergie in GeV, im Vornherein festgelegt?
x_total = int(500000) #Anzahl an x Werten
x_lower_limit = 0.0
x_upper_limit = 0.8
loguni_param=0.01
loguni = False
use_cut = True

set_name = "TrainingData500k_cut_x_08/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/" + set_name

#Test
print("Wir berechnen ", x_total**2, "Werte")


#Werte generieren
#x_1_Rand = (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_1_Rand = stats.uniform.rvs(loc = x_lower_limit, scale = x_upper_limit-x_lower_limit ,size = x_total)
x_1_Rand = x_1_Rand[x_1_Rand >= PDF_CT14.xMin]
plt.hist(x_1_Rand, bins=20, rwidth=0.8)
plt.yscale("linear")
plt.show()
#x_2_Rand = (stats.loguniform.rvs(a=loguni_param, b=1 + loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_2_Rand = stats.uniform.rvs(loc = x_lower_limit, scale = x_upper_limit-x_lower_limit ,size = x_total)
x_2_Rand = x_2_Rand[x_2_Rand >= PDF_CT14.xMin]


print(x_1_Rand.min())


#Feste werte setzen
i=0
x_1_constant=0
x_2_constant=0
while x_1_constant < 0.05 or x_1_constant > 0.15:
    if i >= x_total:
        x_1_constant = x_1_Rand[i-1]
        break
    x_1_constant = x_1_Rand[i]
    i += 1
i=0
while x_2_constant < 0.05 or x_2_constant > 0.15:
    if i >= x_total:
        x_2_constant = x_2_Rand[i-1]
        break
    x_2_constant = x_2_Rand[i]
    i += 1
i=0

#konstante arrays erstellen
x_1_const = np.zeros(shape=x_total) + x_1_constant
x_2_const = np.zeros(shape=x_total) + x_2_constant

print(x_1_const, type(x_1_const))
print(x_2_const, type(x_2_const))

print("x_1_Rand", x_1_Rand, type(x_1_Rand))
print("x_2_Rand", x_2_Rand, type(x_2_Rand))

print(x_1_constant)
print(x_2_constant)
print(np.min(x_1_Rand))
xMin = PDF_CT14.xMin

features = np.stack((x_1_Rand, x_2_Rand))
features = np.transpose(features)
features_x_1_constant = np.stack((x_1_const, x_2_Rand))
features_x_1_constant = np.transpose(features_x_1_constant)
features_x_2_constant = np.stack((x_1_Rand, x_2_const))
features_x_2_constant = np.transpose(features_x_2_constant)

if use_cut:
    features = MC.reweight_cut(features)
    features_x_1_constant = MC.reweight_cut(features_x_1_constant)
    features_x_2_constant = MC.reweight_cut(features_x_2_constant)

reweights_all = ml.calc_reweight(PDF1=PDF_CT14, PDF2=PDF_MMHT, quarks=quarks, x_1=features[:,0], x_2=features[:,1], E=E)
reweights_x_2_constant = ml.calc_reweight(PDF1=PDF_CT14, PDF2=PDF_MMHT, quarks=quarks, x_1=features_x_2_constant[:,0], x_2=features_x_2_constant[:,1], E=E)
reweights_x_1_constant = ml.calc_reweight(PDF1=PDF_CT14, PDF2=PDF_MMHT, quarks=quarks, x_1=features_x_1_constant[:,0], x_2=features_x_1_constant[:,1], E=E)

hadronic_reweight_data = pd.DataFrame(
    {
        "x_1": features[:,0],
        "x_2": features[:,1],
        "reweight": reweights_all
    }
)


hadronic_reweight_data_x_2_constant = pd.DataFrame(
    {
        "x_1": features_x_2_constant[:,0],
        "x_2": features_x_2_constant[:,1],
        "reweight": reweights_x_2_constant
    }
)

hadronic_reweight_data_x_1_constant = pd.DataFrame(
    {
        "x_1": features_x_1_constant[:,0],
        "x_2": features_x_1_constant[:,1],
        "reweight": reweights_x_1_constant,
    }
)

config = pd.DataFrame(
    {
        "x_total": len(reweights_all),
        "x_lower_limit": x_lower_limit,
        "x_upper_limit": x_upper_limit,
        "loguni_param": loguni_param,
        "loguni": loguni,
        "use_cut": use_cut,
    },
    index=[0]
)
x_2_constant_name = "x_2_constant"
x_1_constant_name = "x_1_constant"

#ggf. Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

hadronic_reweight_data.to_csv(path + "all", index=False)
hadronic_reweight_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)
hadronic_reweight_data_x_1_constant.to_csv(path + x_1_constant_name, index=False)
config.to_csv(path + "config", index=False)


print(hadronic_reweight_data)
print(hadronic_reweight_data_x_2_constant)
print(hadronic_reweight_data_x_1_constant)