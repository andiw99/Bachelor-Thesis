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
import ml

quarks = {"quark": [1, 2, 3, 4],
          "charge": [-1/3,2/3, -1/3, 2/3]}
PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF = pdf.mkPDF("CT14nnlo", 0)
#PDF = pdf.mkPDF("MMHT2014nnlo68cl", 0)

#for q in quarks["quark"]:
    #print("Quark ", PID[q], "hat Ladung ", quarks["charge"][q-1])

#Variablen
e = 1.602e-19
E = 6500 #Strahlenergie in GeV, im Vornherein festgelegt?
x_total = int(750000) #Anzahl an x Werten
eta_total = int(750000) # Anzahl an eta Werten
x_lower_limit = 0.01
x_upper_limit = 0.8
eta_limit = 3
loguni_param=0.01
stddev = 1.5
xMin = PDF.xMin
eta_constant = False
eta_values = 200

set_name = "NewRandom/"
root_name ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject" 
location = input("Welcher Rechner?")
if location == "Taurus" or location == "taurus":
    root_name = "/home/s1388135/Bachelor-Thesis"
path = root_name + "/Files/Transfer/Data/" + set_name



#Werte erzeugen
x_1 = (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_1 = x_1[x_1 >= xMin]
plt.hist(x_1, bins=20, rwidth=0.8)
>>>>>>> ca3a8dff4e562631d6d62aefc65d3242f65effce
plt.yscale("linear")
plt.show()
x_2 = (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_2 = x_2[x_2 >= xMin]

eta = np.array([], dtype="float32")
if eta_constant:
    x_1_Rand = x_1
    x_2_Rand = x_2
    x_1 = np.array([], dtype="float32")
    x_2 = np.array([], dtype="float32")
    for eta_value in np.linspace(-eta_limit, eta_limit, num=eta_values):
        eta_frac = np.zeros(shape=(int(eta_total), ), dtype="float32")
        eta_frac += eta_value
        eta = np.concatenate((eta, eta_frac))
        x_1 = np.concatenate((x_1, x_1_Rand))
        x_2 = np.concatenate((x_2, x_2_Rand))

else:
    eta_upper=np.array([])
    eta_lower = np.array([])
    while eta.size < eta_total:
        eta_upper  = np.append(eta_upper, -abs(stats.norm.rvs(scale=stddev, size=int((eta_total/2))-eta_upper.size)) + eta_limit)
        eta_upper = eta_upper[eta_upper >= 0]
        eta_lower = np.append(eta_lower, abs(stats.norm.rvs(scale=stddev , size = int((eta_total/2))- eta_lower.size)) - eta_limit)
        eta_lower = eta_lower[eta_lower <= 0]
        eta = np.append(eta_upper, eta_lower)
    plt.hist(eta, bins=20, rwidth=0.8)
    plt.show()

print(x_1.size)
print(eta.size)
print(x_1.shape)
print(eta.shape)
print(x_1)
print(eta)
print(type(eta))
print(type(x_1))


diff_WQ = ml.calc_diff_WQ(PDF=PDF, quarks=quarks, x_1=x_1, x_2=x_2, eta=eta, E=E)


hadronic_diff_WQ_data = pd.DataFrame(
    {
        "x_1": x_1,
        "x_2": x_2,
        "eta": eta,
        "WQ": diff_WQ
    }
)


config = pd.DataFrame(
    {
        "total_data": x_total,
        "x_lower_limit": x_lower_limit,
        "x_upper_limit": x_upper_limit,
        "eta_limit": eta_limit,
        "loguni_param": loguni_param,
        "stddev": stddev,

    },
    index=[0]
)


#ggf. Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

hadronic_diff_WQ_data.to_csv(path + "all", index=False)

config.to_csv(path + "config", index=False)

print(hadronic_diff_WQ_data)








