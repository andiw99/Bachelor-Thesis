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
x_total = int(5000000) #Anzahl an x Werten
eta_total = int(5000000) # Anzahl an eta Werten
x_lower_limit = 0
x_upper_limit = 0.8
eta_limit = 3
loguni_param=0.01
stddev = 1.5
xMin = PDF.xMin
np.random.seed(10)

set_name = "NewRandom/"
root_name ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject" 
location = input("Welcher Rechner?")
if location == "Taurus" or location == "taurus":
    root_name = "/home/s1388135/Bachelor-Thesis"
path = root_name + "/Files/Transfer/Data/" + set_name

#Test
total_data = x_total**2 * eta_total
total_data = x_total**2 * eta_total
print("Wir berechnen ", total_data, "Werte")


#Listen mit Funktionswerten anlegen
#Ausgangswerte
diff_WQ_list = []
diff_WQ_list_eta_x_2_constant=[]
diff_WQ_list_eta_x_1_constant = []
diff_WQ_list_x_constant = []
diff_WQ_list_3D = []
#Eingangswerte
x_1_list = []
x_2_list = []
eta_list = []
eta_list_x_constant = []
eta_list_3D = []
x_1_list_eta_x_1_constant = []
x_2_list_eta_x_1_constant = []
x_1_list_eta_x_2_constant = []
x_2_list_eta_x_2_constant = []
x_1_list_x_constant = []
x_2_list_x_constant = []
x_1_list_3D = []
x_2_list_3D = []
eta_list_eta_x_1_constant = []
eta_list_eta_x_2_constant = []
x_2_list_eta_constant = []
step = 0

#diff WQ berechnen und Listen füllen

x_1_Rand = (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_1_Rand = x_1_Rand[x_1_Rand >= xMin]
plt.hist(x_1_Rand, bins=20, rwidth=0.8)
plt.yscale("linear")
plt.show()
x_2_Rand = (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_2_Rand = x_2_Rand[x_2_Rand >= xMin]
eta_Rand_upper=np.array([])
eta_Rand_lower = np.array([])
eta_Rand = np.array([])
while eta_Rand.size < eta_total:
    eta_Rand_upper  = np.append(eta_Rand_upper, -abs(stats.norm.rvs(scale=stddev, size=int((eta_total/2))-eta_Rand_upper.size)) + eta_limit)
    eta_Rand_upper = eta_Rand_upper[eta_Rand_upper >= 0]
    eta_Rand_lower = np.append(eta_Rand_lower, abs(stats.norm.rvs(scale=stddev , size = int((eta_total/2))- eta_Rand_lower.size)) - eta_limit)
    eta_Rand_lower = eta_Rand_lower[eta_Rand_lower <= 0]
    eta_Rand = np.append(eta_Rand_upper, eta_Rand_lower)
plt.hist(eta_Rand, bins=20, rwidth=0.8)
plt.show()

print(x_1_Rand.min())

print(eta_Rand.size)

#Feste werte setzen
i=0
x_1_constant=0
x_2_constant=0
eta_constant = 0
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

while eta_constant <0.5 or eta_constant > 2.0:
    if i >= eta_total:
        eta_constant = eta_Rand[i]
        break
    eta_constant = eta_Rand[i]
    i += 1

print(x_1_constant)
print(x_2_constant)
print(eta_constant)
print(np.min(x_1_Rand))
print(xMin)

diff_WQ = ml.calc_diff_WQ(PDF=PDF, quarks=quarks, x_1=x_1_Rand, x_2=x_2_Rand, eta=eta_Rand, E=E)

print(diff_WQ)


hadronic_diff_WQ_data = pd.DataFrame(
    {
        "x_1": x_1_Rand,
        "x_2": x_2_Rand,
        "eta": eta_Rand,
        "WQ": diff_WQ
    }
)


config = pd.DataFrame(
    {
        "total_data": total_data,
        "x_lower_limit": x_lower_limit,
        "x_upper_limit": x_upper_limit,
        "eta_limit": eta_limit,
        "loguni_param": loguni_param,
        "stddev": stddev,

    },
    index=[0]
)

x_constant_name =  "x_constant"
eta_x_2_constant_name =  "eta_x_2_constant"
eta_x_1_constant_name =  "eta_x_1_constant"
x_2_constant_name = "x_2_constant__3D"

#ggf. Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

hadronic_diff_WQ_data.to_csv(path + "all", index=False)

config.to_csv(path + "config", index=False)

print(hadronic_diff_WQ_data)








