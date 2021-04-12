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
x_total = int(200) #Anzahl an x Werten
eta_total = int(200) # Anzahl an eta Werten
x_lower_limit = 0
x_upper_limit = 0.65
eta_limit = 3
loguni_param=0.005
stddev = 1.5
xMin = PDF.xMin
np.random.seed(10)

set_name = "TestData/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/" + set_name

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


for x_1 in x_1_Rand:
    if x_1 < xMin:
        x_1 = xMin
    for x_2 in x_2_Rand:
        if x_2 < xMin:
            x_2 = xMin
        for eta in eta_Rand:
            diff_WQ = 0
            # Viererimpuls initialisieren
            Q2 = 2 * x_1 * x_2 * (E ** 2)

            # Listen der Eingangswerte füllen:
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            eta_list.append(eta)
            # WQ berechnen, Summe über Quarks:
            diff_WQ = ml.calc_diff_WQ(PDF=PDF, quarks=quarks, x_1=x_1, x_2=x_2, eta=eta, E=E)
            # diff WQ in Liste einfügen:
            diff_WQ_list.append(diff_WQ)

            # Listet mit konstantem x anlegen
            if x_1 == x_1_constant and x_2 == x_2_constant:
                eta_list_x_constant.append(eta)
                diff_WQ_list_x_constant.append(diff_WQ)
                x_1_list_x_constant.append(x_1)
                x_2_list_x_constant.append(x_2)

            # Listen mit konstantem eta, konstantem x_2 anlegen
            if eta == eta_constant and x_2 == x_2_constant:
                x_1_list_eta_x_2_constant.append(x_1)
                diff_WQ_list_eta_x_2_constant.append(diff_WQ)
                eta_list_eta_x_2_constant.append(eta)
                x_2_list_eta_x_2_constant.append(x_2)

            #Listen mit konstantem eta, konstantem x_1 anlegen
            if eta == eta_constant and x_1 == x_1_constant:
                x_1_list_eta_x_1_constant.append(x_1)
                x_2_list_eta_x_1_constant.append(x_2)
                eta_list_eta_x_1_constant.append(eta)
                diff_WQ_list_eta_x_1_constant.append(diff_WQ)

            # Liste für den 3D-Plot erstellen
            if x_2 == x_2_constant:
                x_1_list_3D.append(x_1)
                x_2_list_3D.append(x_2)
                eta_list_3D.append(eta)
                diff_WQ_list_3D.append(diff_WQ)

            step += 1
            if step % 50000 == 0:
                print("x_1:", x_1, "x_2:", x_2, "eta:", eta, "diff_WQ:", diff_WQ)


hadronic_diff_WQ_data = pd.DataFrame(
    {
        "x_1": x_1_list,
        "x_2": x_2_list,
        "eta": eta_list,
        "WQ": diff_WQ_list
    }
)
hadronic_diff_WQ_data_x_constant = pd.DataFrame(
    {
        "x_1": x_1_list_x_constant,
        "x_2": x_2_list_x_constant,
        "eta": eta_list_x_constant,
        "WQ": diff_WQ_list_x_constant
    }
)


hadronic_diff_WQ_data_eta_x_2_constant = pd.DataFrame(
    {
        "x_1": x_1_list_eta_x_2_constant,
        "x_2": x_2_list_eta_x_2_constant,
        "eta": eta_list_eta_x_2_constant,
        "WQ": diff_WQ_list_eta_x_2_constant
    }
)

hadronic_diff_WQ_data_eta_x_1_constant = pd.DataFrame(
    {
        "x_1": x_1_list_eta_x_1_constant,
        "x_2": x_2_list_eta_x_1_constant,
        "eta": eta_list_eta_x_1_constant,
        "WQ": diff_WQ_list_eta_x_1_constant
    }
)

hadronic_diff_WQ_data_x_2_constant = pd.DataFrame(
    {
        "x_1": x_1_list_3D,
        "x_2": x_2_list_3D,
        "eta": eta_list_3D,
        "WQ": diff_WQ_list_3D
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
hadronic_diff_WQ_data_x_constant.to_csv(path + x_constant_name, index=False)
hadronic_diff_WQ_data_eta_x_2_constant.to_csv(path + eta_x_2_constant_name, index=False)
hadronic_diff_WQ_data_eta_x_1_constant.to_csv(path + eta_x_1_constant_name, index=False)
hadronic_diff_WQ_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)
config.to_csv(path + "config", index=False)

print(hadronic_diff_WQ_data)
print(hadronic_diff_WQ_data_x_constant)
print(hadronic_diff_WQ_data_eta_x_2_constant)
print(hadronic_diff_WQ_data_eta_x_1_constant)







