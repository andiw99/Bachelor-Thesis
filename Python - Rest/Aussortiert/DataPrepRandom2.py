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
quarks = {"quark": [1, 2, 3, 4],
          "charge": [-1/3,2/3, -1/3, 2/3]}
PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF = pdf.mkPDF("PlottingData_CT14nnlo", 0)

#for q in quarks["quark"]:
    #print("Quark ", PID[q], "hat Ladung ", quarks["charge"][q-1])

#Variablen
e = 1.602e-19
E = 10 #Strahlenergie in GeV, im Vornherein festgelegt?
x_total = int(50) #Anzahl an x Werten
eta_total = int(500) # Anzahl an eta Werten
x_lower_limit = 0
x_upper_limit = 1
eta_limit = 3
loguni_param=0.005

set_name = "RandomSamples/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/" + set_name

name = "Loguniform+Gauss+Linear"
#Test
print("Wir berechnen ", x_total**2 * eta_total, "Werte")

#Step-sizes herausfinden, um anständig die anderen Listen zu generieren
(_, x_step) = np.linspace(start=np.log10(x_lower_limit+1), stop=np.log10(x_upper_limit+1), num=x_total, retstep=True)
(_, eta_step) = np.linspace(start=-eta_limit, stop=eta_limit, num=eta_total, retstep=True)
log_x_step = 10 ** x_step


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

x_1_Rand = stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=x_total) * (x_upper_limit - x_lower_limit) + x_lower_limit
x_1_Rand = x_1_Rand[x_1_Rand >= PDF.xMin]
plt.hist(x_1_Rand, bins=20, rwidth=0.8)
plt.yscale("linear")
plt.show()
x_2_Rand = stats.loguniform.rvs(a=x_lower_limit + loguni_param, b=x_upper_limit + loguni_param, size=x_total) - loguni_param
x_2_Rand = x_2_Rand[x_2_Rand >= PDF.xMin]
eta_Rand=np.array([])
while eta_Rand.size < eta_total:
    eta_Rand  = np.append(eta_Rand, -abs(stats.norm.rvs(scale=1.2, size=eta_total-eta_Rand.size)) + eta_limit)
    eta_Rand = eta_Rand[eta_Rand >= 0]

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
    x_1_constant = x_1_Rand[i]
    i += 1
i=0
while x_2_constant < 0.05 or x_2_constant > 0.15:
    x_2_constant = x_2_Rand[i]
    i += 1
i=0

while eta_constant <0.5 or eta_constant > 2.0:
    eta_constant = eta_Rand[i]
    i += 1

print(x_1_constant)
print(x_2_constant)
print(eta_constant)
print(np.min(x_1_Rand))
print(PDF.xMin)

for x_1 in x_1_Rand:
    if x_1 < PDF.xMin:
        x_1 = PDF.xMin
    for x_2 in x_2_Rand:
        if x_2 < PDF.xMin:
            x_2 = PDF.xMin
        for eta in eta_Rand:
            diff_WQ = 0
            # Viererimpuls initialisieren
            Q2 = 2 * x_1 * x_2 * (E ** 2)
            # Listen der Eingangswerte füllen:
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            eta_list.append(eta)
            # WQ berechnen, Summe über Quarks:
            """
            for q in quarks["quark"]:
                diff_WQ += (((quarks["charge"][q - 1]) ** 4)/(192 * np.pi * x_1 * x_2 * E ** 2)) * \
                           ((np.abs(PDF.xfxQ2(q, x_1, Q2) * PDF.xfxQ2(-q, x_2, Q2)) + np.abs(PDF.xfxQ2(-q, x_1, Q2) * PDF.xfxQ2(q, x_2, Q2)))/(x_1 * x_2)) * \
                           (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)
            # diff WQ in Liste einfügen:
            diff_WQ_list.append(diff_WQ)
            """
            for q in quarks["quark"]:
                diff_WQ += (((quarks["charge"][q - 1]) ** 4)/(192 * np.pi * x_1 * x_2 * E ** 2)) * \
                           ((np.abs(PDF.xfxQ2(q, x_1, Q2) * PDF.xfxQ2(-q, x_2, Q2)) + np.abs(PDF.xfxQ2(-q, x_1, Q2) * PDF.xfxQ2(q, x_2, Q2)))/(x_1 * x_2)) * \
                           (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)

            diff_WQ_list.append(diff_WQ)

            # Listet mit konstantem x anlegen
            if x_1 == x_1_constant and x_2 == x_2_constant:
                eta_list_x_constant.append(eta)
                diff_WQ_list_x_constant.append(diff_WQ)
                x_1_list_x_constant.append(x_1)
                x_2_list_x_constant.append(x_2)

            # Listen mit konstantem eta, konstantem x_2 anlegen
            if eta == eta_constant and x_2 == x_2_constant:
                print("x_1", x_1, "diff_WQ", diff_WQ)
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

            diff_WQ = 0
            step += 1
            #if step % 50000 == 0:
                #print("x_1:", x_1, "x_2:", x_2, "eta:", eta, "diff_WQ:", diff_WQ)


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

print(hadronic_diff_WQ_data)
print(hadronic_diff_WQ_data_x_constant)
print(hadronic_diff_WQ_data_eta_x_2_constant)
print(hadronic_diff_WQ_data_eta_x_1_constant)







