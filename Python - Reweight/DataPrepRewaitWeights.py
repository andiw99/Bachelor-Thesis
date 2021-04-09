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
#quarks = {"quark": [1, 2],
 #         "charge": [-1/3,2/3]}

PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF_CT14 = pdf.mkPDF("CT14nnlo", 0)
PDF_MMHT = pdf.mkPDF("MMHT2014nnlo68cl", 0)

#Variablen
e = 1.602e-19
E = 6500 #Strahlenergie in GeV, im Vornherein festgelegt?
x_total = int(300) #Anzahl an x Werten
x_lower_limit = 0.0
x_upper_limit = 0.65
#loguni_param=0.01

set_name = "Uniform+Strange+middlex/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/" + set_name

#Test
print("Wir berechnen ", x_total**2, "Werte")



#Listen mit Funktionswerten anlegen
#Ausgangswerte
reweight_list = []
reweight_list_x_2_constant=[]
reweight_list_x_1_constant = []
reweight_list_3D = []
#Eingangswerte
x_1_list = []
x_2_list = []
x_1_list_x_1_constant = []
x_2_list_x_1_constant = []
x_1_list_x_2_constant = []
x_2_list_x_2_constant = []
x_1_list_x_constant = []
x_2_list_x_constant = []
x_1_list_3D = []
x_2_list_3D = []

step = 0

#diff WQ berechnen und Listen füllen

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


print(x_1_constant)
print(x_2_constant)
print(np.min(x_1_Rand))
xMin = PDF_CT14.xMin

for x_1 in x_1_Rand:
    if x_1 < xMin:
        x_1 = xMin
    for x_2 in x_2_Rand:
        if x_2 < xMin:
            x_2 = xMin
        sum1_frac = 0
        sum2_frac = 0
        sum1 = 0
        sum2 = 0
        # Viererimpuls initialisieren
        Q2 = 2 * x_1 * x_2 * (E ** 2)

        # Listen der Eingangswerte füllen:
        x_1_list.append(x_1)
        x_2_list.append(x_2)
        # reweight berechnen, Summe über Quarks:
        for q in quarks["quark"]:
            sum1_frac = ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, PDF_CT14.xfxQ2(q, x_1, Q2) * PDF_CT14.xfxQ2(-q, x_2, Q2)) + \
                    np.maximum(0, PDF_CT14.xfxQ2(-q, x_1, Q2) * PDF_CT14.xfxQ2(q, x_2, Q2))) / (x_1 * x_2))
            sum2_frac = ((quarks["charge"][q-1])**4) *  \
                    ((np.maximum(0, PDF_MMHT.xfxQ2(q, x_1, Q2) * PDF_MMHT.xfxQ2(-q, x_2, Q2)) + \
                    np.maximum(0, PDF_MMHT.xfxQ2(-q, x_1, Q2) * PDF_MMHT.xfxQ2(q, x_2, Q2))) / (x_1 * x_2))
            sum1 += sum1_frac
            sum2 += sum2_frac

        # diff reweight in Liste einfügen:
        reweight = sum1/sum2
        reweight_list.append(reweight)

        # Listen mit konstantem x_2 anlegen
        if x_2 == x_2_constant:
            x_1_list_x_2_constant.append(x_1)
            reweight_list_x_2_constant.append(reweight)
            x_2_list_x_2_constant.append(x_2)

        #Listen mit konstantem x_1 anlegen
        if x_1 == x_1_constant:
            x_1_list_x_1_constant.append(x_1)
            x_2_list_x_1_constant.append(x_2)
            reweight_list_x_1_constant.append(reweight)

        reweight = 0
        step += 1
        if step % 50000 == 0:
            print("x_1:", x_1, "x_2:", x_2, "reweight:", reweight)


hadronic_reweight_data = pd.DataFrame(
    {
        "x_1": x_1_list,
        "x_2": x_2_list,
        "reweight": reweight_list
    }
)


hadronic_reweight_data_x_2_constant = pd.DataFrame(
    {
        "x_1": x_1_list_x_2_constant,
        "x_2": x_2_list_x_2_constant,
        "reweight": reweight_list_x_2_constant
    }
)

hadronic_reweight_data_x_1_constant = pd.DataFrame(
    {
        "x_1": x_1_list_x_1_constant,
        "x_2": x_2_list_x_1_constant,
        "reweight": reweight_list_x_1_constant
    }
)


x_constant_name =  "x_constant"
x_2_constant_name =  "x_2_constant"
x_1_constant_name =  "x_1_constant"

#ggf. Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

hadronic_reweight_data.to_csv(path + "all", index=False)
hadronic_reweight_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)
hadronic_reweight_data_x_1_constant.to_csv(path + x_1_constant_name, index=False)
hadronic_reweight_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)

print(hadronic_reweight_data)
print(hadronic_reweight_data_x_2_constant)
print(hadronic_reweight_data_x_1_constant)







