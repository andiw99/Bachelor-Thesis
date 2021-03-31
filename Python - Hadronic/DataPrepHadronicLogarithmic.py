import lhapdf as pdf
import pandas as pd
import numpy as np
import os

#quarks = {"quark": [1, 2, 3, 4],
#          "charge": [-1/3,2/3,-1/3,2/3]}
#PID = {1: "d", 2: "u", 3: "s", 4: "c"}
quarks = {"quark": [1, 2],
          "charge": [-1/3,2/3]}
PID = {1: "d", 2: "u", 3: "s", 4: "c"}

#PDF initialisieren
PDF = pdf.mkPDF("NNPDF30_nnlo_as_0118", 0)

#for q in quarks["quark"]:
    #print("Quark ", PID[q], "hat Ladung ", quarks["charge"][q-1])

#Variablen
e = 1.602e-19
E = 10 #Strahlenergie in GeV, im Vornherein festgelegt?
x_total = int(25) #Anzahl an x Werten
eta_total = int(1500) # Anzahl an eta Werten
x_lower_limit = 0
x_upper_limit = 1
eta_limit = 3

set_name = "log_neg_3D/"
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/" + set_name

name = "logarithmic_hadronic_data_no_negative"
#Test
print("Wir berechnen ", x_total**2 * eta_total, "Werte")

#Step-sizes herausfinden, um anständig die anderen Listen zu generieren
(_, x_step) = np.linspace(start=np.log10(x_lower_limit+1), stop=np.log10(x_upper_limit+1), num=x_total, retstep=True)
(_, eta_step) = np.linspace(start=-eta_limit, stop=eta_limit, num=eta_total, retstep=True)
log_x_step = 10 ** x_step
#Feste werte setzen
x_constant = x_lower_limit + 20 * log_x_step
eta_constant = 150 * eta_step
x_constant = 0.10956947206784506
print(x_constant)
print(eta_constant)

#Listen mit Funktionswerten anlegen
#Ausgangswerte
diff_WQ_list = []
diff_WQ_list_eta_constant=[]
diff_WQ_list_x_constant = []
diff_WQ_list_3D = []
#Eingangswerte
x_1_list = []
x_2_list = []
eta_list = []
eta_list_x_constant = []
eta_list_3D = []
x_list_eta_constant = []
x_list_x_constant = []
x_list_3D = []
eta_list_eta_constant = []
x_2_list_eta_constant = []
step = 0

#diff WQ berechnen und Listen füllen
for x_1_raw in np.linspace(start=np.log10(x_lower_limit+1), stop=np.log10(x_upper_limit+1), num=x_total, endpoint=False):
    x_1 = 10 ** (x_1_raw) - 1
    #Polstelle behandeln
    if x_1 < PDF.xMin:
        x_1 = PDF.xMin
    for x_2_raw in np.linspace(start=np.log10(x_lower_limit+1), stop=np.log10(x_upper_limit+1), num=x_total, endpoint=False):
        x_2 = 10 ** (x_2_raw) - 1
        #Polstelle behandeln
        if x_2 < PDF.xMin:
            x_2 = PDF.xMin
        for eta in np.linspace(start=0, stop=eta_limit, num=eta_total):
            # diff WQ nullen, damit die nächste Summe wieder in sinnvolles Ergebnis liefert:
            diff_WQ = 0
            #Viererimpuls initialisieren
            Q2 = 2* x_1 * x_2 * (E**2)
            #Listen der Eingangswerte füllen:
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            eta_list.append(eta)
            #WQ berechnen, Summe über Quarks:
            for q in quarks["quark"]:
                diff_WQ +=  (((quarks["charge"][q-1])**4)/(192* np.pi * x_1 * x_2 * E**2)) * \
                            ((PDF.xfxQ2(q, x_1, Q2) * PDF.xfxQ2(-q, x_2, Q2) + PDF.xfxQ2(-q, x_1, Q2) * PDF.xfxQ2(q, x_2, Q2)) / (x_1 * x_2)) * \
                            (1 + (np.tanh(eta + 1/2 * np.log(x_2/x_1)))**2)
            #diff WQ in Liste einfügen:
            diff_WQ_list.append(diff_WQ)

            #Listet mit konstantem x anlegen
            if x_1 == x_constant and x_2 == x_constant:
                eta_list_x_constant.append(eta)
                diff_WQ_list_x_constant.append(diff_WQ)
                x_list_x_constant.append(x_1)


            #Listen mit konstantem eta, konstantem x_2 anlegen
            if eta == eta_constant and x_2 == x_constant:
                x_list_eta_constant.append(x_1)
                diff_WQ_list_eta_constant.append(diff_WQ)
                eta_list_eta_constant.append(eta)

            #Liste für den 3D-Plot erstellen
            if x_2 == x_constant:
                x_list_3D.append(x_1)
                eta_list_3D.append(eta)
                diff_WQ_list_3D.append(diff_WQ)

            step += 1
            if step % 50000 == 0:
                print("x_1:", x_1 , "x_2:", x_2, "eta:", eta, "diff_WQ:", diff_WQ)


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
        "x_1": x_list_x_constant,
        "x_2": x_list_x_constant,
        "eta": eta_list_x_constant,
        "WQ": diff_WQ_list_x_constant
    }
)


hadronic_diff_WQ_data_eta_x_2_constant = pd.DataFrame(
    {
        "x_1": x_list_eta_constant,
        "x_2": x_list_x_constant,
        "eta": eta_list_eta_constant,
        "WQ": diff_WQ_list_eta_constant
    }
)

hadronic_diff_WQ_data_eta_x_1_constant = pd.DataFrame(
    {
        "x_1": x_list_x_constant,
        "x_2": x_list_eta_constant,
        "eta": eta_list_eta_constant,
        "WQ": diff_WQ_list_eta_constant
    }
)

hadronic_diff_WQ_data_x_2_constant = pd.DataFrame(
    {
        "x_1": x_list_3D,
        "x_2": x_list_x_constant,
        "eta": eta_list_3D,
        "WQ": diff_WQ_list_3D
    }
)

x_constant_name = name + "__x_constant__" + str("{:.2f}".format(x_constant))
eta_x_2_constant_name = name + "__eta_x_2_constant__" + str("{:.2f}".format(eta_constant))
eta_x_1_constant_name = name + "__eta_x_1_constant__" + str("{:.2f}".format(eta_constant))
x_2_constant_name = name + "__x_2_constant__3D"

#ggf. Verzeichnis erstellen
if not os.path.exists(path=path):
    os.makedirs(path)

hadronic_diff_WQ_data.to_csv(path + name, index=False)
hadronic_diff_WQ_data_x_constant.to_csv(path + x_constant_name, index=False)
hadronic_diff_WQ_data_eta_x_2_constant.to_csv(path + eta_x_2_constant_name, index=False)
hadronic_diff_WQ_data_eta_x_1_constant.to_csv(path + eta_x_1_constant_name, index=False)
hadronic_diff_WQ_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)

print(hadronic_diff_WQ_data)
print(hadronic_diff_WQ_data_x_constant)
print(hadronic_diff_WQ_data_eta_x_2_constant)
print(hadronic_diff_WQ_data_eta_x_1_constant)






