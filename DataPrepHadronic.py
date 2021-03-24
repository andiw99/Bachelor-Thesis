import lhapdf as pdf
import pandas as pd
import numpy as np

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
x_total = int(100) #Anzahl an x Werten
eta_total = int(400) # Anzahl an eta Werten

#Test
print("Wir berechnen ", x_total**2 * eta_total, "Werte")
"""
x_1_test = 0.99
x_2_test = 0.99
diff_WQ_test = 0
Q2_test =2 * x_1_test * x_2_test *(E**2)
for eta_test_raw in range(-30, 30):
    eta_test = eta_test_raw/10
    for q in quarks["quark"]:
        diff_WQ_test += (((quarks["charge"][q - 1]) ** 4) / (192 * np.pi * x_1_test * x_2_test * E ** 2)) * \
                        ((PDF.xfxQ2(q, x_1_test, Q2_test) * PDF.xfxQ2(-q, x_2_test, Q2_test) + PDF.xfxQ2(-q, x_1_test, Q2_test) * PDF.xfxQ2(q, x_2_test, Q2_test)) / (x_1_test * x_2_test)) * \
                        (1 + (np.tanh(eta_test * 1 / 2 * np.log(x_2_test / x_1_test))) ** 2)
        print("Quark:", PID[q])
        print("1: ", PDF.xfxQ2(q, x_1_test, Q2_test) * PDF.xfxQ2(-q, x_2_test, Q2_test))
        print("2: ", PDF.xfxQ2(-q, x_1_test, Q2_test) * PDF.xfxQ2(q, x_2_test, Q2_test))
        print("tanh:", (1 + (np.tanh(eta_test + 1 / 2 * np.log(x_2_test / x_1_test))) ** 2))
        print("eta_test:", eta_test, "diff WQ test:", diff_WQ_test)
"""

#Listen mit Funktionswerten anlegen
#Ausgangswerte
diff_WQ_list = []
#Eingangswerte
x_1_list = []
x_2_list = []
eta_list = []


#diff WQ berechnen und Listen füllen
step = 0
for x_1_raw in range(x_total):
    x_1 = x_1_raw/x_total
    #Polstelle behandeln
    if x_1 < PDF.xMin:
        x_1 = PDF.xMin
    for x_2_raw in range(x_total):
        x_2 = x_2_raw/x_total
        #Polstelle behandeln
        if x_2 < PDF.xMin:
            x_2 = PDF.xMin
        for eta_raw in range(-int(1/2 * eta_total), int(1/2 * eta_total)):
            # diff WQ nullen, damit die nächste Summe wieder in sinnvolles Ergebnis liefert:
            diff_WQ = 0
            eta = 3*eta_raw/(1/2 * eta_total)
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

hadronic_diff_WQ_data.to_csv("hadronic_WQ_data", index=False)
print(hadronic_diff_WQ_data)






