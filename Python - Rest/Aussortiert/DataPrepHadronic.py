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
x_total = int(25) #Anzahl an x Werten
eta_total = int(200) # Anzahl an eta Werten
x_lower_limit = 0.05
x_upper_limit = 0.075
eta_limit = 3

name = "some_hadronic_data"
#Test
test = []
for x_ in np.linspace(0,np.log10(100),50):
    x = 10**(x_)
    test.append(x)
print(test)
print(np.logspace(0,2,50))
exit()
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

#Step-sizes herausfinden, um anständig die anderen Listen zu generieren
(_, x_step) = np.linspace(start=x_lower_limit, stop=x_upper_limit, num=x_total, retstep=True)
(_, eta_step) = np.linspace(start=-eta_limit, stop=eta_limit, num=eta_total, retstep=True)
#Feste werte setzen
x_constant = x_lower_limit + 20 * x_step
eta_constant = (-eta_limit) + 150 * eta_step

print(x_constant)
print(eta_constant)

#Listen mit Funktionswerten anlegen
#Ausgangswerte
diff_WQ_list = []
diff_WQ_list_eta_constant=[]
diff_WQ_list_x_constant = []
#Eingangswerte
x_1_list = []
x_2_list = []
eta_list = []
eta_list_x_constant = []
x_list_eta_constant = []
x_1_list_x_constant = []
x_2_list_x_constant = []
eta_list_eta_constant = []
x_2_list_eta_constant = []

#diff WQ berechnen und Listen füllen
step = 0
for x_1 in np.linspace(start=x_lower_limit, stop=x_upper_limit, num=x_total):
    #Polstelle behandeln
    if x_1 < PDF.xMin:
        x_1 = PDF.xMin
    for x_2 in np.linspace(start=x_lower_limit, stop=x_upper_limit, num=x_total):
        #Polstelle behandeln
        if x_2 < PDF.xMin:
            x_2 = PDF.xMin
        for eta in np.linspace(start=-eta_limit, stop=eta_limit, num=eta_total):
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
                x_1_list_x_constant.append(x_1)
                x_2_list_x_constant.append(x_2)

            #Listen mit konstantem eta anlegen
            if eta == eta_constant and x_2 == x_constant:
                x_list_eta_constant.append(x_1)
                diff_WQ_list_eta_constant.append(diff_WQ)
                eta_list_eta_constant.append(eta)
                x_2_list_eta_constant.append(x_2)

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
    {   "x_1": x_1_list_x_constant,
        "x_2": x_2_list_x_constant,
        "eta": eta_list_x_constant,
        "WQ": diff_WQ_list_x_constant
    }
)

hadronic_diff_WQ_data_eta_constant = pd.DataFrame(
    {
        "x_1": x_list_eta_constant,
        "x_2": x_2_list_eta_constant,
        "eta": eta_list_eta_constant,
        "WQ": diff_WQ_list_eta_constant
    }
)

x_constant_name = name + "__x_constant__" + str(x_constant)
eta_constant_name = name + "__eta_constant__" + str(eta_constant)

hadronic_diff_WQ_data.to_csv("HadronicData/" + name, index=False)
hadronic_diff_WQ_data_x_constant.to_csv("HadronicData/" + x_constant_name, index=False)
hadronic_diff_WQ_data_eta_constant.to_csv("HadronicData/" + eta_constant_name, index=False)

print(hadronic_diff_WQ_data)
print(hadronic_diff_WQ_data_x_constant)
print(hadronic_diff_WQ_data_eta_constant)






