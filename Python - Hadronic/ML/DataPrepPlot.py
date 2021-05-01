import random
import lhapdf as pdf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import ml


def main():
    quarks = {"quark": [1, 2, 3, 4],
              "charge": [-1/3,2/3, -1/3, 2/3]}

    #PDF initialisieren
    PDF = pdf.mkPDF("CT14nnlo", 0)
    #PDF = pdf.mkPDF("MMHT2014nnlo68cl", 0)

    #Variablen
    E = 6500 #Strahlenergie in GeV, im Vornherein festgelegt?
    x_total = int(5000) #Anzahl an x Werten
    eta_total = x_total # Anzahl an eta Werten
    x_lower_limit = 0
    x_upper_limit = 1
    eta_limit = 3
    loguni_param=0.05
    stddev = 4
    xMin = PDF.xMin
    random_seed = 30
    np.random.seed(random_seed)

    set_name = "PlottingDataHighX/"
    path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Transfer/Data/" + set_name

    #Test
    total_data = x_total
    print("Wir berechnen ", total_data, "Werte")

    x_1 = np.array([])
    x_2 = np.array([])
    while x_1.size < x_total:
        x_1 = np.concatenate((x_1, (
                    stats.loguniform.rvs(a=loguni_param, b=1 + loguni_param,
                                         size=x_total - x_1.size) - loguni_param) * (
                                          x_upper_limit - x_lower_limit) + x_lower_limit))
        x_1 = x_1[x_1 >= xMin]

    while x_2.size < x_total:
        x_2 = np.concatenate((x_2, (
                    stats.loguniform.rvs(a=loguni_param, b=1 + loguni_param,
                                         size=x_total - x_2.size) - loguni_param) * (
                                          x_upper_limit - x_lower_limit) + x_lower_limit))
        x_2 = x_2[x_2 >= xMin]
    plt.hist(x_1, bins=20, rwidth=0.8)
    plt.yscale("linear")
    plt.show()
    eta_upper=np.array([])
    eta_lower = np.array([])
    eta = np.array([])
    while eta.size < eta_total:
        eta_upper  = np.append(eta_upper, -abs(stats.norm.rvs(scale=stddev, size=int((eta_total/2))-eta_upper.size)) + eta_limit)
        eta_upper = eta_upper[eta_upper >= 0]
        eta_lower = np.append(eta_lower, abs(stats.norm.rvs(scale=stddev , size = int((eta_total/2))- eta_lower.size)) - eta_limit)
        eta_lower = eta_lower[eta_lower <= 0]
        eta = np.append(eta_upper, eta_lower)
    plt.hist(eta, bins=20, rwidth=0.8)
    plt.show()

    print(x_1.min())
    print(eta.size)

    #Feste werte setzen
    i=0
    x_1_constant=0
    x_2_constant=0
    eta_constant = 0
    while x_1_constant < 0.4 or x_1_constant > 0.46:
        if i >= x_total:
            x_1_constant = x_1[i-1]
            break
        x_1_constant = x_1[i]
        i += 1
    i=0
    while x_2_constant < 0.4 or x_2_constant > 0.46:
        if i >= x_total:
            x_2_constant = x_2[i-1]
            break
        x_2_constant = x_2[i]
        i += 1
    i=0

    while eta_constant < 0.5 or eta_constant > 1.0:
        if i >= eta_total:
            eta_constant = eta[i]
            break
        eta_constant = eta[i]
        i += 1

    #arrays mit konstanten werten anlegen
    x_1_constant_list = np.zeros(shape=x_total)
    x_1_constant_list += x_1_constant
    x_2_constant_list = np.zeros(shape=x_total)
    x_2_constant_list += x_2_constant
    eta_constant_list = np.zeros(shape=x_total)
    eta_constant_list += eta_constant

    print(x_1_constant)
    print(x_2_constant)
    print(eta_constant)
    print(np.min(x_1))
    print(xMin)

    diff_WQ_eta_x_1_constant = ml.calc_diff_WQ(E=E, PDF=PDF, quarks=quarks, x_1 = x_1_constant_list, eta=eta_constant_list, x_2=x_2)
    diff_WQ_eta_x_2_constant = ml.calc_diff_WQ(E=E, PDF=PDF, quarks=quarks, x_1 = x_1, eta=eta_constant_list, x_2=x_2_constant_list)
    diff_WQ_x_constant = ml.calc_diff_WQ(E=E, PDF=PDF, quarks=quarks, x_1 = x_1_constant_list, eta=eta, x_2=x_2_constant_list)
    diff_WQ_eta_constant = ml.calc_diff_WQ(E=E, PDF=PDF, quarks=quarks, x_1 = x_1, eta=eta_constant_list, x_2=x_2_constant_list)
    diff_WQ_x_2_constant = ml.calc_diff_WQ(E=E, PDF=PDF, quarks=quarks, x_1 = x_1, eta=eta, x_2=x_2_constant_list)

    hadronic_diff_WQ_data_x_constant = pd.DataFrame(
        {
            "x_1": x_1_constant_list,
            "x_2": x_2_constant_list,
            "eta": eta,
            "WQ": diff_WQ_x_constant
        }
    )


    hadronic_diff_WQ_data_eta_x_2_constant = pd.DataFrame(
        {
            "x_1": x_1,
            "x_2": x_2_constant_list,
            "eta": eta_constant_list,
            "WQ": diff_WQ_eta_x_2_constant
        }
    )

    hadronic_diff_WQ_data_eta_x_1_constant = pd.DataFrame(
        {
            "x_1": x_1_constant_list,
            "x_2": x_2,
            "eta": eta_constant_list,
            "WQ": diff_WQ_eta_x_1_constant
        }
    )

    hadronic_diff_WQ_data_x_2_constant = pd.DataFrame(
        {
            "x_1": x_1,
            "x_2": x_2_constant_list,
            "eta": eta,
            "WQ": diff_WQ_x_2_constant
        }
    )

    hadronic_diff_WQ_eta_constant = pd.DataFrame(
        {
            "x_1": x_1,
            "x_2": x_2,
            "eta": eta_constant_list,
            "WQ": diff_WQ_eta_constant
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
            "random_seed": random_seed,

        },
        index=[0]
    )

    x_constant_name =  "x_constant"
    eta_x_2_constant_name =  "eta_x_2_constant"
    eta_x_1_constant_name =  "eta_x_1_constant"
    x_2_constant_name = "x_2_constant__3D"
    eta_constant_name = "eta_constant__3D"

    #ggf. Verzeichnis erstellen
    if not os.path.exists(path=path):
        os.makedirs(path)

    hadronic_diff_WQ_data_x_constant.to_csv(path + x_constant_name, index=False)
    hadronic_diff_WQ_data_eta_x_2_constant.to_csv(path + eta_x_2_constant_name, index=False)
    hadronic_diff_WQ_data_eta_x_1_constant.to_csv(path + eta_x_1_constant_name, index=False)
    hadronic_diff_WQ_data_x_2_constant.to_csv(path + x_2_constant_name, index=False)
    hadronic_diff_WQ_eta_constant.to_csv(path + eta_constant_name, index=False)
    config.to_csv(path + "config", index=False)

    print(hadronic_diff_WQ_data_x_constant)
    print(hadronic_diff_WQ_data_eta_x_2_constant)
    print(hadronic_diff_WQ_data_eta_x_1_constant)
    print(hadronic_diff_WQ_eta_constant)


if __name__ == "__main__":
    main()
