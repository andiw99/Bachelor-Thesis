import random
import lhapdf as pdf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import time

import MC


def main():
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
    E = 6500 #Strahlenergie in GeV, im Vornherein festgelegt?
    x_total = int(10000000) #Anzahl an x Werten
    eta_total = int(10000000) # Anzahl an eta Werten
    x_lower_limit = 0
    x_upper_limit = 1
    eta_limit = 2.37
    loguni_param=0.001 #alt 0.01
    stddev = 2
    xMin = 1e-6
    eta_constant = False
    x_Grid = False
    cuts = True
    lfs = False
    eta_gauss = True
    num_eta_values = 25
    nr_sets = 20

    set_name = "MC10M/"
    root_name ="/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit"
    location = None #input("Welcher Rechner?")
    if location == "Taurus" or location == "taurus":
        root_name = "/home/s1388135/Bachelor-Thesis"
    if lfs:
        root_name = "/media/andiw/90D8E3C1D8E3A3A6/Users/andiw/Studium/Semester 6/Bachelor-Arbeit/LFS"
    path = root_name + "/Files/Hadronic/Data/" + set_name

    for i in range(nr_sets):
        #Werte erzeugen
        x_2 = np.array([])
        x_1 = np.array([])
        if x_Grid:
            x = np.linspace(start=x_lower_limit, stop=x_upper_limit, num=x_total)
            for x_1_value in x:
                x_2_frac = np.zeros(shape=(int(len(x)), ), dtype="float32")
                x_2_frac += x_1_value
                x_2 = np.concatenate((x_2, x_2_frac))
                x_1 = np.concatenate((x_1, x))
        else:
            while x_1.size < x_total:
                x_1 = np.concatenate((x_1, (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=int(np.minimum(20e+6, x_total - x_1.size))) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit), dtype="float32")
                x_1 = x_1[x_1 >= xMin]

            while x_2.size < x_total:
                x_2 = np.concatenate((x_2, (stats.loguniform.rvs(a=loguni_param, b=1+loguni_param, size=int(np.minimum(20e+6, x_total-x_2.size))) - loguni_param) * (x_upper_limit - x_lower_limit) + x_lower_limit), dtype="float32")
                x_2 = x_2[x_2 >= xMin]

        plt.hist(x_1, bins=30, rwidth=0.8)
        plt.yscale("linear")
        plt.show()

        eta = np.array([], dtype="float32")
        if eta_constant:
            x_1_Rand = x_1
            x_2_Rand = x_2
            print(x_2_Rand)
            print(x_1_Rand)
            print(x_2_Rand.size)
            print(x_1_Rand.size)

            #TODO eventuell eher mit assign machen?
            x_1 = np.array([], dtype="float32")
            x_2 = np.array([], dtype="float32")
            eta_values = np.linspace(-eta_limit, eta_limit, num=num_eta_values)
            eta_values = MC.crack_cut(eta_values)
            for eta_value in eta_values:
                eta_frac = np.zeros(shape=(int(len(x_1_Rand)), ), dtype="float32")
                eta_frac += eta_value
                eta = np.concatenate((eta, eta_frac))
                x_1 = np.concatenate((x_1, x_1_Rand))
                x_2 = np.concatenate((x_2, x_2_Rand))

        else:
            if eta_gauss:
                while eta.size < eta_total:
                    eta = np.concatenate((eta, stats.norm.rvs(scale=stddev, size=int(np.minimum(20e+6, int(eta_total - eta.size))))), dtype="float32")
                    eta = eta[np.abs(eta) < eta_limit]
            eta_upper=np.array([])
            eta_lower = np.array([])
            while eta.size < eta_total:
                eta_upper  = np.append(eta_upper, -abs(stats.norm.rvs(scale=stddev, size=int((eta_total/2))-eta_upper.size)) + eta_limit)
                eta_upper = eta_upper[eta_upper >= 0]
                eta_lower = np.append(eta_lower, abs(stats.norm.rvs(scale=stddev , size = int((eta_total/2))- eta_lower.size)) - eta_limit)
                eta_lower = eta_lower[eta_lower <= 0]
                eta = np.append(eta_upper, eta_lower)


        print("die daten wurden generiert")

        features = np.array([x_1, x_2, eta])
        features = np.transpose(features)
        print("die daten wurden in features umgewandelt")
        if cuts:
            features = MC.pt_cut(features)
            features, eta_cut = MC.eta_cut(features=features, return_cut=True)
        print("die daten wurden gecuttet")

        imposter_numbers=0
        imposter_check=False
        if imposter_check:
            for feature in features:
                if np.abs(feature[2]) > 2.37 or np.abs((feature[2] + 1/2 * np.log((feature[1]**2)/(feature[0]**2)))) > 2.37:
                    imposter_numbers +=1
                    print("imposter gefunden, betrag von eta zu groß")
                elif 1.37 < np.abs(feature[2]) < 1.52 or 1.37 < np.abs(MC.calc_other_eta(feature[2], x_1=feature[0], x_2=feature[1])) < 1.52:
                    print("imposter gefunden, Event im Crack")
                if MC.calc_pt(feature[2], feature[0], feature[1], E=6500) < 40:
                    print("imposter gefunden, pt zu klein")

            print("insgesamt", imposter_numbers, "imposter")

        plt.hist(features[:,0], bins=40, rwidth=0.9)
        plt.show()
        plt.hist(features[:,1], bins=40, rwidth=0.9)
        plt.show()
        plt.hist(features[:,2], bins=50, rwidth=0.9)
        plt.show()

        time_pre_calc = time.time()
        diff_WQ = ml.calc_diff_WQ(PDF=PDF, quarks=quarks, x_1=features[:,0], x_2=features[:,1], eta=features[:,2], E=E)
        comp_time_per_million = (time.time()-time_pre_calc)/len(diff_WQ) * 1e+6


        #berechnen, wie viele punkte akzeptiert und wie viele gecuttet wurden:
        acceptance_ratio = diff_WQ.size/x_total

        hadronic_diff_WQ_data = pd.DataFrame(
            {
                "x_1": features[:,0],
                "x_2": features[:,1],
                "eta": features[:,2],
                "WQ": diff_WQ
            }
        )


        config = pd.DataFrame(
            {
                "x_total": x_total,
                "total_data": diff_WQ.size,
                "x_lower_limit": x_lower_limit,
                "x_upper_limit": x_upper_limit,
                "eta_limit": eta_limit,
                "loguni_param": loguni_param,
                "stddev": stddev,
                "cuts": str(cuts),
                "eta constant": eta_constant,
                "x_Grid": x_Grid,
                "acceptance_ratio": "{:.5f}".format(acceptance_ratio),
                "eta_gauss": eta_gauss
            },
            index=[0]
        )


        #ggf. Verzeichnis erstellen
        if not os.path.exists(path=path + str(i) + "/"):
            os.makedirs(path+ str(i) + "/")

        hadronic_diff_WQ_data.to_csv(path + str(i) + "/all", index=False)

        config.to_csv(path + str(i) + "/config", index=False)

        print(hadronic_diff_WQ_data)


if __name__ == "__main__":
    main()
