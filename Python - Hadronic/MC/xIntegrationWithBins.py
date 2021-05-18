import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import integrate
import ml
import MC
import os


def main():
    #Random Samples einlesen, und zwar alle und dann in bins einteilen:
    # eventuell mehrere datensets
    dataset_paths = list()
    directory = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC10M/"
    datasets = os.listdir(directory)
    for dataset in datasets:
        if dataset != "conifg":
            dataset_paths.append(directory + dataset + "/")
    print(dataset_paths)
    """
    dataset_paths.append("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC30M_newgauss/")
    dataset_paths.append(
       "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC50M_II/")
    dataset_paths.append(
       "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC50M_III/")
    dataset_paths.append(
        "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC50M_IV/")
    dataset_paths.append(
       "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Data/MC50M_V/")
    """
    label_name = "WQ"
    max_iterations = 100
    with_ml = True
    # Config der RandomSample generierung einlesen
    config = pd.read_csv(dataset_paths[0] + "config")
    # model einlesen
    model_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/Models/best_model"
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/"
    (model, transformer) = ml.import_model_transformer(model_path=model_path)

    # Variablen initialisieren (der Verteilungen):
    variables = dict()
    for key in config:
        variables[key] = float(config[key][0])
    nr_bins = int(variables["total_data"] / 75000)
    eta_interval = np.linspace(start=-variables["eta_limit"],
                               stop=variables["eta_limit"], num=(nr_bins + 1) + (nr_bins) % 2)  #dafür sorgen, dass eta interval ungearde
                                                                                                    #viele

    # Crack values hereinbringen und werte im crack entfernen, array wieder ordnen
    eta_interval = np.concatenate(
        (eta_interval, np.array([-1.52, -1.37, 1.37, 1.52])))
    eta_interval = eta_interval[
        (np.abs(eta_interval) >= 1.52) | (np.abs(eta_interval) <= 1.37)]
    eta_interval = np.sort(eta_interval)
    nr_bins = len(eta_interval) - 1
    print("nr_bins", nr_bins)
    print("eta_interval", eta_interval)
    analytic_integrals = list()
    analytic_stddevs = list()
    ml_integrals = list()
    ml_stddevs = list()
    #Für jede Datenmenge MC-Integration machen
    for i,path in enumerate(dataset_paths):
        if i >= max_iterations:
            break
        features, labels = MC.data_handling(data_path=path + "all", label_name=label_name, return_pd=False)

        config = pd.read_csv(path + "config")

        #Variablen initialisieren (der Verteilungen):
        variables = dict()
        for key in config:
            variables[key] = float(config[key][0])

        #verschiedene eta Werte isolieren
        features_eta_constant = dict()
        labels_eta_constant = dict()
        predictions = dict()
        # Aus den eta-Werten Bins machen
        # bins festlegen, pro bin ca. 50000 Punkte

        for i in range(nr_bins):
            # Die features in die bins aufteilen
            features_eta_constant["{:.2f}".format(eta_interval[i])] = features[(features[:,2] < eta_interval[i+1]) & (features[:,2] > eta_interval[i])]
            labels_eta_constant["{:.2f}".format(eta_interval[i])] = labels[(features[:,2] < eta_interval[i+1]) & (features[:,2] > eta_interval[i])]
        print("Die Daten wurden in Bins eingeteilt, features, labels werden freigegeben")
        del features
        del labels
        if with_ml:
            for i in range(nr_bins):
                try:
                    predictions["{:.2f}".format(eta_interval[i])] = transformer.retransform(model.predict(features_eta_constant["{:.2f}".format(eta_interval[i])]))
                except ValueError:
                    predictions["{:.2f}".format(eta_interval[i])] = np.array([[0]])

        # Wahrscheinlichkeitsverteilungen initialisieren
        scaling_loguni = 1/(variables["x_upper_limit"]-variables["x_lower_limit"])
        print("scaling_loguni:", scaling_loguni)
        loguni = MC.class_loguni(loguni_param=variables["loguni_param"], x_lower_limit=variables["x_lower_limit"], x_upper_limit=variables["x_upper_limit"],scaling=scaling_loguni)

        # Benötige ich die Gauß verteilung?
        er_fc = MC.erf(mu=0, sigma=variables["stddev"])
        scaling_gauss = 1 / (
                    er_fc(variables["eta_limit"]) - er_fc(-variables["eta_limit"]))
        gauss = MC.gaussian(mu=0, sigma=variables["stddev"], scaling=scaling_gauss)
        er_fc = MC.erf(mu=0, sigma=variables["stddev"], scaling=scaling_gauss)
        I = integrate.quad(gauss, a=-variables["eta_limit"], b=variables["eta_limit"])
        print("ist gauß normiert?", I)



        I2 = integrate.quad(loguni, a=variables["x_lower_limit"], b=variables["x_upper_limit"])
        print("Normierung loguni", I2)

        analytic_integral = np.zeros(shape=nr_bins)
        ml_integral = np.zeros(shape=nr_bins)
        quadratic_analytic_integral = np.zeros(shape=nr_bins)
        quadratic_ml_integral = np.zeros(shape=nr_bins)
        analytic_stddev = np.zeros(shape=nr_bins)
        ml_stddev = np.zeros(shape=nr_bins)
        eta = np.zeros(shape=nr_bins)
        print("wir sind vor der schleife die die MC-Integration macht")
        # ratio mit einbeziehen
        ratio = (variables["x_total"]/variables["total_data"])
        for i,eta_value in enumerate(features_eta_constant.keys()):
            # Ich denke ich benötige die gauss verteilugn
            scaling = 1/(er_fc(eta_interval[i+1]) - er_fc(eta_interval[i])) * (eta_interval[i+1] - eta_interval[i])
            analytic_integral[i] = float(tf.math.reduce_mean(labels_eta_constant[eta_value][:,0] /
                                                         (scaling * ratio * loguni(features_eta_constant[eta_value][:,0])
                                                          * loguni(features_eta_constant[eta_value][:,1]) * gauss(features_eta_constant[eta_value][:,2]))))
            quadratic_analytic_integral[i] = float(tf.math.reduce_mean(tf.math.square(labels_eta_constant[eta_value][:,0] /
                                                         ((scaling * ratio * loguni(features_eta_constant[eta_value][:,0]) *
                                                   loguni(features_eta_constant[eta_value][:,1]) * gauss(features_eta_constant[eta_value][:,2]))))))
            analytic_stddev[i] = np.sqrt((quadratic_analytic_integral[i] - analytic_integral[i]**2) * 1/(len(labels_eta_constant[eta_value][:,0])-1))

            if with_ml:
                ml_integral[i] = float(tf.math.reduce_mean((predictions[eta_value][:,0])/
                                                      (scaling * ratio * loguni(features_eta_constant[eta_value][:,0]) *
                                                       loguni(features_eta_constant[eta_value][:,1]) * gauss(features_eta_constant[eta_value][:,2]))))
                quadratic_ml_integral[i] = float(tf.math.reduce_mean(tf.math.square(predictions[eta_value][:,0] / (scaling * ratio * loguni(features_eta_constant[eta_value][:,0]) *
                                                       loguni(features_eta_constant[eta_value][:,1]) * gauss(features_eta_constant[eta_value][:,2])))))

                ml_stddev[i] = np.sqrt((quadratic_ml_integral[i] - ml_integral[i]**2) * 1/(len(predictions[eta_value][:,0])-1))

                ml_integrals.append(ml_integral)
                ml_stddevs.append(ml_stddev)

            eta[i] = (eta_interval[i+1] - eta_interval[i])/2 + eta_interval[i]

            if (i + 1) % 2 == 0:
                print(i+1, "/", nr_bins)


        # Berechnungen den Listen hinzufuegen
        analytic_integrals.append(analytic_integral)
        analytic_stddevs.append(analytic_stddev)

        #Speicher wieder freigeben
        del features_eta_constant
        del labels_eta_constant
        del predictions

    analytic_integrals = np.array(analytic_integrals)
    ml_integrals = np.array(ml_integrals)
    analytic_stddevs = np.array(analytic_stddevs)
    ml_stddevs = np.array(ml_stddevs)
    print("eta", eta)
    print("analytic_integrals", analytic_integrals)
    print("ml_integrals", ml_integrals)
    print("analytic_stds", analytic_stddevs)
    print("ml_stdds", ml_stddevs)
    #Mitteln und Fehler fortpflanzen
    analytic_integral = np.mean(analytic_integrals, axis=0)
    ml_integral = np.mean(ml_integrals, axis=0)
    analytic_stddev = 1/len(analytic_stddevs) * np.sqrt(np.sum(np.square(analytic_stddevs), axis=0))
    if with_ml:
        ml_stddev = 1/len(analytic_stddevs) * np.sqrt(np.sum(np.square(ml_stddevs), axis=0))
    print("len analytic integrals", len(analytic_integrals))
    print("len ml integrals", len(ml_integrals))
    print("ml integrals[:,0]", ml_integrals[:,0])
    print("analytic_integrals[:,0]", analytic_integrals[:,0], len(analytic_integrals[:,0]))
    analytic_stddev_stat = np.std(analytic_integrals, axis=0, ddof=1) * 1/np.sqrt(len(analytic_integrals))
    ml_stddev_stat = np.std(ml_integrals, axis=0, ddof=1) * 1/np.sqrt(len(analytic_integrals))



    print("ml_integral in pb ", MC.gev_to_pb(ml_integral), "stddev in pb", MC.gev_to_pb(ml_stddev), "statistical stddev in pb", MC.gev_to_pb(ml_stddev_stat))
    print("analytic integral in pb:", MC.gev_to_pb(analytic_integral),"stddev in pb", MC.gev_to_pb(analytic_stddev), "statistical stddev in pb", MC.gev_to_pb(analytic_stddev_stat))

    order = np.argsort(eta)
    eta = np.array(eta)[order]
    analytic_integral = np.array(analytic_integral)[order]
    if not with_ml:
        ml_integral = analytic_integral
        ml_stddev = analytic_stddev
    ml_integral = np.array(ml_integral)[order]

    ml.make_MC_plot(x=eta, analytic_integral=MC.gev_to_pb(analytic_integral), ml_integral=MC.gev_to_pb(ml_integral), xlabel=r"$\eta$",
                    ylabel=r"$\frac{d\sigma}{d\eta}[\mathrm{pb}]$", save_path=save_path, name="xIntegration_correct_std",
                    analytic_errors=MC.gev_to_pb(analytic_stddev), ml_errors=MC.gev_to_pb(ml_stddev), ylims=(0.98, 1.02))
    plt.show()
    ml.make_MC_plot(x=eta, analytic_integral=MC.gev_to_pb(analytic_integral), ml_integral=MC.gev_to_pb(ml_integral), xlabel=r"$\eta$",
                    ylabel=r"$\frac{d\sigma}{d\eta}[\mathrm{pb}]$", save_path=save_path, name="xIntegration_correct_stat_std",
                    analytic_errors=MC.gev_to_pb(analytic_stddev_stat), ml_errors=MC.gev_to_pb(ml_stddev_stat), ylims=(0.98, 1.02))
    plt.show()

    # TODO funktioniert das hier schon?


    print("ml_integral in pb ", MC.gev_to_pb(ml_integral), "stddev in pb", MC.gev_to_pb(ml_stddev), "statistical stddev in pb", MC.gev_to_pb(ml_stddev_stat))
    print("analytic_integral in GeV", analytic_integral, "analytic integral in pb:", MC.gev_to_pb(analytic_integral), "stddev in pb", MC.gev_to_pb(analytic_stddev_stat))

    analytic_integral[np.isnan(analytic_integral)] = 0
    print(analytic_integral)
    sigma_total = integrate.trapezoid(y=MC.gev_to_pb(analytic_integral),x=eta)
    print("sigma_total", sigma_total)
    sigma_total = 0
    for i in range(len(analytic_integral)):
        sigma_total += (eta_interval[i+1] - eta_interval[i]) * analytic_integral[i]
    print("sigma_totla andere variante", MC.gev_to_pb(sigma_total))

if __name__ == "__main__":
    main()

