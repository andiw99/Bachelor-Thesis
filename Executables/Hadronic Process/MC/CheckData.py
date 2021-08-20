import ml
import MC
import numpy as np
import os


def main():
    # Daten einlesen
    dataset_paths = []
    directory = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/MC10M/"
    datasets = os.listdir(directory)
    for dataset in datasets:
        if dataset != "conifg":
            dataset_paths.append(directory + dataset + "/")
    dataset_paths.append("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/Bachelor-Arbeit/Files/Hadronic/Data/MC50M_I")
    label_name = "WQ"

    x_1_means = []
    x_2_means = []
    eta_means = []
    for data_path in dataset_paths:
        features, labels = MC.data_handling(data_path=data_path + "/all", label_name=label_name)

        # Kenndaten ausspucken
        x_1_mean = np.mean(features[:,0])
        x_2_mean = np.mean(features[:,1])
        eta_mean = np.mean(features[:,2])
        print("Mittlwert x_1", x_1_mean, "stddev x_1", np.std(features[:,0]), "x_1_min", np.min(features[:,0]), "x_1_max", np.max(features[:,0]))
        print("Mittlwert x_2", x_2_mean, "stddev x_2",
              np.std(features[:, 1]), "x_2_min", np.min(features[:,1]), "x_2_max", np.max(features[:,1]))
        print("Mittlwert eta", eta_mean, "stddev eta",
              np.std(features[:, 2]), "eta_min", np.min(features[:,2]), "eta_max", np.max(features[:,2]))
        x_1_means.append(x_1_mean)
        x_2_means.append(x_2_mean)
        eta_means.append(eta_mean)

    print("mean of means x_1",np.mean(x_1_means))
    print("mean of means x_2",np.mean(x_2_means))
    print("diff", np.mean(x_1_means) -np.mean(x_2_means))
    print("eta_mean of means", np.mean(eta_means))

if __name__ == "__main__":
    main()
