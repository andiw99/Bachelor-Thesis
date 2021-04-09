from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#Daten einlesen
diff_WQ_eta_data  = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/diff_WQ_eta_data")
theta_data = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/ThetaData")
theta_test_data = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TestThetaData")


#Listen erzeugen
eta = diff_WQ_eta_data["Eta"]
diff_WQ_eta = diff_WQ_eta_data["WQ"]

theta = theta_data["Theta"]
diff_WQ_theta = theta_data["WQ"]

plt.plot(eta, diff_WQ_eta)
plt.ylabel(r"$\frac{d\sigma}{d\eta}$")
plt.show()

Graph_theta = plt.plot(theta, diff_WQ_theta)
plt.title("Train Data")
plt.ylabel(r"$\frac{d\sigma}{d\theta}$")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()

order = np.argsort(theta_test_data["Theta"] )
theta = np.array(theta_test_data["Theta"])[order]
diff_WQ_theta= np.array(theta_test_data["WQ"])[order]
print(theta_test_data["Theta"])
print(theta)

Graph_theta = plt.plot(theta, diff_WQ_theta)
plt.ylabel(r"$\frac{d\sigma}{d\theta}$")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()


