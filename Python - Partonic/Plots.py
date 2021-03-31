from matplotlib import pyplot as plt
import pandas as pd

#Daten einlesen
diff_WQ_eta_data  = pd.read_csv("diff_WQ_eta_data")
diff_WQ_theta_data = pd.read_csv("diff_WQ_theta_data")

#Listen erzeugen
eta = diff_WQ_eta_data["Eta"]
diff_WQ_eta = diff_WQ_eta_data["WQ"]

theta = diff_WQ_theta_data["Theta"]
diff_WQ_theta = diff_WQ_theta_data["WQ"]

plt.plot(eta, diff_WQ_eta)
plt.ylabel(r"$\frac{d\sigma}{d\eta}$")
plt.show()

Graph_theta = plt.plot(theta, diff_WQ_theta)
plt.ylabel(r"$\frac{d\sigma}{d\theta}$")
plt.ylim([0, 50])
plt.xlim([0.01, 3.14])
plt.show()

