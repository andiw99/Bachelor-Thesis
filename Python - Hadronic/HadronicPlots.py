import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import numpy as np

#Daten einlesen
#set_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_3D/"
set_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/RandomSamples/"
#eta constant
hadronic_data_eta_constant = pd.read_csv( set_path + "eta_x_2_constant")
#x-constant beide x_1, x_2
hadronic_data_x_constant = pd.read_csv(set_path + "x_constant")
#3D-plot, x_2 constant
hadronic_data_x_2_constant = pd.read_csv(set_path + "x_2_constant__3D")

#eta constant
order = np.argsort(hadronic_data_eta_constant["x_1"])
x_list = np.array(hadronic_data_eta_constant["x_1"])[order]
WQ_list = np.array(hadronic_data_eta_constant["WQ"])[order]
plt.plot(x_list, WQ_list)
plt.xlabel("x")
plt.show()

#x-constant
order = np.argsort(hadronic_data_x_constant["eta"])
eta_list = np.array(hadronic_data_x_constant["eta"])[order]
WQ_list = np.array(hadronic_data_x_constant["WQ"])[order]
plt.plot(eta_list, WQ_list)
plt.xlabel("eta")
plt.show()

#3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

eta = hadronic_data_x_2_constant["eta"]
x_1 = hadronic_data_x_2_constant["x_1"]
#x_1[x_1>0.5]=np.nan
#x_1[x_1<0.1]=np.nan
WQ = hadronic_data_x_2_constant["WQ"]
#plot the surface
surf = ax.plot_trisurf(eta, x_1, WQ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("eta")
ax.set_ylabel("x_1")
ax.set_zlabel("WQ")
ax.set_zscale("log")
plt.tight_layout()
ax.view_init(10, 50)
plt.show()
for angle in range(0,360):
    if angle % 60 ==0:
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.01)

