import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import numpy as np

#eta constant
hadronic_data_eta_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/hadronic__eta_constant__1.0133779264214047")
plt.plot(hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["WQ"])
plt.xlim(0.0075, 0.2)
plt.ylim(0, 0.08)
plt.show()

#x-constant
hadronic_data_x_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/hadronic__x_constant__0.2")
plt.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
plt.show()

#3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
hadronic_data_x_2_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/log_neg_3D/logarithmic_hadronic_data_no_negative__x_2_constant__3D")

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
ax.set_ylim(0.1, 0.5)
plt.tight_layout()
plt.show()
for angle in range(0,360):
    if angle % 60 ==0:
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.01)

