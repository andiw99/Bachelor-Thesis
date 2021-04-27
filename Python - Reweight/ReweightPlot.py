import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import numpy as np

show_3D_plots = True

#Daten einlesen
set_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Reweight/Data/TrainingData200k_cut_x_08/"
#x_2 constant
data_x_2_constant = pd.read_csv( set_path + "x_2_constant")
#x-constant beide x_1, x_2
data_x_1_constant = pd.read_csv(set_path + "x_1_constant")
#3D-plot, x_2 constant
if show_3D_plots:
    all = pd.read_csv(set_path + "all")

#x_2 constant
order = np.argsort(data_x_2_constant["x_1"])
x_list = np.array(data_x_2_constant["x_1"])[order]
reweight_list = np.array(data_x_2_constant["reweight"])[order]
print("minimum bei:", np.min(reweight_list))
print("maximum bei:", np.max(reweight_list))
#exit()
plt.plot(x_list, reweight_list)
plt.xlabel(r"$x_1$")
plt.show()

#x_1 constant
order = np.argsort(data_x_1_constant["x_2"])
x_2_list = np.array(data_x_1_constant["x_2"])[order]
reweight_list = np.array(data_x_1_constant["reweight"])[order]
plt.plot(x_2_list, reweight_list)
plt.xlabel(r"$x_2$")
plt.show()

#3D plot
if show_3D_plots:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    #x_1[x_1>0.5]=np.nan
    #x_1[x_1<0.1]=np.nan
    reweight = np.array(all["reweight"])
    print("min bei", np.min(reweight))
    print("max bei", np.max(reweight))
    x_1 = np.array(all["x_1"])
    x_2 = np.array(all["x_2"])
    #plot the surface
    surf = ax.plot_trisurf(x_2, x_1, reweight, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel("x_2")
    ax.set_ylabel("x_1")
    ax.set_zlabel("reweight")
    plt.tight_layout()
    ax.view_init(10, 50)
    plt.show()
    for angle in range(0,360):
        if angle % 60 ==0:
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.01)

