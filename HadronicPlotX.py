from matplotlib import pyplot as plt
import pandas as pd

#Daten einlesen
hadronic_WQ_data = pd.read_csv("hadronic_WQ_data")

#Listen erzeugen
x_1 = hadronic_WQ_data["x_1"]
x_2 = hadronic_WQ_data["x_2"]
eta = hadronic_WQ_data["eta"]
WQ = hadronic_WQ_data["WQ"]

#Ich möchte WQ angucken für festes eta=1, x_2 = 0.2
x_variable = []
WQ_x = []
step = 0
for i in range(len(x_1)):
    step +=1
    if eta[i] == 1.005 and x_2[i] == 0.2:
        x_variable.append(x_1[i])
        WQ_x.append(WQ[i])
    if step % 250000 == 0:
        print(step,"/",len(x_1))

print(len(x_variable), x_variable)
print(WQ_x)

plt.plot(x_variable, WQ_x)
plt.xlim(0.01,1)
plt.ylim(0,0.06)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$\frac{d³\sigma}{d\eta dx_1 dx_2}$")
plt.text(0.7, 0.055, r"$x_2 = 0.2, \eta=1$")
plt.show()

