from matplotlib import pyplot as plt
import pandas as pd
import time

time1 = time.time()

# Daten einlesen
hadronic_WQ_data = pd.read_csv("less_hadronic_WQ_data")

# Listen erzeugen
x_1 = hadronic_WQ_data["x_1"]
x_2 = hadronic_WQ_data["x_2"]
eta = hadronic_WQ_data["eta"]
WQ = hadronic_WQ_data["WQ"]

# ich möchte WQ angucken für festes x_1=x_2=0,2
eta_variable = []
WQ_eta = []
step = 0
for i in range(len(x_1)):
    step += 1
    if x_1[i] == 0.2 and x_2[i] == 0.2:
        eta_variable.append(eta[i])
        WQ_eta.append(WQ[i])
    if step % 250000 == 0:
        print(step, "/", len(x_1))

plt.plot(eta_variable, WQ_eta)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\frac{d³\sigma}{d\eta dx_1 dx_2}$")
plt.text(1, 1.2e-4, r"$x_1 = 0.2, $"
                    r"$x_2 = 0.2$")

plt.tight_layout()
plt.show()

time2 = time.time()

print("Zeit, um Plot zu generieren:", time2 - time1)
