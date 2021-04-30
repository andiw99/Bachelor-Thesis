import pandas as pd
import numpy as np
import MC
import os
from matplotlib import pyplot as plt

#Daten präparieren
epsilon = 0.01
total = 5000
name = "PlottingData5k_ep_0.01"
importance_sampling = False
offset = 0.4
power = 4

#daten random uniform
diff_WQ_theta = MC.diff_WQ_theta(s=200**2, q=1/3)
if importance_sampling:
    custom_dist = MC.x_power_dist(power=power, offset=offset, a=epsilon, b=np.pi-epsilon, normed=True)
    theta = custom_dist.rvs(size=total)
    plt.hist(theta, bins=20)
    plt.show()
else:
    theta = np.random.uniform(low=epsilon, high= np.pi - epsilon, size=total)
WQ = diff_WQ_theta(theta)
WQ = list(WQ)
theta = list(theta)

data = pd.DataFrame(
    {
        "Theta": theta,
        "WQ": WQ
    }
)

config = pd.DataFrame(
    {
        "epsilon": epsilon,
        "total": total,
        "importance_sampling": True,
        "power": power,
        "offset": offset,
    },
    index=[0]
)

# ggf. Verzeichnis erstellen
path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/" + name
if not os.path.exists(path=path):
    os.makedirs(path)

data.to_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/" + name + "/all", index=False)
config.to_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/" + name + "/config", index=False)
print(data)