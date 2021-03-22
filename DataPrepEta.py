import pandas as pd
import numpy as np



#Daten präparieren
diff_WQ_eta = []
diff_WQ_theta = []
eta = []
theta = []


#Listen füllen
for i in range(-30000, 30000):
    x = 10e-5 * i
    diff_WQ_eta.append(1+(np.tanh(x))**2)
    eta.append(x)

diff_WQ_eta_data = pd.DataFrame(
    {
        "Eta": eta,
        "WQ": diff_WQ_eta
    }
)
diff_WQ_eta_data.to_csv("diff_WQ_eta_data", index=False)
print(diff_WQ_eta_data)