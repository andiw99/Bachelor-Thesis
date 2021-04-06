import pandas as pd
import numpy as np



#Daten präparieren
diff_WQ_eta = []
diff_WQ_theta = []
eta = []
theta = []


#Listen füllen
for i in np.linspace(-3, 3, 60000):
    x = diff_WQ_eta.append(1+(np.tanh(i))**2)
    eta.append(x)

diff_WQ_eta_data = pd.DataFrame(
    {
        "Eta": eta,
        "WQ": diff_WQ_eta
    }
)
diff_WQ_eta_data.to_csv("diff_WQ_eta_data", index=False)
print(diff_WQ_eta_data)