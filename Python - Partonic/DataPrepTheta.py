import pandas as pd
import numpy as np

#Konstanten definieren
Q_q = 2/3 #Ladung der zusammentreffenden Quqrks
e = 1.602e-19 #elementarLadung
p = 10.0e+3     #Impuls der Konstituenten in eV
c = 3.0e+8
s = 4*(p*e*c)**2 #Schwerpunktsenergie

#Daten präparieren
diff_WQ_eta = []
diff_WQ_theta = []
eta = []
theta = []


#Listen füllen
for i in range(1000, (int(np.pi * 10e3) - 1000)):
    x = 10e-5 * i
    diff_WQ_theta.append((1+np.cos(x)**2)/(np.sin(x)**2))
    eta.append(x)

diff_WQ_theta_data = pd.DataFrame(
    {
        "Theta": eta,
        "WQ": diff_WQ_theta
    }
)
diff_WQ_theta_data.to_csv("diff_WQ_theta_data", index=False)
print(diff_WQ_theta_data)