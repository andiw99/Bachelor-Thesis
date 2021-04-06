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
#Trainingsdaten als Grid
xGrid = np.linspace(0.05, np.pi-0.05, 60000)
train_WQ = (1+np.cos(xGrid)**2)/(np.sin(xGrid)**2)
train_theta = list(xGrid)
train_WQ = list(train_WQ)

#Testdaten random uniform
xRand = np.random.uniform(low=0.07, high= np.pi - 0.07, size=6000)
test_WQ = (1+np.cos(xRand)**2)/(np.sin(xRand)**2)
test_WQ = list(test_WQ)
test_theta = list(xRand)

diff_WQ_theta_data = pd.DataFrame(
    {
        "Theta": train_theta,
        "WQ": train_WQ
    }
)

test_data = pd.DataFrame(
    {
        "Theta": test_theta,
        "WQ": test_WQ
    }
)

diff_WQ_theta_data.to_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/ThetaData", index=False)
test_data.to_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/TestThetaData", index=0)
print(diff_WQ_theta_data)