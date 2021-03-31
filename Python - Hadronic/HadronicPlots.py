import pandas as pd
from matplotlib import pyplot as plt


hadronic_data_eta_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/hadronic__eta_constant__1.0133779264214047")
plt.plot(hadronic_data_eta_constant["x_1"], hadronic_data_eta_constant["WQ"])
plt.xlim(0.0075, 0.2)
plt.ylim(0, 0.08)
plt.show()

hadronic_data_x_constant = pd.read_csv("/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Hadronic/HadronicData/hadronic__x_constant__0.2")
plt.plot(hadronic_data_x_constant["eta"], hadronic_data_x_constant["WQ"])
plt.show()