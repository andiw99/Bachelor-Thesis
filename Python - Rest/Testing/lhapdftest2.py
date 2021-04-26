import lhapdf as pdf
import numpy as np

PDF = pdf.mkPDF("PlottingData_CT14nnlo", 0)
quarks = {"quark": [1, 2, 3, 4],
          "charge": [-1/3,2/3, -1/3, 2/3]}
print("xfxQ2:", PDF.xfxQ2(1, 0.5, 0.001))

Q2 = 10
q = 1
E = 10

print("xMin", PDF.xMin)

PDF_Up_Proton = []
PDF_AntiUp_Proton = []
x_fraction = []
x_2 = 0.1
eta = 1.3
for x_1 in np.linspace(0.002, 0.08, 100):
    x_fraction.append(x_1)
    if x_1 < PDF.xMin:
        x_1 = PDF.xMin
    PDF_q = 0
    for q in quarks["quark"]:
        PDF_q +=(((quarks["charge"][q - 1]) ** 4)/(192 * np.pi * x_1 * x_2 * E ** 2)) * \
                ((np.abs(PDF.xfxQ2(q, x_1, Q2) * PDF.xfxQ2(-q, x_2, Q2)) + np.abs(PDF.xfxQ2(-q, x_1, Q2) * PDF.xfxQ2(q, x_2, Q2)))/(x_1 * x_2)) \
                * (1 + (np.tanh(eta + 1 / 2 * np.log(x_2 / x_1))) ** 2)
    PDF_Up_Proton.append(PDF_q)
            #PDF_AntiUp_Proton.append(((np.abs(PDF.xfxQ2(q, x_1, Q2) * PDF.xfxQ2(-q, x_2, Q2)) + np.abs(PDF.xfxQ2(-q, x_1, Q2) * PDF.xfxQ2(q, x_2, Q2)))/(x_1 * x_2)))


from matplotlib import pyplot as plt
plt.plot(x_fraction, PDF_Up_Proton)
plt.xscale("linear")
plt.yscale("log")
plt.xlabel("Up-Quark")
plt.show()
"""
plt.plot(x_fraction, PDF_AntiUp_Proton)
plt.xscale("linear")
plt.yscale("log")
plt.xlabel("Anti-Up-Quark")
plt.show()
"""