import lhapdf as pdf
from matplotlib import pyplot as plt

MadePDF = pdf.mkPDF("NNPDF30_nnlo_as_0118", 0)
print("MadePDF:", MadePDF)

print("Was macht xfxQ2:", MadePDF.xfxQ2(2, 0.001, 0.001))
print("Was macht xfxQ2:", MadePDF.xfxQ2(2, 0.5, 0.001))

#Viererimpulsübertrag festlegen
Q2 = 0.001
"""
PDF_Up_Proton = []
x_fraction = []
for i in range(1000):
    x = i/1000
    x_fraction.append(x)
    PDF_Up_Proton.append(MadePDF.xfxQ2(2, x, Q2))
print()

plt.plot(x_fraction, PDF_Up_Proton)
plt.show()
"""