import lhapdf as pdf


PDF = pdf.mkPDF("NNPDF30_nnlo_as_0118", 0)

print("xfxQ2:", PDF.xfxQ2(1, 0.5, 0.001))

Q2 = 10

PDF_Up_Proton = []
x_fraction = []
for i in range(0, 10000):
    x = i/10000
    x_fraction.append(x)
    PDF_Up_Proton.append(PDF.xfxQ2(2, x, Q2))


from matplotlib import pyplot as plt
plt.plot(x_fraction, PDF_Up_Proton)
plt.ylim(0, 1)
plt.show()