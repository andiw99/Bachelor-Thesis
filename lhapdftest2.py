import lhapdf as pdf


PDF = pdf.mkPDF("NNPDF30_nnlo_as_0118", 0)

print("xfxQ2:", PDF.xfxQ2(1, 0.5, 0.001))

Q2 = 10

print("xMin", PDF.xMin)

PDF_Up_Proton = []
PDF_AntiUp_Proton = []
x_fraction = []
for i in range(0, 1000000):
    x = i/1000000
    x_fraction.append(x)
    if x < PDF.xMin:
        PDF_Up_Proton.append(PDF.xfxQ2(2, PDF.xMin, Q2))
        PDF_AntiUp_Proton.append(PDF.xfxQ2(-2, PDF.xMin, Q2))
    else:
        PDF_Up_Proton.append(PDF.xfxQ2(2, x, Q2))
        PDF_AntiUp_Proton.append(PDF.xfxQ2(-2, x, Q2))


from matplotlib import pyplot as plt
plt.plot(x_fraction, PDF_Up_Proton)
plt.xscale("log")
plt.yscale("linear")
plt.ylim(0, 10)
plt.xlabel("Up-Quark")
plt.show()

plt.plot(x_fraction, PDF_AntiUp_Proton)
plt.xscale("log")
plt.yscale("linear")
plt.ylim(0, 10)
plt.xlabel("Anti-Up-Quark")
plt.show()