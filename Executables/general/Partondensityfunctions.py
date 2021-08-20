"""
zeichnet Partondichtefunktionen
"""
from matplotlib import pyplot as plt
import lhapdf as pdf
import numpy as np


def main():
    print(__doc__)

    # Intervalle festlegen
    x = 10 ** np.linspace(-10, 0, 200)
    x_small = np.linspace(0.005, 0.01, 100)
    E = 200 * np.ones(shape=200)
    E2 = 100 * np.ones(shape=200)
    E3 = 50 * np.ones(shape=200)
    E_small = 200 * np.ones(shape=100)

    PDF1 = pdf.mkPDF("CT14nnlo", 0)
    PDF2 = pdf.mkPDF("MMHT2014nnlo68cl", 0)

    pdf_values_1 = PDF1.xfxQ2(1, x, (E ** 2))
    pdf_values_2 = PDF1.xfxQ2(1, x, (E2 ** 2))
    pdf_values_3 = PDF1.xfxQ2(1, x, (E3 ** 2))


    pdf_values_small_1 = PDF1.xfxQ2(1, x_small, E_small ** 2)
    pdf_values_small_2 = PDF2.xfxQ2(1, x_small, E_small ** 2)
    print(PDF1.xMin)

    plt.plot(x, pdf_values_1, label=r"d-Quark, $Q^2 = (200 \mathrm{GeV})^2$")
    plt.plot(x, pdf_values_2, label=r"d-Quark, $Q^2 = (100 \mathrm{GeV})^2$")
    plt.plot(x, pdf_values_3, label=r"d-Quark, $Q^2 = (50 \mathrm{GeV})^2$")

    plt.xscale("log")
    plt.yscale("linear")
    plt.xlabel("x")
    plt.ylabel(r"$xf\left(x,Q^2\right)$")
    plt.legend()
    plt.grid(True, color="lightgray")
    plt.show()

    plt.plot(x_small, pdf_values_small_1, label="d-Quark, CT14nnlo, $Q^2 = (200 \mathrm{GeV})^2$")
    plt.plot(x_small, pdf_values_small_2, label="d-Quark, MMHT2014nnlo, $Q^2 = (200 \mathrm{GeV})^2$")
    plt.xlabel("x")
    plt.ylabel(r"$xf\left(x,Q^2\right)$")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.legend()
    plt.grid(True, color="lightgray")
    plt.show()

if __name__ == "__main__":
    main()
