#! /usr/bin/env python


import lhapdf

p = lhapdf.mkPDF("CT10", 0)
p = lhapdf.mkPDF("CT10/0")

print("Was macht xfxQ2:", p.xfxQ2(21, 1e-3, 1e4))

for pid in p.flavors():
    print(p.xfxQ(pid, 0.01, 91.2))

# TODO: demonstrate looping over PDF set members
pset = lhapdf.getPDFSet("CT10nlo")
print(pset.description)
pcentral = pset.mkPDF(0)
pdfs1 = pset.mkPDFs()
pdfs2 = lhapdf.mkPDFs("CT10nlo")  # a direct way to get all the set's PDFs

import numpy as np

xs = [x for x in np.logspace(-7, 0, 5)]
qs = [q for q in np.logspace(1, 4, 4)]
gluon_xfs = np.empty([len(xs), len(qs)])
for ix, x in enumerate(xs):
    for iq, q in enumerate(qs):
        gluon_xfs[ix, iq] = p.xfxQ(21, x, q)
print(gluon_xfs)

print(lhapdf.version())
print(lhapdf.__version__)
lhapdf.pathsPrepend("/path/to/extra/pdfsets")
print(lhapdf.paths())
# ...