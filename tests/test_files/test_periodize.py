# Test file
# Goal : check lattice_model.periodize_matrix() and lattice_model.periodize_vector()
#        by printing their output and checking their mutual consistency:
#        for A = outer(w*, v), periodize_matrix(A, k) must equal
#        Lc * outer(periodize_vector(w, k)*, periodize_vector(v, k))
#        (the conjugate/transpose bookkeeping follows the same implicit-transpose
#        convention used throughout the API for square dim_GF matrices), and both
#        must satisfy a sum rule at k=0.
# --------------------------------------------------------------------------------
import numpy as np

import pyqcm

# ----------------------------------------------------------------
# 2-cluster graphene model with bath (same model as test_2clusters.py)
# ----------------------------------------------------------------

clus = pyqcm.cluster_model(4, n_bath=6, name='clus')
clus.new_operator('eb1', 'one-body', [(5, 5, 1), (15, 15, 1), (7, 7, 1), (17, 17, 1), (9, 9, 1), (19, 19, 1)])
clus.new_operator('eb2', 'one-body', [(6, 6, 1), (16, 16, 1), (8, 8, 1), (18, 18, 1), (10, 10, 1), (20, 20, 1)])
clus.new_operator('tb1', 'one-body', [(1, 5, 1), (11, 15, 1), (2, 7, 1), (12, 17, 1), (3, 9, 1), (13, 19, 1)])
clus.new_operator('tb2', 'one-body', [(1, 6, 1), (11, 16, 1), (2, 8, 1), (12, 18, 1), (3, 10, 1), (13, 20, 1)])

clus0 = pyqcm.cluster(clus, ((-1, -1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)), pos=(1, 0, 0))
clus1 = pyqcm.cluster(clus, ((1, 1, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 0)), pos=(-1, 0, 0))

model = pyqcm.lattice_model('graphene_4_2C', [clus0, clus1], ((4, 2, 0), (2, -2, 0)), lattice=((1, -1, 0), (2, 1, 0)))
model.set_basis([(1, 0, 0), [-0.5, np.sqrt(3) / 2, 0]])

model.interaction_operator('U')
model.hopping_operator('t', (-1, 0, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (0, -1, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (1, 1, 0), 1, orbitals=(1, 2))

model.set_target_sectors(['R0:N10:S0'] * 2)
model.set_parameters("""
    U=4.0
    mu=0.5*U
    t=1.0
    t_1=1.1
    t_2=0.9
    tb1_1=0.5
    tb2_1=1.0*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
    tb1_2=-1.0*tb1_1
    tb2_2=-1.0*tb2_1
    eb1_2=1.0*eb1_1
    eb2_2=1.0*eb2_1
""")

I = pyqcm.model_instance(model)  # triggers model.finalize(), setting dimGF/dimGF_red

d = model.dimGF
dr = model.dimGF_red
Lc = d / dr
print("dim_GF =", d, " dim_GF_red =", dr, " Lc =", Lc)

# --------------------------------------------------------------------------------
# Lehmann representation of the hybridization function

rng = np.random.default_rng(0)
v = rng.normal(size=d) + 1j * rng.normal(size=d)
w = rng.normal(size=d) + 1j * rng.normal(size=d)
k = np.array([0.21, -0.34, 0.0])

Vp = model.periodize_vector(v, k)
Wp = model.periodize_vector(w, k)
print("\nperiodize_vector(v, k) =\n", Vp)
print("periodize_vector(w, k) =\n", Wp)

A = np.outer(w.conj(), v)
Ap = model.periodize_matrix(A, k)
print("\nperiodize_matrix(outer(w*, v), k) =\n", Ap)

Ap_expected = Lc * np.outer(Wp.conj(), Vp)
err = np.max(np.abs(Ap - Ap_expected))
print("max|diff| with Lc*outer(periodize_vector(w)*, periodize_vector(v)) =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# sum rule at k=0 : sum of the periodized quantity equals sum of the input / Lc

k0 = np.array([0.0, 0.0, 0.0])

v0p = model.periodize_vector(v, k0)
err0 = abs(np.sum(v0p) - np.sum(v) / Lc)
print("\nk=0 vector sum-rule |diff| =", err0)
assert err0 < 1e-9

A0p = model.periodize_matrix(A, k0)
err0m = abs(np.sum(A0p) - np.sum(A) / Lc)
print("k=0 matrix sum-rule |diff| =", err0m)
assert err0m < 1e-9

print("\ntest_periodize: all checks passed")
