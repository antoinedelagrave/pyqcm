# Test file
# Goal : check the round trip continued_fraction addition / subtraction, in
#        both the scalar and matrix (block) cases: build two random continued
#        fractions A and B, form C = A + B, then subtract B back from C and
#        check that the result matches A again.
# --------------------------------------------------------------------------------
import numpy as np
from pyqcm.green_structure import lehmann, lehmann_matrix

# --------------------------------------------------------------------------------
# random test frequencies (away from the real axis, where all functions below
# are analytic)

rng = np.random.default_rng(0)
z = np.array([0.3 + 0.5j, -1.2 + 0.8j, 2.1 - 0.4j, 0.0 + 1.3j, 5.0 + 0.2j, -3.3 - 0.6j])

def max_diff(f, g):
    return np.max(np.abs(np.asarray(f) - np.asarray(g)))

# --------------------------------------------------------------------------------
# A and B: two independent random Lehmann representations, converted to their
# continued-fraction (Jacobi) form -- this is the "two continued fractions" A, B

Wa = np.sort(rng.uniform(-2, 2, 6))
Ra = rng.uniform(0.1, 1.0, 6)
lehmann_A = lehmann(Wa, Ra)
cf_A = lehmann_A.to_continued_fraction()

Wb = np.sort(rng.uniform(-2, 2, 5))
Rb = rng.uniform(0.1, 1.0, 5)
lehmann_B = lehmann(Wb, Rb)
cf_B = lehmann_B.to_continued_fraction()

print("continued fraction A:\n", cf_A)
print("continued fraction B:\n", cf_B)

err = max_diff(cf_A.evaluate(z), lehmann_A.evaluate(z))
print("A: continued_fraction vs lehmann : max|diff| =", err)
assert err < 1e-9

err = max_diff(cf_B.evaluate(z), lehmann_B.evaluate(z))
print("B: continued_fraction vs lehmann : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# C = A + B (continued_fraction.add_to)

cf_C = cf_A.add_to(cf_B)
print("continued fraction C = A + B:\n", cf_C)

ref_C = lehmann_A.evaluate(z) + lehmann_B.evaluate(z)
err = max_diff(cf_C.evaluate(z), ref_C)
print("C = A + B                  : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# D = C - B (lehmann.subtract_to_continued_fraction, bi-orthogonal Lanczos of
# Foley, these, Annexe A.2), which should be equal to A again

D = cf_C.to_lehmann().subtract_to_continued_fraction(cf_B.to_lehmann())
print("continued fraction D = C - B:\n", D)
print("D.B has negative entries (signed weight) :", np.any(D.B < 0))

err = max_diff(D.evaluate(z), lehmann_A.evaluate(z))
print("D = C - B  vs  A            : max|diff| =", err)
assert err < 1e-9

err = max_diff(D.evaluate(z), cf_A.evaluate(z))
print("D = C - B  vs  A (cf form)  : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# same round trip in the matrix (block) case: A, B, C, D are now L x L matrix
# functions, built from lehmann_matrix / continued_fraction_matrix. The total
# number of pooled poles at the subtraction step (Ma + 2*Mb) must be a
# multiple of L for lehmann_matrix.subtract_to_continued_fraction (see its
# docstring), hence the pole counts below.

L = 2
Ma, Mb = 4, 4

WmA = np.sort(rng.uniform(-2, 2, Ma))
QmA = rng.standard_normal((L, Ma)) + 1j * rng.standard_normal((L, Ma))
lehmann_Am = lehmann_matrix(WmA, QmA)
cf_Am = lehmann_Am.to_continued_fraction()

WmB = np.sort(rng.uniform(-2, 2, Mb))
QmB = rng.standard_normal((L, Mb)) + 1j * rng.standard_normal((L, Mb))
lehmann_Bm = lehmann_matrix(WmB, QmB)
cf_Bm = lehmann_Bm.to_continued_fraction()

print("\nmatrix continued fraction A:\n", cf_Am)
print("matrix continued fraction B:\n", cf_Bm)

err = max_diff(cf_Am.evaluate(z), lehmann_Am.evaluate(z))
print("A: continued_fraction_matrix vs lehmann_matrix : max|diff| =", err)
assert err < 1e-9

err = max_diff(cf_Bm.evaluate(z), lehmann_Bm.evaluate(z))
print("B: continued_fraction_matrix vs lehmann_matrix : max|diff| =", err)
assert err < 1e-9

cf_Cm = cf_Am.add_to(cf_Bm)
ref_Cm = lehmann_Am.evaluate(z) + lehmann_Bm.evaluate(z)
err = max_diff(cf_Cm.evaluate(z), ref_Cm)
print("C = A + B                  : max|diff| =", err)
assert err < 1e-9

Dm = cf_Cm.to_lehmann().subtract_to_continued_fraction(cf_Bm.to_lehmann())
print("Dm.selfadjoint (bra != ket weight) :", not Dm.selfadjoint)

err = max_diff(Dm.evaluate(z), lehmann_Am.evaluate(z))
print("D = C - B  vs  A            : max|diff| =", err)
assert err < 1e-9

err = max_diff(Dm.evaluate(z), cf_Am.evaluate(z))
print("D = C - B  vs  A (cf form)  : max|diff| =", err)
assert err < 1e-9

print("\ntest_CF_addsub: all checks passed")
