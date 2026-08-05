# Test file
# Goal : check the round trip continued_fraction addition / subtraction, in
#        both the scalar and matrix (block) cases: build two random continued
#        fractions A and B, form C = A + B, then subtract B back from C and
#        check that the result matches A again.
# --------------------------------------------------------------------------------
import warnings
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

# --------------------------------------------------------------------------------
# regression test: a pole shared between the two operands of
# subtract_to_continued_fraction, but reconstructed independently on each side
# (so it differs by a round-off-sized amount, as happens e.g. in
# periodization_CDMFT_PY.py between the cluster Green function's MCF and the
# exactly-computed hybridization function). Without merging, pooling two such
# near- but not exactly-degenerate poles is ill-conditioned and leaves a
# spurious extra pole (with an unphysical, e.g. negative, residue) instead of
# the clean cancellation the exact math predicts; merge_tol (default 1e-8)
# snaps them together beforehand to avoid this.

# scalar case ---------------------------------------------------------------

Wa2 = np.array([-1.5, -0.3, 0.9])
Ra2 = np.array([0.6, 0.3, 0.5])
lehmann_A2 = lehmann(Wa2, Ra2)

Wb2 = np.array([0.7, 2.0])          # 0.7 is the pole that will be shared
Rb2 = np.array([0.4, 0.2])
lehmann_B2 = lehmann(Wb2, Rb2)

self_2 = lehmann_A2.add_to(lehmann_B2)     # pools A2's and B2's poles exactly

Wx2 = Wb2.copy()
Wx2[0] += 1e-10                            # the shared pole at 0.7, off by round-off
lehmann_X2 = lehmann(Wx2, Rb2)

D_unmerged = self_2.subtract_to_continued_fraction(lehmann_X2, merge_tol=0.0)
D_merged = self_2.subtract_to_continued_fraction(lehmann_X2)   # default merge_tol=1e-8

print("\nnear-degenerate pole regression (scalar):")
print("unmerged: n_poles =", D_unmerged.A.size, " has negative B :", np.any(D_unmerged.B < 0))
print("merged  : n_poles =", D_merged.A.size, " has negative B :", np.any(D_merged.B < 0))
assert D_unmerged.A.size > D_merged.A.size      # unmerged leaves spurious extra poles
assert np.any(D_unmerged.B < 0)                 # ... one with an unphysical negative residue
assert not np.any(D_merged.B < 0)

err_unmerged = max_diff(D_unmerged.evaluate(z), lehmann_A2.evaluate(z))
err_merged = max_diff(D_merged.evaluate(z), lehmann_A2.evaluate(z))
print("unmerged eval error vs A2 :", err_unmerged)
print("merged   eval error vs A2 :", err_merged)
assert err_merged < 1e-9
assert err_merged < err_unmerged

# matrix case -----------------------------------------------------------------
# uses its own RNG (independent of the module-level `rng` state) so this
# regression stays deterministic regardless of what runs before it.

rng2 = np.random.default_rng(2)

WmA2 = np.array([-1.7, -0.4, 0.3, 1.9])
QmA2 = rng2.standard_normal((L, 4)) + 1j * rng2.standard_normal((L, 4))
lehmann_Am2 = lehmann_matrix(WmA2, QmA2)

WmB2 = np.array([-1.1, 0.7, 1.2, 2.3])      # 0.7 is the pole that will be shared
QmB2 = rng2.standard_normal((L, 4)) + 1j * rng2.standard_normal((L, 4))
lehmann_Bm2 = lehmann_matrix(WmB2, QmB2)

self_m2 = lehmann_Am2.add_to(lehmann_Bm2)

WmX2 = WmB2.copy()
WmX2[1] += 1e-10                            # the shared pole at 0.7, off by round-off
lehmann_Xm2 = lehmann_matrix(WmX2, QmB2)

with warnings.catch_warnings(record=True) as caught_unmerged:
    warnings.simplefilter("always")
    Dm_unmerged = self_m2.subtract_to_continued_fraction(lehmann_Xm2, merge_tol=0.0)
with warnings.catch_warnings(record=True) as caught_merged:
    warnings.simplefilter("always")
    Dm_merged = self_m2.subtract_to_continued_fraction(lehmann_Xm2)   # default merge_tol=1e-8

print("\nnear-degenerate pole regression (matrix):")
print("unmerged triggered a partial-breakdown warning :", any("partial breakdown" in str(w.message) for w in caught_unmerged))
print("merged   triggered a partial-breakdown warning :", any("partial breakdown" in str(w.message) for w in caught_merged))
assert any("partial breakdown" in str(w.message) for w in caught_unmerged)
assert not caught_merged

err_unmerged_m = max_diff(Dm_unmerged.evaluate(z), lehmann_Am2.evaluate(z))
err_merged_m = max_diff(Dm_merged.evaluate(z), lehmann_Am2.evaluate(z))
print("unmerged eval error vs Am2 :", err_unmerged_m)
print("merged   eval error vs Am2 :", err_merged_m)
assert err_merged_m < 1e-9
assert err_merged_m < err_unmerged_m

print("\ntest_CF_addsub: all checks passed")
