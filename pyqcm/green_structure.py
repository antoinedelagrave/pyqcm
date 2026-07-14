import numpy as np
from numpy.polynomial import polynomial as npoly

#===============================================================================
class lehmann:
    """Lehmann representation of a scalar analytic function with poles on the real axis.

    The function has the form

        f(z) = sum_i R[i] / (z - W[i])

    where the poles W[i] lie on the real axis and the residues R[i] are positive.
    """

    def __init__(self, W, R, check_positive=True):
        """
        :param W: array-like of pole locations (real).
        :param R: array-like of residues (must be positive unless check_positive is False).
        :param check_positive: if True (default), require all residues to be positive.
        """
        self.W = np.asarray(W, dtype=float)
        self.R = np.asarray(R, dtype=float)

        if self.W.shape != self.R.shape:
            raise ValueError(
                f"W and R must have the same shape, got {self.W.shape} and {self.R.shape}"
            )
        if self.W.ndim != 1:
            raise ValueError(f"W and R must be one-dimensional, got {self.W.ndim} dimensions")
        if check_positive and np.any(self.R <= 0.0):
            raise ValueError("all residues R must be positive")

    def evaluate(self, z):
        """Evaluate the function at an arbitrary complex number (or array of).

        :param z: complex scalar or array of complex numbers.
        :return: f(z), of the same shape as z.
        """
        z = np.asarray(z, dtype=complex)
        return np.sum(self.R / (z[..., np.newaxis] - self.W), axis=-1)

    def moments(self, n):
        """Return the first n moments of the associated spectral weight.

        The spectral weight is the discrete measure A(x) = sum_i R[i] delta(x - W[i]),
        whose moments are m_k = sum_i R[i] W[i]**k. This returns m_0, m_1, ..., m_{n-1}
        (so n moments, of orders 0 through n-1). m_0 is the total weight sum(R).

        :param n: number of moments to compute.
        :return: numpy array of length n with the moments of orders 0 to n-1.
        """
        if n < 0:
            raise ValueError(f"number of moments must be non-negative, got {n}")
        return self.R @ np.vander(self.W, n, increasing=True)

    def modified_moments(self, a, b):
        """Return the modified moments nu_k = integral pi_k dA of the spectral weight.

        The pi_k are the reference polynomials defined by the three-term recurrence

            pi_{k+1}(x) = (x - a[k]) pi_k(x) - b[k] pi_{k-1}(x),   pi_0 = 1, pi_{-1} = 0

        so nu_k = sum_i R[i] pi_k(W[i]). This returns len(a)+1 moments (orders 0
        to len(a)). Using well-chosen reference polynomials (e.g. Chebyshev
        polynomials scaled to the spectral range, see
        continued_fraction.chebyshev_recurrence) makes the reconstruction by
        continued_fraction.from_modified_moments far better conditioned than the
        ordinary power moments.

        :param a: reference recurrence diagonal coefficients.
        :param b: reference recurrence off-diagonal coefficients (b[0] unused).
        :return: numpy array of len(a)+1 modified moments.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if b.size < a.size:
            raise ValueError("b must be at least as long as a")

        M = a.size + 1
        nu = np.zeros(M)
        pkm1 = np.zeros_like(self.W)   # pi_{k-1}
        pk = np.ones_like(self.W)      # pi_k, starting at pi_0 = 1
        nu[0] = self.R @ pk
        for k in range(1, M):
            pk, pkm1 = (self.W - a[k - 1]) * pk - b[k - 1] * pkm1, pk
            nu[k] = self.R @ pk
        return nu

    def to_continued_fraction(self, tol=1e-12):
        """Convert to the Jacobi continued-fraction format via the Lanczos recursion.

        The Lehmann representation is the Stieltjes transform of the discrete
        measure with mass R[i] at W[i]. Applying the Lanczos algorithm to the
        diagonal matrix diag(W) with starting vector proportional to sqrt(R)
        generates the recurrence coefficients of the orthogonal polynomials of
        that measure, which are precisely the Jacobi continued-fraction
        coefficients:

            A[k] = a_k   (diagonal / Lanczos alpha)
            B[k] = b_k   (off-diagonal / Lanczos beta), with B[0] = sqrt(sum R).

        Full reorthogonalization is used for numerical stability, and the
        recursion is truncated if it breaks down early (e.g. repeated poles).

        :param tol: threshold on the Lanczos beta below which the recursion stops.
        :return: an equivalent continued_fraction instance.
        """
        n = self.W.size
        if n == 0:
            return continued_fraction(np.empty(0), np.empty(0))

        A = np.zeros(n)
        B = np.zeros(n)
        B[0] = np.sqrt(self.R.sum())

        V = np.zeros((n, n))          # columns are the Lanczos vectors
        V[:, 0] = np.sqrt(self.R) / B[0]
        v_prev = np.zeros(n)
        beta = 0.0
        m = n
        for k in range(n):
            v = V[:, k]
            w = self.W * v - beta * v_prev
            alpha = v @ w
            A[k] = alpha
            w = w - alpha * v
            # full reorthogonalization against all previous Lanczos vectors
            w -= V[:, :k + 1] @ (V[:, :k + 1].T @ w)
            beta = np.linalg.norm(w)
            if beta < tol:
                m = k + 1
                break
            if k + 1 < n:
                B[k + 1] = beta
                v_prev = v
                V[:, k + 1] = w / beta

        return continued_fraction(A[:m], B[:m])

    def add_to(self, X):
        """Return the lehmann representation of the sum self + X.

        Adding two Lehmann functions simply pools their poles and residues, since
        sum_i R[i]/(z-W[i]) + sum_j R'[j]/(z-W'[j]) is again of the same form.

        :param X: another lehmann instance.
        :return: a lehmann representing self + X.
        """
        if not isinstance(X, lehmann):
            raise TypeError("X must be a lehmann instance")
        W = np.concatenate([self.W, X.W])
        R = np.concatenate([self.R, X.R])
        # operands are already valid; skip the check (residues may include zeros)
        return lehmann(W, R, check_positive=False)

    def __repr__(self):
        return f"lehmann(W={self.W!r}, R={self.R!r})"




#===============================================================================
class continued_fraction:
    """Jacobi continued-fraction representation of a scalar function with poles on the real axis.

    The function has the form

        f(z) = B[0]**2 / (z - A[0] - B[1]**2 / (z - A[1] - B[2]**2 / (z - A[2] - ...)))

    where A holds the diagonal coefficients a_i and B holds the off-diagonal
    coefficients b_i (B[0] is the overall weight, whose square multiplies the
    fraction; for a Green function A and B are the Lanczos coefficients). Both
    arrays have the same length.
    """

    def __init__(self, A, B):
        """
        :param A: array-like of diagonal coefficients a_i.
        :param B: array-like of off-diagonal coefficients b_i.
        """
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)

        if self.A.shape != self.B.shape:
            raise ValueError(
                f"A and B must have the same shape, got {self.A.shape} and {self.B.shape}"
            )
        if self.A.ndim != 1:
            raise ValueError(f"A and B must be one-dimensional, got {self.A.ndim} dimensions")

    def evaluate(self, z, terminator=None):
        """Evaluate the function at an arbitrary complex number (or array of).

        By default the fraction is truncated, i.e. the tail below the last floor
        is set to zero, which yields a function with discrete poles. Passing a
        terminator instead seeds the last floor with the value of the (infinite)
        tail, restoring a continuous spectrum; see square_root_terminator.

        :param z: complex scalar or array of complex numbers.
        :param terminator: optional callable t(z) returning the value of the tail
            below the last floor (default: 0, i.e. plain truncation).
        :return: f(z), of the same shape as z.
        """
        z = np.asarray(z, dtype=complex)
        f = np.zeros_like(z)
        if terminator is not None:
            f = f + np.asarray(terminator(z), dtype=complex)
        # evaluate the continued fraction from the bottom up
        for a, b in zip(self.A[::-1], self.B[::-1]):
            f = b**2 / (z - a - f)
        return f

    @staticmethod
    def square_root_terminator(a_inf=0.0, b_inf=None):
        """Return the square-root terminator for asymptotic coefficients a_inf, b_inf.

        When the coefficients settle to constants (a_k -> a_inf, b_k -> b_inf), the
        infinite tail t of the continued fraction is the fixed point of the
        recursion, t = b_inf**2 / (z - a_inf - t), i.e.

            t(z) = 0.5 * ((z - a_inf) - sqrt((z - a_inf)**2 - 4 b_inf**2)),

        with the branch chosen so that t is retarded (Im t < 0 for Im z > 0). Used
        as the terminator argument of evaluate, it replaces the discrete poles of a
        truncated fraction by the continuous band [a_inf - 2 b_inf, a_inf + 2 b_inf].

        :param a_inf: asymptotic diagonal coefficient.
        :param b_inf: asymptotic off-diagonal coefficient (required).
        :return: a callable t(z).
        """
        if b_inf is None:
            raise ValueError("b_inf must be provided")

        def terminator(z):
            z = np.asarray(z, dtype=complex)
            d = z - a_inf
            s = np.sqrt(d * d - 4.0 * b_inf**2)
            # pick the retarded branch: Im(t) opposite in sign to Im(z)
            wrong = (np.imag(s) * np.imag(d) < 0) | ((np.imag(d) == 0) & (np.imag(s) < 0))
            s = np.where(wrong, -s, s)
            return 0.5 * (d - s)

        return terminator

    @staticmethod
    def period2_terminator(b1, b2, a1=0.0, a2=0.0):
        """Return the period-2 terminator for coefficients with a two-cycle asymptote.

        For a gapped (two-band) spectrum the coefficients do not settle to single
        constants but oscillate with period 2, (a, b) -> (a1, b1), (a2, b2),
        (a1, b1), .... The infinite tail is then the fixed point of the two-step
        recursion,

            t  = b1**2 / (z - a1 - t'),   t' = b2**2 / (z - a2 - t),

        i.e. t solves the quadratic

            d1 t**2 - (d1 d2 + b1**2 - b2**2) t + b1**2 d2 = 0,   d1 = z - a1, d2 = z - a2,

        with the retarded branch (Im t < 0 for Im z > 0). Here (a1, b1) are the
        coefficients of the first omitted floor (just below the truncation) and
        (a2, b2) those of the next; b1 and b2 alternate thereafter. This restores
        the two bands of a gapped spectrum. It generalizes square_root_terminator,
        which is the period-1 case b1 = b2 = b_inf, a1 = a2 = a_inf.

        :param b1: off-diagonal coefficient of the first omitted floor.
        :param b2: off-diagonal coefficient of the next floor.
        :param a1: diagonal coefficient of the first omitted floor (default 0).
        :param a2: diagonal coefficient of the next floor (default 0).
        :return: a callable t(z).
        """
        def terminator(z):
            z = np.asarray(z, dtype=complex)
            d1 = z - a1
            d2 = z - a2
            num = d1 * d2 + b1**2 - b2**2
            s = np.sqrt(num * num - 4.0 * d1 * d2 * b1**2)
            t_minus = (num - s) / (2.0 * d1)
            t_plus = (num + s) / (2.0 * d1)
            # retarded branch: Im(t) opposite in sign to Im(z)
            return np.where(np.imag(t_minus) * np.imag(z) <= 0, t_minus, t_plus)

        return terminator

    @staticmethod
    def from_moments(m):
        """Build a continued_fraction from the moments of the spectral weight.

        Given the moments m_k = sum_i R[i] W[i]**k for k = 0 ... 2n-1, the
        Chebyshev algorithm produces the recurrence coefficients of the
        associated orthogonal polynomials, which are the Jacobi continued-fraction
        coefficients: A[k] = alpha_k and B[k] = sqrt(beta_k), with B[0] = sqrt(m_0).
        Chaining with to_lehmann() then recovers the poles and residues (Gaussian
        quadrature). From len(m) moments, n = len(m) // 2 coefficients are found.

        Warning: with ordinary power moments this reconstruction is severely
        ill-conditioned (Hankel determinants); it is only reliable for modest n,
        and a negative beta from inconsistent/noisy moments yields NaN.

        :param m: array-like of at least 2 moments (orders 0, 1, 2, ...).
        :return: a continued_fraction reproducing those moments.
        """
        m = np.asarray(m, dtype=float)
        n = m.size // 2
        if n < 1:
            raise ValueError("need at least 2 moments to recover one coefficient")

        A = np.zeros(n)
        B = np.zeros(n)
        A[0] = m[1] / m[0]
        B[0] = np.sqrt(m[0])
        alpha, beta = A[0], m[0]

        sig_km2 = np.zeros(m.size)      # sigma_{k-2}
        sig_km1 = m.copy()             # sigma_{k-1}, starting at sigma_0 = m
        for k in range(1, n):
            sig_k = np.zeros(m.size)
            l = np.arange(k, 2 * n - k)
            sig_k[l] = sig_km1[l + 1] - alpha * sig_km1[l] - beta * sig_km2[l]
            alpha = sig_k[k + 1] / sig_k[k] - sig_km1[k] / sig_km1[k - 1]
            beta = sig_k[k] / sig_km1[k - 1]
            if not beta > 0.0:
                raise ValueError(
                    f"non-positive beta ({beta:.3e}) at coefficient {k}: the moments are "
                    "inconsistent or the reconstruction has exceeded its conditioning"
                )
            A[k] = alpha
            B[k] = np.sqrt(beta)
            sig_km2, sig_km1 = sig_km1, sig_k

        return continued_fraction(A, B)

    @staticmethod
    def chebyshev_recurrence(N, xmin, xmax):
        """Recurrence coefficients of the (monic) Chebyshev polynomials on [xmin, xmax].

        Returns arrays a, b of length N defining the reference polynomials used by
        lehmann.modified_moments and from_modified_moments. On [-1, 1] the monic
        Chebyshev polynomials have a_k = 0, b_1 = 1/2, b_k = 1/4 (k >= 2); these
        are shifted to the interval center and scaled by ((xmax-xmin)/2)**2. b[0]
        is unused by the algorithms.

        :param N: number of coefficients.
        :param xmin, xmax: interval bracketing the spectral support.
        :return: (a, b) numpy arrays of length N.
        """
        center = 0.5 * (xmax + xmin)
        s2 = (0.5 * (xmax - xmin))**2
        a = np.full(N, center)
        b = np.zeros(N)
        if N > 1:
            b[1] = 0.5 * s2
        if N > 2:
            b[2:] = 0.25 * s2
        return a, b

    @staticmethod
    def from_modified_moments(nu, a, b):
        """Build a continued_fraction from modified moments (modified Chebyshev algorithm).

        Given modified moments nu_k = integral pi_k dA (k = 0 ... 2n-1) against the
        reference polynomials pi_k with recurrence coefficients a, b (see
        chebyshev_recurrence), this Sack-Donovan-Wheeler / Gautschi algorithm
        returns the Jacobi continued-fraction coefficients A[k] = alpha_k,
        B[k] = sqrt(beta_k). It reduces to from_moments when a = b = 0 (monomials),
        but with well-chosen references it is far better conditioned. Chain with
        to_lehmann() to recover the poles and residues.

        :param nu: modified moments, orders 0 ... 2n-1 (n = len(nu)//2).
        :param a: reference recurrence diagonal coefficients (length >= 2n-1).
        :param b: reference recurrence off-diagonal coefficients (length >= 2n-1).
        :return: a continued_fraction reproducing those modified moments.
        """
        nu = np.asarray(nu, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = nu.size // 2
        if n < 1:
            raise ValueError("need at least 2 modified moments to recover one coefficient")
        if a.size < 2 * n - 1 or b.size < 2 * n - 1:
            raise ValueError(f"a and b must have length >= {2 * n - 1}")

        A = np.zeros(n)
        B = np.zeros(n)
        A[0] = a[0] + nu[1] / nu[0]
        B[0] = np.sqrt(nu[0])
        alpha, beta = A[0], nu[0]

        sig_km2 = np.zeros(nu.size)    # sigma_{k-2}
        sig_km1 = nu.copy()           # sigma_{k-1}, starting at sigma_0 = nu
        for k in range(1, n):
            sig_k = np.zeros(nu.size)
            l = np.arange(k, 2 * n - k)
            sig_k[l] = (sig_km1[l + 1] - (alpha - a[l]) * sig_km1[l]
                        - beta * sig_km2[l] + b[l] * sig_km1[l - 1])
            alpha = a[k] + sig_k[k + 1] / sig_k[k] - sig_km1[k] / sig_km1[k - 1]
            beta = sig_k[k] / sig_km1[k - 1]
            if not beta > 0.0:
                raise ValueError(
                    f"non-positive beta ({beta:.3e}) at coefficient {k}: the moments are "
                    "inconsistent or the reconstruction has exceeded its conditioning"
                )
            A[k] = alpha
            B[k] = np.sqrt(beta)
            sig_km2, sig_km1 = sig_km1, sig_k

        return continued_fraction(A, B)

    def to_lehmann(self):
        """Convert to the Lehmann format by diagonalizing the Jacobi matrix.

        The coefficients define a symmetric tridiagonal Jacobi matrix J, with the
        diagonal coefficients A on the diagonal and the off-diagonal coefficients
        B[1:] on the two off-diagonals. Its eigen-decomposition J = sum_i W_i
        |psi_i><psi_i| yields

            f(z) = B[0]**2 <e_0|(z - J)^-1|e_0> = sum_i B[0]**2 psi_i[0]**2 / (z - W_i)

        so the poles are the eigenvalues and the residues are the squared first
        components of the eigenvectors, scaled by the weight B[0]**2. This is the
        inverse of lehmann.to_continued_fraction() (Golub-Welsch relation).

        :return: an equivalent lehmann instance.
        """
        n = self.A.size
        if n == 0:
            return lehmann(np.empty(0), np.empty(0))

        W, psi = np.linalg.eigh(np.diag(self.A) + np.diag(self.B[1:], 1) + np.diag(self.B[1:], -1))
        R = self.B[0]**2 * psi[0, :]**2
        # residues are non-negative by construction, but may be exactly zero
        return lehmann(W, R, check_positive=False)

    def add_to(self, X):
        """Return the continued_fraction representation of the sum self + X.

        There is no simple direct rule for adding two continued fractions, so this
        goes through the Lehmann representation: both operands are converted to
        Lehmann form, their poles and residues pooled, and the result converted
        back to a continued fraction.

        :param X: another continued_fraction instance.
        :return: a continued_fraction representing self + X.
        """
        if not isinstance(X, continued_fraction):
            raise TypeError("X must be a continued_fraction instance")
        return self.to_lehmann().add_to(X.to_lehmann()).to_continued_fraction()

    def __repr__(self):
        return f"continued_fraction(A={self.A!r}, B={self.B!r})"


#===============================================================================
class rational_fraction:
    """Rational representation f(z) = P(z) / Q(z) of a function with poles on the real axis.

    The numerator P has degree N-1 and the denominator Q has degree N, so f decays
    like 1/z at infinity (no polynomial part). Polynomials are stored as coefficient
    arrays in ascending order (the numpy.polynomial convention), i.e.

        P(z) = P[0] + P[1] z + ... + P[N-1] z**(N-1)
        Q(z) = Q[0] + Q[1] z + ... + Q[N]   z**N

    so len(Q) == len(P) + 1.
    """

    def __init__(self, P, Q):
        """
        :param P: numerator coefficients (ascending order), length N.
        :param Q: denominator coefficients (ascending order), length N+1.
        """
        self.P = np.asarray(P, dtype=float)
        self.Q = np.asarray(Q, dtype=float)

        if self.P.ndim != 1 or self.Q.ndim != 1:
            raise ValueError("P and Q must be one-dimensional coefficient arrays")
        if self.Q.size != self.P.size + 1:
            raise ValueError(
                f"expected deg(Q) = deg(P) + 1, got len(P)={self.P.size}, len(Q)={self.Q.size}"
            )
        if self.Q[-1] == 0.0:
            raise ValueError("leading coefficient of Q must be non-zero")

    def evaluate(self, z):
        """Evaluate the function at an arbitrary complex number (or array of).

        :param z: complex scalar or array of complex numbers.
        :return: f(z), of the same shape as z.
        """
        z = np.asarray(z, dtype=complex)
        return npoly.polyval(z, self.P) / npoly.polyval(z, self.Q)

    def to_lehmann(self):
        """Convert to the Lehmann format by partial-fraction decomposition.

        Since deg(P) < deg(Q) and (for this class) Q has simple real roots, f has the
        partial-fraction expansion f(z) = sum_i R[i] / (z - W[i]) with W[i] the roots
        of Q and R[i] = P(W[i]) / Q'(W[i]).

        :return: an equivalent lehmann instance.
        """
        W = npoly.polyroots(self.Q)
        R = npoly.polyval(W, self.P) / npoly.polyval(W, npoly.polyder(self.Q))
        order = np.argsort(W.real)
        return lehmann(W.real[order], R.real[order])

    @staticmethod
    def from_lehmann(lh):
        """Build a rational_fraction from a Lehmann representation.

        The denominator is the monic polynomial with the poles as roots,
        Q(z) = prod_i (z - W[i]), and the numerator is
        P(z) = sum_i R[i] prod_{j != i} (z - W[j]) = sum_i R[i] Q(z) / (z - W[i]).

        :param lh: a lehmann instance.
        :return: an equivalent rational_fraction.
        """
        W = lh.W
        R = lh.R
        Q = npoly.polyfromroots(W)
        P = np.zeros(W.size)
        for i in range(W.size):
            P += R[i] * npoly.polyfromroots(np.delete(W, i))
        return rational_fraction(P, Q)

    def add_to(self, X):
        """Return the rational_fraction representation of the sum self + X.

        Adding two rational fractions combines them over a common denominator,
        P1/Q1 + P2/Q2 = (P1 Q2 + P2 Q1) / (Q1 Q2). If self and X have denominators
        of degree N1 and N2, the result has numerator degree N1+N2-1 and denominator
        degree N1+N2, so it is again a valid rational_fraction.

        :param X: another rational_fraction instance.
        :return: a rational_fraction representing self + X.
        """
        if not isinstance(X, rational_fraction):
            raise TypeError("X must be a rational_fraction instance")
        P = npoly.polyadd(npoly.polymul(self.P, X.Q), npoly.polymul(X.P, self.Q))
        Q = npoly.polymul(self.Q, X.Q)
        return rational_fraction(P, Q)

    def __repr__(self):
        return f"rational_fraction(P={self.P!r}, Q={self.Q!r})"


#===============================================================================
# Matrix-valued (block) generalizations
#===============================================================================
#
# The classes below carry the same three representations to matrix-valued
# functions G(z), i.e. L x L matrix functions analytic off the real axis with
# the block Stieltjes / Lehmann structure
#
#     G(z) = sum_i Q[:,i] Q[:,i]^dagger / (z - W[i]),
#
# where the poles W[i] are real scalars shared by all matrix entries and the
# residues are the rank-one Hermitian positive-semidefinite matrices
# Q[:,i] Q[:,i]^dagger built from the Lehmann amplitude vectors Q[:,i] (length
# L). For L = 1 each class reduces exactly to its scalar counterpart with the
# residue R[i] = |Q[0,i]|**2. The continued fraction becomes a matrix (block)
# Jacobi continued fraction and the rational form P(z)/Q(z) keeps a scalar
# denominator Q (common to all entries) with a matrix numerator P.


def _hermitize(X):
    """Return the Hermitian part 0.5 (X + X^dagger) of a square matrix."""
    return 0.5 * (X + X.conj().T)


def _sqrtm_psd(X):
    """Hermitian positive-semidefinite square root of a Hermitian matrix X."""
    w, U = np.linalg.eigh(_hermitize(X))
    w = np.clip(w, 0.0, None)
    return (U * np.sqrt(w)) @ U.conj().T


def _surface_selfenergy(z, E, V, maxiter=100, tol=1e-14):
    """Self-energy Sigma of a semi-infinite block chain by Sancho-Rubio decimation.

    For a semi-infinite chain of blocks with on-site block E and inter-cell
    hopping V (H_{m,m+1} = V, H_{m+1,m} = V^dagger), returns the surface
    self-energy Sigma solving the fixed point Sigma = V (z I - E - Sigma)^{-1}
    V^dagger, so the surface Green function is (z I - E - Sigma)^{-1}. The
    Lopez-Sancho iterative decimation converges quadratically and, for Im z > 0,
    selects the retarded branch -- unlike the plain fixed-point iteration it
    converges inside the spectral bands as well. Fully vectorized over z.

    :param z: complex scalar or array of frequencies.
    :param E: on-site block (n x n).
    :param V: inter-cell hopping block (n x n).
    :return: Sigma, of shape z.shape + (n, n).
    """
    z = np.asarray(z, dtype=complex)
    E = np.asarray(E, dtype=complex)
    V = np.asarray(V, dtype=complex)
    n = E.shape[0]
    shape = z.shape + (n, n)
    zI = z[..., np.newaxis, np.newaxis] * np.eye(n)
    es = np.broadcast_to(E, shape).copy()
    e = np.broadcast_to(E, shape).copy()
    alpha = np.broadcast_to(V, shape).copy()
    beta = np.broadcast_to(V.conj().T, shape).copy()
    for _ in range(maxiter):
        g = np.linalg.inv(zI - e)
        agb = alpha @ g @ beta
        es = es + agb
        e = e + agb + beta @ g @ alpha
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if np.max(np.abs(alpha)) + np.max(np.abs(beta)) < tol:
            break
    return es - E


def _invsqrtm_pd(X, tol=1e-13):
    """Inverse Hermitian square root of a Hermitian positive-definite matrix X."""
    w, U = np.linalg.eigh(_hermitize(X))
    if np.any(w <= tol * max(1.0, w.max())):
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return (U * (1.0 / np.sqrt(w))) @ U.conj().T


#===============================================================================
class lehmann_matrix:
    """Lehmann representation of an L x L matrix analytic function with poles on the real axis.

    The function has the form

        G(z)[a,b] = sum_i Q[a,i] conj(Q[b,i]) / (z - W[i])

    i.e. G(z) = sum_i Q[:,i] Q[:,i]^dagger / (z - W[i]), where the poles W[i] lie
    on the real axis and the residue at pole i is the rank-one Hermitian
    positive-semidefinite matrix Q[:,i] Q[:,i]^dagger built from the Lehmann
    amplitude vector Q[:,i] (of length L). This is the matrix analogue of
    lehmann; for L = 1 it coincides with it (residue |Q[0,i]|**2).
    """

    def __init__(self, W, Q):
        """
        :param W: array-like of pole locations (real), length M.
        :param Q: array-like of Lehmann amplitudes, shape (L, M); column i is the
            amplitude vector of pole W[i]. May be complex.
        """
        self.W = np.asarray(W, dtype=float)
        self.Q = np.atleast_2d(np.asarray(Q, dtype=complex))

        if self.W.ndim != 1:
            raise ValueError(f"W must be one-dimensional, got {self.W.ndim} dimensions")
        if self.Q.ndim != 2:
            raise ValueError(f"Q must be two-dimensional (L, M), got {self.Q.ndim} dimensions")
        if self.Q.shape[1] != self.W.size:
            raise ValueError(
                f"Q must have one column per pole, got Q.shape={self.Q.shape} and {self.W.size} poles"
            )
        self.L = self.Q.shape[0]

    def evaluate(self, z):
        """Evaluate the matrix function at an arbitrary complex number (or array of).

        :param z: complex scalar or array of complex numbers.
        :return: G(z), of shape z.shape + (L, L).
        """
        z = np.asarray(z, dtype=complex)
        coeff = 1.0 / (z[..., np.newaxis] - self.W)          # (..., M)
        return np.einsum('...m,lm,km->...lk', coeff, self.Q, self.Q.conj())

    def moments(self, n):
        """Return the first n matrix moments of the associated spectral weight.

        The spectral weight is the discrete matrix measure
        A(x) = sum_i Q[:,i] Q[:,i]^dagger delta(x - W[i]), whose moments are the
        L x L matrices m_k = sum_i W[i]**k Q[:,i] Q[:,i]^dagger. This returns
        m_0, ..., m_{n-1} (n moments); m_0 = sum_i Q[:,i] Q[:,i]^dagger is the
        total weight.

        :param n: number of moments to compute.
        :return: complex array of shape (n, L, L) with the moments of orders 0 to n-1.
        """
        if n < 0:
            raise ValueError(f"number of moments must be non-negative, got {n}")
        powers = np.vander(self.W, n, increasing=True)       # (M, n)
        return np.einsum('ik,li,ji->klj', powers, self.Q, self.Q.conj())

    def modified_moments(self, a, b):
        """Return the modified matrix moments nu_k = integral pi_k dA of the spectral weight.

        The pi_k are the scalar reference polynomials defined by the three-term
        recurrence

            pi_{k+1}(x) = (x - a[k]) pi_k(x) - b[k] pi_{k-1}(x),   pi_0 = 1, pi_{-1} = 0

        so nu_k = sum_i pi_k(W[i]) Q[:,i] Q[:,i]^dagger. This returns len(a)+1
        matrix moments (orders 0 to len(a)). Using well-chosen reference
        polynomials (see continued_fraction.chebyshev_recurrence) makes the
        reconstruction by continued_fraction_matrix.from_modified_moments far
        better conditioned than the ordinary power moments.

        :param a: reference recurrence diagonal coefficients.
        :param b: reference recurrence off-diagonal coefficients (b[0] unused).
        :return: complex array of shape (len(a)+1, L, L).
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if b.size < a.size:
            raise ValueError("b must be at least as long as a")

        M = a.size + 1
        nu = np.zeros((M, self.L, self.L), dtype=complex)
        pkm1 = np.zeros_like(self.W)   # pi_{k-1}
        pk = np.ones_like(self.W)      # pi_k, starting at pi_0 = 1
        nu[0] = (self.Q * pk) @ self.Q.conj().T
        for k in range(1, M):
            pk, pkm1 = (self.W - a[k - 1]) * pk - b[k - 1] * pkm1, pk
            nu[k] = (self.Q * pk) @ self.Q.conj().T
        return nu

    def to_continued_fraction(self, tol=1e-12):
        """Convert to the block Jacobi continued-fraction format via the block Lanczos recursion.

        The matrix Lehmann representation is the block Stieltjes transform of the
        discrete matrix measure with weight Q[:,i] Q[:,i]^dagger at W[i]. Applying
        the block Lanczos algorithm to the diagonal matrix diag(W) with starting
        block Q^dagger generates the L x L recurrence coefficients of the
        associated matrix orthogonal polynomials, which are the block Jacobi
        continued-fraction coefficients:

            A[k]  (Hermitian diagonal block / block-Lanczos alpha)
            B[k]  (off-diagonal block / block-Lanczos beta), with B[0] from the
                  QR factor of the starting block (Q^dagger = V0 B[0]).

        Full reorthogonalization is used for numerical stability, and the
        recursion is truncated when the residual block norm drops below tol
        (Krylov space exhausted). Simple (non-deflating) block Lanczos is used, so
        badly rank-deficient residual blocks are treated as termination rather
        than deflated; this is adequate when the number of poles is a small
        multiple of L (the generic case).

        :param tol: threshold on the residual block norm below which the recursion stops.
        :return: an equivalent continued_fraction_matrix instance.
        """
        L = self.L
        M = self.W.size
        if M == 0:
            return continued_fraction_matrix(np.empty((0, L, L)), np.empty((0, L, L)))

        def orthonormalize(w):
            """w = V B with V (M, L) orthonormal columns; deflated directions
            (residual singular values below tol) get a zero column of V and a
            zero row of B, so they decouple exactly. Returns (V, B, n_active)."""
            U, s, Vh = np.linalg.svd(w, full_matrices=False)
            scale = s.max() if s.size else 1.0
            active = s > tol * max(1.0, scale)
            B = (s[:, np.newaxis] * Vh)
            B[~active, :] = 0.0
            U = U.copy()
            U[:, ~active] = 0.0
            return U, B, int(active.sum())

        # starting block: Q^dagger = V0 B0  (V0 has orthonormal columns)
        V0, B0, _ = orthonormalize(self.Q.conj().T)          # V0: (M, L), B0: (L, L)

        A = []
        B = [B0]
        Vs = [V0]
        v_prev = np.zeros((M, L), dtype=complex)
        b_cur = B0                                            # B[k] connecting level k-1 to k
        for k in range(M):
            v = Vs[-1]
            hv = self.W[:, np.newaxis] * v                    # diag(W) @ v
            alpha = _hermitize(v.conj().T @ hv)
            A.append(alpha)
            w = hv - v @ alpha
            if k > 0:
                w = w - v_prev @ b_cur.conj().T
            # full reorthogonalization against all previous Lanczos blocks (twice)
            for _ in range(2):
                for Vj in Vs:
                    w = w - Vj @ (Vj.conj().T @ w)
            v_next, b_next, n_active = orthonormalize(w)
            if n_active == 0:                                 # Krylov space exhausted
                break
            B.append(b_next)
            v_prev = v
            b_cur = b_next
            Vs.append(v_next)

        return continued_fraction_matrix(np.array(A), np.array(B))

    def add_to(self, X):
        """Return the lehmann_matrix representation of the sum self + X.

        Adding two matrix Lehmann functions pools their poles and amplitudes,
        since sum_i Q[:,i] Q[:,i]^d/(z-W[i]) + sum_j Q'[:,j] Q'[:,j]^d/(z-W'[j])
        is again of the same form. Both operands must have the same matrix size L.

        :param X: another lehmann_matrix instance.
        :return: a lehmann_matrix representing self + X.
        """
        if not isinstance(X, lehmann_matrix):
            raise TypeError("X must be a lehmann_matrix instance")
        if X.L != self.L:
            raise ValueError(f"matrix sizes differ: {self.L} and {X.L}")
        W = np.concatenate([self.W, X.W])
        Q = np.concatenate([self.Q, X.Q], axis=1)
        return lehmann_matrix(W, Q)

    def __repr__(self):
        return f"lehmann_matrix(L={self.L}, W={self.W!r}, Q={self.Q!r})"


#===============================================================================
class continued_fraction_matrix:
    """Block Jacobi continued-fraction representation of an L x L matrix function.

    The function has the form

        G(z) = B[0]^d M_0^{-1} B[0],   M_k = z I - A[k] - B[k+1]^d M_{k+1}^{-1} B[k+1],

    truncated (M_{n-1} = z I - A[n-1]) or closed with a terminator. A holds the
    Hermitian diagonal blocks A[k] and B holds the off-diagonal blocks B[k]
    (B[0] is the overall weight block, whose square B[0]^d B[0] = m_0 gives the
    zeroth matrix moment; for a matrix Green function A and B are the block
    Lanczos coefficients). Both arrays have shape (n, L, L). This is the matrix
    analogue of continued_fraction.
    """

    def __init__(self, A, B):
        """
        :param A: array-like of Hermitian diagonal blocks, shape (n, L, L).
        :param B: array-like of off-diagonal blocks, shape (n, L, L).
        """
        self.A = np.asarray(A, dtype=complex)
        self.B = np.asarray(B, dtype=complex)

        if self.A.shape != self.B.shape:
            raise ValueError(
                f"A and B must have the same shape, got {self.A.shape} and {self.B.shape}"
            )
        if self.A.ndim != 3 or self.A.shape[1] != self.A.shape[2]:
            raise ValueError(f"A and B must have shape (n, L, L), got {self.A.shape}")
        self.L = self.A.shape[1]

    def evaluate(self, z, terminator=None):
        """Evaluate the matrix function at an arbitrary complex number (or array of).

        By default the fraction is truncated (the tail below the last floor is the
        zero matrix), yielding a function with discrete poles. Passing a
        terminator seeds the last floor with the value of the (infinite) tail,
        restoring a continuous spectrum; see square_root_terminator.

        :param z: complex scalar or array of complex numbers.
        :param terminator: optional callable t(z) returning the L x L tail matrix
            below the last floor, of shape z.shape + (L, L) (default: 0).
        :return: G(z), of shape z.shape + (L, L).
        """
        z = np.asarray(z, dtype=complex)
        L = self.L
        eye = np.eye(L)
        f = np.zeros(z.shape + (L, L), dtype=complex)
        if terminator is not None:
            f = f + np.asarray(terminator(z), dtype=complex)
        zI = z[..., np.newaxis, np.newaxis] * eye
        # evaluate the block continued fraction from the bottom up
        for a, b in zip(self.A[::-1], self.B[::-1]):
            minv = np.linalg.inv(zI - a - f)
            f = b.conj().T @ minv @ b
        return f

    @staticmethod
    def square_root_terminator(a_inf, b_inf, maxiter=10000, tol=1e-13):
        """Return the matrix square-root terminator for asymptotic blocks a_inf, b_inf.

        When the coefficient blocks settle to constants (A[k] -> a_inf,
        B[k] -> b_inf), the infinite tail t of the block continued fraction is the
        (retarded) fixed point of the matrix recursion

            t(z) = b_inf^dagger (z I - a_inf - t)^{-1} b_inf.

        This is the surface self-energy of a semi-infinite block chain with
        on-site block a_inf and hopping b_inf, computed robustly (including inside
        the spectral bands) by Sancho-Rubio decimation with the retarded branch
        selected for Im z > 0. Used as the terminator argument of evaluate, it
        replaces the discrete poles of a truncated fraction by the continuous
        band(s) generated by (a_inf, b_inf). Matrix analogue of
        continued_fraction.square_root_terminator.

        :param a_inf: asymptotic Hermitian diagonal block (L x L).
        :param b_inf: asymptotic off-diagonal block (L x L).
        :param maxiter: maximum number of decimation iterations.
        :param tol: convergence threshold for the decimation.
        :return: a callable t(z).
        """
        a_inf = np.asarray(a_inf, dtype=complex)
        b_inf = np.asarray(b_inf, dtype=complex)

        def terminator(z):
            # chain hopping V = b_inf^dagger gives Sigma = b_inf^d (z-a-Sigma)^{-1} b_inf
            return _surface_selfenergy(z, a_inf, b_inf.conj().T, maxiter, tol)

        return terminator

    @staticmethod
    def period2_terminator(b1, b2, a1=None, a2=None, maxiter=10000, tol=1e-13):
        """Return the period-2 matrix terminator for coefficient blocks with a two-cycle asymptote.

        For a gapped (two-band) spectrum the coefficient blocks oscillate with
        period 2, (A, B) -> (a1, b1), (a2, b2), (a1, b1), .... The infinite tail
        is the (retarded) fixed point of the two-step recursion

            t  = b1^dagger (z I - a1 - t')^{-1} b1,
            t' = b2^dagger (z I - a2 - t )^{-1} b2.

        Here (a1, b1) are the coefficient blocks of the first omitted floor (just
        below the truncation) and (a2, b2) those of the next. It is computed by
        applying Sancho-Rubio decimation to the doubled (2L x 2L) super-cell that
        combines the two sublattices, which stays robust inside the bands. This
        restores the two bands of a gapped spectrum and generalizes
        square_root_terminator (the period-1 case a1=a2, b1=b2). Matrix analogue
        of continued_fraction.period2_terminator.

        :param b1: off-diagonal block of the first omitted floor.
        :param b2: off-diagonal block of the next floor.
        :param a1: diagonal block of the first omitted floor (default: zero).
        :param a2: diagonal block of the next floor (default: zero).
        :param maxiter: maximum number of decimation iterations.
        :param tol: convergence threshold for the decimation.
        :return: a callable t(z).
        """
        b1 = np.asarray(b1, dtype=complex)
        b2 = np.asarray(b2, dtype=complex)
        L = b1.shape[0]
        a1 = np.zeros((L, L)) if a1 is None else np.asarray(a1, dtype=complex)
        a2 = np.zeros((L, L)) if a2 is None else np.asarray(a2, dtype=complex)
        b1d = b1.conj().T

        # Going down the omitted floors the on-site blocks alternate a1, a2, a1,...
        # while the couplings alternate b2, b1, b2, ... (b1 enters floor 0 from the
        # truncation; b2 couples floor 0 to floor 1). Group two floors into a
        # 2L x 2L super-cell: intra-cell coupling b2, inter-cell coupling b1.
        E0 = np.block([[a1, b2.conj().T], [b2, a2]])
        V = np.zeros((2 * L, 2 * L), dtype=complex)
        V[L:, :L] = b1d                                       # site (a2) -> next site (a1) via b1

        def terminator(z):
            z = np.asarray(z, dtype=complex)
            sig = _surface_selfenergy(z, E0, V, maxiter, tol)
            zI = z[..., np.newaxis, np.newaxis] * np.eye(2 * L)
            g00 = np.linalg.inv(zI - E0 - sig)[..., :L, :L]   # GF at the a1 floor
            return b1d @ g00 @ b1

        return terminator

    @staticmethod
    def chebyshev_recurrence(N, xmin, xmax):
        """Recurrence coefficients of the (monic) Chebyshev polynomials on [xmin, xmax].

        Identical to continued_fraction.chebyshev_recurrence: the reference
        polynomials pi_k used by lehmann_matrix.modified_moments and
        from_modified_moments are scalar, so the same (a, b) apply in the matrix
        case. Provided here for convenience.

        :param N: number of coefficients.
        :param xmin, xmax: interval bracketing the spectral support.
        :return: (a, b) numpy arrays of length N.
        """
        return continued_fraction.chebyshev_recurrence(N, xmin, xmax)

    @staticmethod
    def from_moments(m):
        """Build a continued_fraction_matrix from the matrix moments of the spectral weight.

        Given the L x L moments m_k = sum_i W[i]**k Q[:,i] Q[:,i]^dagger for
        k = 0 ... 2n-1, the block Chebyshev algorithm produces the block Jacobi
        continued-fraction coefficients A[k] (Hermitian) and B[k], with
        B[0]^dagger B[0] = m_0. Chaining with to_lehmann() then recovers the poles
        and amplitudes (block Gaussian quadrature). From len(m) moments,
        n = len(m) // 2 coefficient blocks are found.

        Warning: as in the scalar case, ordinary power moments make this
        reconstruction severely ill-conditioned; it is only reliable for modest n.
        A non positive-definite block norm from inconsistent/noisy moments raises.
        This uniform-block algorithm also assumes the number of poles is a multiple
        of L (all block norms full rank); otherwise it raises and
        lehmann_matrix.to_continued_fraction should be used instead.

        :param m: array-like of at least 2 matrix moments, shape (2n, L, L).
        :return: a continued_fraction_matrix reproducing those moments.
        """
        m = np.asarray(m, dtype=complex)
        s = m.shape[0]
        a = np.zeros(max(s - 1, 1))
        b = np.zeros(max(s - 1, 1))
        return continued_fraction_matrix.from_modified_moments(m, a, b)

    @staticmethod
    def from_modified_moments(nu, a, b):
        """Build a continued_fraction_matrix from modified matrix moments (block modified Chebyshev).

        Given modified matrix moments nu_k = integral pi_k dA (k = 0 ... 2n-1)
        against the scalar reference polynomials pi_k with recurrence coefficients
        a, b (see chebyshev_recurrence), the block modified Chebyshev / Sack-
        Donovan-Wheeler algorithm returns the block Jacobi continued-fraction
        coefficients A[k] (Hermitian) and B[k]. It reduces to from_moments when
        a = b = 0 (monomials) but is far better conditioned with well-chosen
        references. Chain with to_lehmann() to recover the poles and amplitudes.
        Like from_moments, this uniform-block algorithm assumes the number of poles
        is a multiple of L (all block norms full rank) and otherwise raises; use
        lehmann_matrix.to_continued_fraction for the general (deflating) case.

        Internally the algorithm propagates the matrix cross-moments
        sigma_{k,l} = <P_k, pi_l> of the (monic, left-normalized) matrix
        orthogonal polynomials P_k, obtains the monic recurrence coefficients
        (alpha_k, beta_k) and the block norms gamma_k = <P_k, P_k>, then maps them
        to the orthonormal (Hermitian) block Jacobi form via
        A[k] = gamma_k^{-1/2} alpha_k gamma_k^{1/2},
        B[k] = gamma_k^{1/2} gamma_{k-1}^{-1/2}, and B[0] = gamma_0^{1/2}.

        :param nu: modified matrix moments, orders 0 ... 2n-1, shape (2n, L, L).
        :param a: reference recurrence diagonal coefficients (length >= 2n-1).
        :param b: reference recurrence off-diagonal coefficients (length >= 2n-1).
        :return: a continued_fraction_matrix reproducing those modified moments.
        """
        nu = np.asarray(nu, dtype=complex)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        S = nu.shape[0]
        L = nu.shape[1]
        n = S // 2
        if n < 1:
            raise ValueError("need at least 2 modified moments to recover one coefficient")
        if a.size < 2 * n - 1 or b.size < 2 * n - 1:
            raise ValueError(f"a and b must have length >= {2 * n - 1}")

        eye = np.eye(L)
        A = []
        B = []

        # The monic matrix orthogonal polynomials P_k obey (left convention)
        #     x P_k = P_{k+1} + alpha_k P_k + beta_k P_{k-1},
        # with alpha_k = H_k gamma_k^{-1},  beta_k = gamma_k gamma_{k-1}^{-1},
        # H_k = <x P_k, P_k> (Hermitian) and gamma_k = <P_k, P_k>. The orthonormal
        # (Hermitian) block Jacobi coefficients follow from the left normalization
        # p_k = gamma_k^{-1/2} P_k:  A[k] = gamma_k^{-1/2} H_k gamma_k^{-1/2},
        # B[k] = gamma_k^{1/2} gamma_{k-1}^{-1/2}.
        gamma0 = _hermitize(nu[0])                           # gamma_0 = <P_0, P_0> = m_0
        g0_scale = max(1.0, np.linalg.eigvalsh(gamma0).max())  # reference weight scale
        g0_sqrt = _sqrtm_psd(gamma0)
        g0_isqrt = _invsqrtm_pd(gamma0)
        H0 = nu[1] + a[0] * gamma0                           # <x P_0, P_0> = m_1
        alpha = H0 @ np.linalg.inv(gamma0)                   # monic alpha_0
        beta = gamma0                                        # monic beta_0 (unused: hits sigma_{-1})
        A.append(_hermitize(g0_isqrt @ H0 @ g0_isqrt))
        B.append(g0_sqrt)
        gp_isqrt = g0_isqrt

        sig_km2 = np.zeros_like(nu)                          # sigma_{k-2}
        sig_km1 = nu.copy()                                  # sigma_{k-1}, starting at sigma_0 = nu
        for k in range(1, n):
            sig_k = np.zeros_like(nu)
            for l in range(k, 2 * n - k):
                sig_k[l] = (sig_km1[l + 1]
                            - (alpha - a[l] * eye) @ sig_km1[l]
                            - beta @ sig_km2[l]
                            + b[l] * sig_km1[l - 1])
            gamma_k = _hermitize(sig_k[k])
            w = np.linalg.eigvalsh(gamma_k)
            # Classify the block norm gamma_k against the total-weight scale g0_scale
            # (not the block's own scale): a negative eigenvalue means inconsistent
            # moments; a whole vanishing block means the measure is exhausted; a
            # block with only *some* vanishing directions needs deflation, which
            # this uniform-block algorithm does not do (arises when the number of
            # poles is not a multiple of L).
            if w.min() < -1e-9 * g0_scale:
                raise ValueError(
                    f"negative block norm (min eig {w.min():.3e}) at coefficient {k}: the moments "
                    "are inconsistent or the reconstruction has exceeded its conditioning"
                )
            if w.max() < 1e-13 * g0_scale:
                break                                        # measure exhausted: truncate here
            if w.min() < 1e-11 * w.max():
                raise ValueError(
                    f"rank-deficient block norm at coefficient {k}: the number of poles is not a "
                    "multiple of L, which this uniform-block algorithm cannot deflate. Use "
                    "lehmann_matrix.to_continued_fraction for the general case."
                )
            inv_gkm1 = np.linalg.inv(sig_km1[k - 1])
            # H_k = <x P_k, P_k> = sigma_{k,k+1} + a_k gamma_k - gamma_k gamma_{k-1}^{-1} sigma_{k-1,k}
            H_k = sig_k[k + 1] + a[k] * gamma_k - gamma_k @ inv_gkm1 @ sig_km1[k]
            alpha = H_k @ np.linalg.inv(gamma_k)             # monic alpha_k
            beta = gamma_k @ inv_gkm1                        # monic beta_k
            gk_sqrt = _sqrtm_psd(gamma_k)
            gk_isqrt = _invsqrtm_pd(gamma_k)
            A.append(_hermitize(gk_isqrt @ H_k @ gk_isqrt))
            B.append(gk_sqrt @ gp_isqrt)
            gp_isqrt = gk_isqrt
            sig_km2, sig_km1 = sig_km1, sig_k

        return continued_fraction_matrix(np.array(A), np.array(B))

    def to_lehmann(self):
        """Convert to the matrix Lehmann format by diagonalizing the block Jacobi matrix.

        The coefficients define a Hermitian block-tridiagonal Jacobi matrix J with
        the diagonal blocks A[k] on the block diagonal and the off-diagonal blocks
        B[k] below (J[k, k-1] = B[k], J[k-1, k] = B[k]^dagger). Its
        eigen-decomposition J = sum_i W_i psi_i psi_i^dagger yields

            G(z) = B[0]^d [(z - J)^{-1}]_{00} B[0]
                 = sum_i (B[0]^d psi_i^{(0)}) (B[0]^d psi_i^{(0)})^d / (z - W_i)

        where psi_i^{(0)} is the first (length-L) block of eigenvector psi_i. So
        the poles are the eigenvalues and the Lehmann amplitudes are
        Q[:,i] = B[0]^dagger psi_i^{(0)}. Inverse of
        lehmann_matrix.to_continued_fraction (block Golub-Welsch).

        :return: an equivalent lehmann_matrix instance.
        """
        n = self.A.shape[0]
        L = self.L
        if n == 0:
            return lehmann_matrix(np.empty(0), np.empty((L, 0)))

        N = n * L
        J = np.zeros((N, N), dtype=complex)
        for k in range(n):
            J[k * L:(k + 1) * L, k * L:(k + 1) * L] = self.A[k]
            if k > 0:
                J[k * L:(k + 1) * L, (k - 1) * L:k * L] = self.B[k]
                J[(k - 1) * L:k * L, k * L:(k + 1) * L] = self.B[k].conj().T

        W, psi = np.linalg.eigh(_hermitize(J))
        psi0 = psi[:L, :]                                    # first block of each eigenvector, (L, N)
        Q = self.B[0].conj().T @ psi0                        # (L, N)
        return lehmann_matrix(W, Q)

    def add_to(self, X):
        """Return the continued_fraction_matrix representation of the sum self + X.

        As in the scalar case there is no simple direct rule, so this goes through
        the Lehmann representation: both operands are converted to matrix Lehmann
        form, their poles and amplitudes pooled, and the result converted back.

        :param X: another continued_fraction_matrix instance.
        :return: a continued_fraction_matrix representing self + X.
        """
        if not isinstance(X, continued_fraction_matrix):
            raise TypeError("X must be a continued_fraction_matrix instance")
        return self.to_lehmann().add_to(X.to_lehmann()).to_continued_fraction()

    def __repr__(self):
        return f"continued_fraction_matrix(L={self.L}, A={self.A!r}, B={self.B!r})"


#===============================================================================
class rational_fraction_matrix:
    """Rational representation G(z) = P(z) / Q(z) of an L x L matrix function.

    The denominator Q is a scalar polynomial of degree N (common to all matrix
    entries, its roots being the shared real poles) and the numerator P is a
    matrix polynomial of degree N-1 with L x L coefficient matrices, so G decays
    like 1/z at infinity (no polynomial part). Both are stored in ascending order
    (numpy.polynomial convention):

        P(z) = P[0] + P[1] z + ... + P[N-1] z**(N-1)     (each P[k] is L x L)
        Q(z) = Q[0] + Q[1] z + ... + Q[N]   z**N         (scalar)

    so P.shape == (N, L, L) and len(Q) == N + 1. Matrix analogue of
    rational_fraction.
    """

    def __init__(self, P, Q):
        """
        :param P: numerator coefficients (ascending order), shape (N, L, L).
        :param Q: scalar denominator coefficients (ascending order), length N+1.
        """
        self.P = np.asarray(P, dtype=complex)
        self.Q = np.asarray(Q, dtype=complex)

        if self.P.ndim != 3 or self.P.shape[1] != self.P.shape[2]:
            raise ValueError(f"P must have shape (N, L, L), got {self.P.shape}")
        if self.Q.ndim != 1:
            raise ValueError("Q must be a one-dimensional (scalar) coefficient array")
        if self.Q.size != self.P.shape[0] + 1:
            raise ValueError(
                f"expected deg(Q) = deg(P) + 1, got P.shape={self.P.shape}, len(Q)={self.Q.size}"
            )
        if self.Q[-1] == 0.0:
            raise ValueError("leading coefficient of Q must be non-zero")
        self.L = self.P.shape[1]

    def evaluate(self, z):
        """Evaluate the matrix function at an arbitrary complex number (or array of).

        :param z: complex scalar or array of complex numbers.
        :return: G(z), of shape z.shape + (L, L).
        """
        z = np.asarray(z, dtype=complex)
        num = np.zeros(z.shape + (self.L, self.L), dtype=complex)
        for k in range(self.P.shape[0] - 1, -1, -1):         # Horner on matrix coefficients
            num = num * z[..., np.newaxis, np.newaxis] + self.P[k]
        den = npoly.polyval(z, self.Q)
        return num / den[..., np.newaxis, np.newaxis]

    def to_lehmann(self, tol=1e-9):
        """Convert to the matrix Lehmann format by partial-fraction decomposition.

        Since deg(P) < deg(Q) and Q has simple real roots W[i], G has the
        partial-fraction expansion G(z) = sum_i R_i / (z - W[i]) with the L x L
        residue matrices R_i = P(W[i]) / Q'(W[i]). Each Hermitian
        positive-semidefinite R_i is diagonalized, R_i = sum_j lam_j u_j u_j^d,
        and every eigenpair with lam_j > tol contributes a Lehmann amplitude
        sqrt(lam_j) u_j at pole W[i] (so a rank-r residue yields r amplitude
        columns sharing that pole).

        :param tol: relative threshold below which residue eigenvalues are dropped.
        :return: an equivalent lehmann_matrix instance.
        """
        roots = npoly.polyroots(self.Q)
        dQ = npoly.polyder(self.Q)
        Ws = []
        cols = []
        for r in roots:
            Pr = np.zeros((self.L, self.L), dtype=complex)
            for k in range(self.P.shape[0] - 1, -1, -1):
                Pr = Pr * r + self.P[k]
            R = _hermitize(Pr / npoly.polyval(r, dQ))
            w, U = np.linalg.eigh(R)
            thr = tol * max(1.0, np.abs(w).max())
            for j in range(self.L):
                if w[j] > thr:
                    Ws.append(r.real)
                    cols.append(np.sqrt(w[j]) * U[:, j])
        if not Ws:
            return lehmann_matrix(np.empty(0), np.empty((self.L, 0)))
        order = np.argsort(Ws)
        W = np.array(Ws)[order]
        Q = np.array(cols).T[:, order]
        return lehmann_matrix(W, Q)

    @staticmethod
    def from_lehmann(lh):
        """Build a rational_fraction_matrix from a matrix Lehmann representation.

        The scalar denominator is the monic polynomial with the poles as roots,
        Q(z) = prod_i (z - W[i]), and the matrix numerator is
        P(z) = sum_i R_i prod_{j != i} (z - W[j]), with R_i = Q[:,i] Q[:,i]^dagger.
        As in the scalar rational_fraction.from_lehmann, the poles are assumed
        distinct.

        :param lh: a lehmann_matrix instance.
        :return: an equivalent rational_fraction_matrix.
        """
        W = lh.W
        L = lh.L
        M = W.size
        Qden = npoly.polyfromroots(W)                        # scalar, length M+1
        P = np.zeros((M, L, L), dtype=complex)
        for i in range(M):
            Ri = np.outer(lh.Q[:, i], lh.Q[:, i].conj())     # (L, L)
            coeffs = npoly.polyfromroots(np.delete(W, i))    # scalar, length M
            P += coeffs[:, np.newaxis, np.newaxis] * Ri
        return rational_fraction_matrix(P, Qden)

    def add_to(self, X):
        """Return the rational_fraction_matrix representation of the sum self + X.

        With a common scalar denominator the usual rule applies,
        P1/Q1 + P2/Q2 = (P1 Q2 + P2 Q1) / (Q1 Q2): the matrix numerators are
        convolved with the other scalar denominator and added, and the scalar
        denominators are multiplied. Both operands must have the same matrix size.

        :param X: another rational_fraction_matrix instance.
        :return: a rational_fraction_matrix representing self + X.
        """
        if not isinstance(X, rational_fraction_matrix):
            raise TypeError("X must be a rational_fraction_matrix instance")
        if X.L != self.L:
            raise ValueError(f"matrix sizes differ: {self.L} and {X.L}")
        P1Q2 = _polymul_matrix_scalar(self.P, X.Q)
        P2Q1 = _polymul_matrix_scalar(X.P, self.Q)
        P = _polyadd_matrix(P1Q2, P2Q1)
        Q = npoly.polymul(self.Q, X.Q)
        return rational_fraction_matrix(P, Q)

    def __repr__(self):
        return f"rational_fraction_matrix(L={self.L}, P={self.P!r}, Q={self.Q!r})"


def _polymul_matrix_scalar(P, q):
    """Convolve a matrix polynomial P, shape (N, L, L), with a scalar polynomial q."""
    P = np.asarray(P)
    q = np.asarray(q)
    N, L, _ = P.shape
    out = np.zeros((N + q.size - 1, L, L), dtype=complex)
    for k in range(q.size):
        out[k:k + N] += q[k] * P
    return out


def _polyadd_matrix(P1, P2):
    """Add two matrix polynomials (shapes (N1, L, L) and (N2, L, L)), padding to the longer."""
    n = max(P1.shape[0], P2.shape[0])
    L = P1.shape[1]
    out = np.zeros((n, L, L), dtype=complex)
    out[:P1.shape[0]] += P1
    out[:P2.shape[0]] += P2
    return out


#===============================================================================
if __name__ == "__main__":
    # Self-test: exercise every conversion for the scalar classes and their
    # matrix (block) generalizations and check that all representations agree.
    # Run with:  python green_structure.py
    import sys

    rng = np.random.default_rng(1)
    z = np.array([0.3 + 0.5j, -1.2 + 0.8j, 2.1 - 0.4j, 0.0 + 1.3j, 5.0 + 0.2j])
    TOL = 1e-9
    results = []

    def check(name, value):
        ok = bool(value < TOL)
        results.append(ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:48s} {value:.2e}")

    def amax(a, b):
        return np.max(np.abs(np.asarray(a) - np.asarray(b)))

    # ---- scalar classes --------------------------------------------------
    print("scalar classes (random 8-pole measure):")
    W = np.sort(rng.uniform(-2, 2, 8))
    R = rng.uniform(0.1, 1.0, 8)
    lh = lehmann(W, R)
    ref = lh.evaluate(z)
    cf = lh.to_continued_fraction()
    rf = rational_fraction.from_lehmann(lh)
    check("lehmann -> continued_fraction -> f", amax(cf.evaluate(z), ref))
    check("continued_fraction -> lehmann -> f", amax(cf.to_lehmann().evaluate(z), ref))
    check("lehmann -> rational_fraction -> f", amax(rf.evaluate(z), ref))
    check("rational_fraction -> lehmann -> f", amax(rf.to_lehmann().evaluate(z), ref))
    m = lh.moments(16)
    check("continued_fraction.from_moments", amax(continued_fraction.from_moments(m).evaluate(z), ref))
    ac, bc = continued_fraction.chebyshev_recurrence(16, -2.2, 2.2)
    nu = lh.modified_moments(ac, bc)
    check("continued_fraction.from_modified_moments",
          amax(continued_fraction.from_modified_moments(nu, ac, bc).evaluate(z), ref))
    check("lehmann.add_to (self + self)", amax(lh.add_to(lh).evaluate(z), 2 * ref))

    # ---- matrix classes --------------------------------------------------
    print("matrix (block) classes (random L x M measures):")
    last_cfm = None
    for L, M in [(1, 8), (2, 8), (2, 7), (3, 9), (3, 10)]:
        Wm = np.sort(rng.uniform(-2, 2, M))
        Qm = rng.standard_normal((L, M)) + 1j * rng.standard_normal((L, M))
        lhm = lehmann_matrix(Wm, Qm)
        refm = lhm.evaluate(z)
        cfm = lhm.to_continued_fraction()
        rfm = rational_fraction_matrix.from_lehmann(lhm)
        last_cfm = cfm
        check(f"L={L} M={M}: lehmann_matrix -> cf -> f", amax(cfm.evaluate(z), refm))
        check(f"L={L} M={M}: cf -> lehmann_matrix -> f", amax(cfm.to_lehmann().evaluate(z), refm))
        check(f"L={L} M={M}: rational_matrix round trip", amax(rfm.to_lehmann().evaluate(z), refm))
        check(f"L={L} M={M}: A blocks Hermitian",
              max((amax(a, a.conj().T) for a in cfm.A), default=0.0))
        if M % L == 0:                                    # from_moments needs uniform blocks
            mm = lhm.moments(2 * (M // L))
            check(f"L={L} M={M}: cf_matrix.from_moments",
                  amax(continued_fraction_matrix.from_moments(mm).evaluate(z), refm))

    # ---- L=1 matrix vs scalar consistency --------------------------------
    print("L=1 matrix vs scalar consistency (same W, R):")
    lh1 = lehmann_matrix(W, np.sqrt(R)[None, :])
    cf1 = lh1.to_continued_fraction()
    rf1 = rational_fraction_matrix.from_lehmann(lh1)
    check("evaluate", amax(lh1.evaluate(z)[..., 0, 0], ref))
    check("continued_fraction evaluate", amax(cf1.evaluate(z)[..., 0, 0], cf.evaluate(z)))
    check("continued_fraction A / B coefficients",
          max(amax(cf1.A[:, 0, 0], cf.A), amax(cf1.B[:, 0, 0], cf.B)))
    check("rational_fraction evaluate", amax(rf1.evaluate(z)[..., 0, 0], rf.evaluate(z)))
    check("from_moments", amax(
        continued_fraction_matrix.from_moments(lh1.moments(16)).evaluate(z)[..., 0, 0],
        continued_fraction.from_moments(m).evaluate(z)))

    # ---- terminators (fixed-point residuals) -----------------------------
    print("terminators (fixed-point residual):")
    a_inf, b_inf = cf.A[-1], cf.B[-1]
    ts = continued_fraction.square_root_terminator(a_inf, b_inf)(z)
    check("scalar square_root_terminator", amax(ts, b_inf**2 / (z - a_inf - ts)))
    A_inf, B_inf = last_cfm.A[-1], last_cfm.B[-1]
    tm = continued_fraction_matrix.square_root_terminator(A_inf, B_inf)(z)
    zI = z[..., None, None] * np.eye(A_inf.shape[0])
    check("matrix square_root_terminator",
          amax(tm, B_inf.conj().T @ np.linalg.inv(zI - A_inf - tm) @ B_inf))

    n_fail = results.count(False)
    print("\n" + ("ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"))
    sys.exit(n_fail)
