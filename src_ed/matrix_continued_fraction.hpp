#ifndef matrix_continued_fraction_h
#define matrix_continued_fraction_h

/**
 * @file matrix_continued_fraction.hpp
 * @brief Definition of the matrix_continued_fraction template (block Lanczos MCF).
 */

/*
 matrix_continued_fraction.hpp

 Matrix-valued continued fraction arising from the block Lanczos method.

 Scalar analogue: continued_fraction (see continued_fraction.hpp).

 A block Lanczos run with block size p and M steps produces:
   A[0..M-1]  -- p×p Hermitian diagonal blocks
   B[0..M-1]  -- p×p upper-triangular off-diagonal QR blocks

 These define a block-tridiagonal projected Hamiltonian T whose (0,0)
 block resolvent F_0(z) = [(zI - T)^{-1}]_{00} is evaluated via the
 backward recursion (matrix continued fraction):

   F_M = 0_{p×p}
   F_j = (zI - A_j - B_j^H F_{j+1} B_j)^{-1},   j = M-1 … 0

 where B_j (j < M-1) are the off-diagonal blocks stored in B; B[M-1] is
 the truncation residual and is not used in the recursion, in exact
 analogy with the last element of the b array in continued_fraction.

 If the starting block phi was not pre-orthonormalised before calling
 blockLanczos, an optional weight matrix W can be supplied so that the
 full projected Green function is  G(z) = W^H F_0(z) W.

 Usage:
   vector<matrix<HilbertField>> A, B;
   int M0 = 50;
   blockLanczos(H, phi, A, B, M0);
   matrix_continued_fraction<HilbertField> mcf(A, B);   // orthonormal starting block
   matrix<Complex> G = mcf.evaluate(z);
*/

#include "continued_fraction.hpp"   // pulls in block_matrix.hpp → matrix.hpp
// hdf5_io.hpp is already pulled in by continued_fraction.hpp
#include "Q_matrix.hpp"

//! Right polar decomposition of a p×p matrix: C = W*P, P Hermitian PSD, W
//! unitary (a partial isometry if C is singular). Used by
//! matrix_continued_fraction::convert_B_format() to move an off-diagonal
//! Lanczos block from upper-triangular (QR) to Hermitian form.
/**
 P = sqrt(C^H C) is obtained from the eigendecomposition of the Hermitian PSD
 matrix C^H C; W = C*P^{-1}. Eigenvalues of C^H C below (accur_deflation)^2
 are treated as zero (their contribution to P^{-1} is dropped) — this only
 matters for a rank-deficient C, which in practice only arises for the unused
 truncation-residual block B[M-1].
*/
template<typename T>
void mcf_polar_decompose(const matrix<T> &C, matrix<T> &W, matrix<T> &P)
{
    int p = (int)C.r;
    matrix<T> Ch(C);
    Ch.hermitian_conjugate();
    matrix<T> M(p);
    M.product(Ch, C);                   // M = C^H * C  (Hermitian PSD)

    vector<double> D(p);
    matrix<T> V(p);
    M.eigensystem(D, V);                // M = V * diag(D) * V^H, D >= 0

    double tol = global_double("accur_deflation");
    tol *= tol;

    P.set_size(p);
    matrix<T> Pinv(p);
    for(int r = 0; r < p; ++r){
        for(int c = 0; c < p; ++c){
            T s(0), si(0);
            for(int m = 0; m < p; ++m){
                double sq = sqrt(max(D[m], 0.0));
                T vc = V(r,m) * conjugate(V(c,m));
                s  += vc * T(sq);
                si += vc * T(D[m] > tol ? 1.0/sq : 0.0);
            }
            P(r,c) = s;
            Pinv(r,c) = si;
        }
    }
    W.set_size(p);
    W.product(C, Pinv);                 // W = C * P^{-1}
}

//! QR decomposition of a p×p matrix via modified Gram-Schmidt: C = Q*R, Q
//! unitary, R upper-triangular. Used by
//! matrix_continued_fraction::convert_B_format() to move an off-diagonal
//! Lanczos block from Hermitian to upper-triangular form.
template<typename T>
void mcf_qr_decompose(const matrix<T> &C, matrix<T> &Q, matrix<T> &R)
{
    int p = (int)C.r;
    Q = C;
    R.set_size(p);
    for(int l = 0; l < p; ++l){
        for(int k = 0; k < l; ++k){
            T z(0);
            for(int i = 0; i < p; ++i) z += conjugate(Q(i,k)) * Q(i,l);
            R(k,l) = z;
            for(int i = 0; i < p; ++i) Q(i,l) -= z * Q(i,k);
        }
        double nrm2 = 0.0;
        for(int i = 0; i < p; ++i) nrm2 += realpart(conjugate(Q(i,l)) * Q(i,l));
        double nrm = sqrt(nrm2);
        R(l,l) = T(nrm);
        for(int i = 0; i < p; ++i) Q(i,l) *= T(1.0/nrm);
    }
}


//! Matrix-valued Jacobi continued fraction.
/**
 Template parameter T is the field of the Lanczos matrices A and B
 (double for a real Hamiltonian, Complex for a complex one).
 The weight matrix W and the Green function G(z) are always complex.
*/
template<typename T>
struct matrix_continued_fraction
{
    int p;                        //!< block size
    vector<matrix<T>> A;          //!< diagonal blocks (partial denominators)
    vector<matrix<T>> B;          //!< off-diagonal blocks (partial numerator factors)
    matrix<Complex>   W;          //!< initial weight (default: identity, always complex)
    bool hermitian_B = false;     //!< true: B[j] Hermitian PSD (polar); false: B[j] upper-triangular (QR)

    //! Default constructor
    matrix_continued_fraction() : p(0) {}

    //! Constructor from block Lanczos output with orthonormal starting block (W = I).
    /**
     @param _A  Diagonal blocks from blockLanczos.
     @param _B  Off-diagonal QR blocks from blockLanczos.
    */
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B)
    {
        p = (int)_A[0].r;
        A = _A;
        B = _B;
        W.set_size(p);
        W.identity();
    }

    //! Constructor analogous to continued_fraction(a, b, e0, norm, create).
    /**
     Applies an energy shift to the diagonal blocks (for electron/hole
     components of the Green function, matching the scalar continued_fraction
     constructor convention) and stores an initial weight matrix W.

     @param _A     Diagonal blocks from blockLanczos.
     @param _B     Off-diagonal QR blocks from blockLanczos.
     @param e0     Ground-state energy used to shift A[j].
     @param _W     p×p weight matrix arising from initial-block normalisation.
                   Pass the identity for an orthonormal starting block.
     @param create If true (creation/electron sector):  A[j] -= e0*I.
                   If false (annihilation/hole sector):  A[j]  = e0*I - A[j].
    */
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B,
                               double e0,
                               const matrix<T> &_W,
                               bool create)
    {
        p = (int)_A[0].r;
        A = _A;
        B = _B;
        for(size_t j = 0; j < A.size(); ++j){
            if(create) A[j] -= e0;            // A[j] -= e0 * I
            else {
                for(size_t r = 0; r < A[j].v.size(); ++r) A[j].v[r] = -A[j].v[r];
                A[j] += e0;                   // A[j] = e0*I - A[j]
            }
        }
        W = to_complex_matrix(_W);
    }

    //! Constructor from pre-processed matrices and explicit complex weight.
    /**
     Used when the MCF coefficients have been constructed externally (e.g., after
     periodization in the lattice model), so no additional energy shift is needed.

     @param _A  Diagonal blocks, already energy-shifted.
     @param _B  Off-diagonal QR blocks.
     @param _W  Weight matrix (p×p, complex); G(z) = W^H F_0(z) W.
    */
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B,
                               const matrix<Complex> &_W)
    {
        if(_A.empty()){ p = 0; return; }
        p = (int)_A[0].r;
        A = _A;
        B = _B;
        W = _W;
    }

    //! Evaluate the matrix continued fraction at complex frequency z.
    /**
     Returns G(z) = W^H F_0(z) W where
       F_M = 0,
       F_j = (zI - A_j - B_j^H F_{j+1} B_j)^{-1},  j = M-1 … 0.
     B[M-1] (the last element) is the truncation residual and is not used.
    */
    matrix<Complex> evaluate(Complex z) const
    {
        int M = (int)A.size();

        // F starts as the M-th floor: F_M = 0
        matrix<Complex> F(p, p);   // zero-initialised

        for(int j = M - 1; j >= 0; --j){

            // Σ = B[j]^H · F · B[j]  (only when j < M-1; otherwise Σ = 0)
            matrix<Complex> sigma(p, p);  // zero-initialised
            if(j < M - 1){
                matrix<Complex> Bj = to_complex_matrix(B[j]);
                // step 1: tmp = F · B[j]
                matrix<Complex> tmp(p);
                tmp.product(F, Bj);
                // step 2: sigma = B[j]^H · tmp
                matrix<Complex> Bjh(Bj);
                Bjh.hermitian_conjugate();
                sigma.product(Bjh, tmp);
            }

            // Build denominator D = z·I - A[j] - Σ
            matrix<Complex> D(p, p);    // zero
            D += z;                     // D = z·I
            D -= to_complex_matrix(A[j]);  // D = z·I - A[j]
            D -= sigma;                 // D = z·I - A[j] - Σ

            D.inverse();                // D  →  D^{-1}
            F = D;                      // F_{j} = D^{-1}
        }

        // Apply weight: G = W^H · F_0 · W
        // W may be non-square (p rows × q columns): result is q×q.
        int q = (int)W.c;
        matrix<Complex> tmp(p, q);
        tmp.product(F, W);              // tmp = F · W  (p × q)
        matrix<Complex> Wh(W);
        Wh.hermitian_conjugate();       // Wh  = W^H   (q × p)
        matrix<Complex> G(q, q);
        G.product(Wh, tmp);             // G   = W^H · F · W  (q × q)

        return G;
    }

    //! Frequency-integrated Green function: ∫_{-∞}^{0} A(ω) dω.
    /**
     Builds the dense (Mp × Mp) Hermitian block-tridiagonal matrix T (diagonal
     blocks A[j]; T(j,j+1) = B[j]^H, T(j+1,j) = B[j] for j = 0..M-2; B[M-1] is
     the truncation residual and is excluded, exactly as in evaluate()/apply()),
     diagonalizes it, and keeps only the eigenmodes with negative eigenvalue.

     For eigenvalue d_k < 0 with eigenvector u_k, writing u_k^{(0)} for its
     first p components (the level-0 block), the pole's residue in the output
     (W-weighted) basis is v_k = W^H u_k^{(0)}, and the returned q×q matrix is
     sum_k v_k v_k^H over negative-eigenvalue modes only — i.e. the same
     per-pole energy filter as Q_matrix::integrated_Green_function, applied
     here to the block-Lanczos representation instead of an explicit Lehmann
     sum. This is exact regardless of whether every pole of this MCF happens
     to lie on one side of zero (e.g. for a hole MCF built from an excited,
     thermally-weighted reference state, where some poles can stray to
     positive energy).
    */
    matrix<Complex> integrated_Green_function() const
    {
        int M = (int)A.size();
        int q = (int)W.c;
        matrix<Complex> result(q, q);
        if(M == 0 || p == 0) return result;

        int N = M*p;
        matrix<Complex> Tmat(N, N);
        for(int j = 0; j < M; ++j){
            matrix<Complex> Aj = to_complex_matrix(A[j]);
            for(int a = 0; a < p; ++a)
                for(int b = 0; b < p; ++b)
                    Tmat(j*p+a, j*p+b) = Aj(a,b);
            if(j < M-1){
                matrix<Complex> Bj = to_complex_matrix(B[j]);
                for(int a = 0; a < p; ++a){
                    for(int b = 0; b < p; ++b){
                        Tmat(j*p+a, (j+1)*p+b) = conjugate(Bj(b,a));   // B[j]^H
                        Tmat((j+1)*p+b, j*p+a) = Bj(b,a);              // B[j]
                    }
                }
            }
        }

        vector<double> d(N);
        matrix<Complex> U(N, N);
        Tmat.eigensystem(d, U);

        matrix<Complex> Wh(W);
        Wh.hermitian_conjugate();

        for(int k = 0; k < N; ++k){
            if(d[k] >= 0.0) continue;
            matrix<Complex> u0(p, 1);
            for(int a = 0; a < p; ++a) u0(a,0) = U(a, k);
            matrix<Complex> v(q, 1);
            v.product(Wh, u0);
            for(int a = 0; a < q; ++a)
                for(int b = 0; b < q; ++b)
                    result(a,b) += v(a,0)*conjugate(v(b,0));
        }
        return result;
    }

    //! Apply the block-tridiagonal Lanczos matrix T to a vector and return T*x.
    /**
     The input vector x must have size M*p (M = A.size(), p = block size).
     T is block-tridiagonal with diagonal blocks A[j], sub-diagonal blocks B[j-1]
     (at block row j, column j-1), and super-diagonal blocks B[j]^H (at block row j,
     column j+1).  B[M-1] is the truncation residual and is NOT applied.

       (T x)_j = B[j-1] x_{j-1} + A[j] x_j + B[j]^H x_{j+1}

     with the boundary terms B[-1] = B[M-1] = 0.
    */
    vector<T> apply(const vector<T>& x) const {
        int M = (int)A.size();
        vector<T> y(M * p, T(0));
        for(int j = 0; j < M; ++j){
            // diagonal: A[j] * x_j
            for(int col = 0; col < p; ++col)
                for(int row = 0; row < p; ++row)
                    y[j*p + row] += A[j](row, col) * x[j*p + col];
            // sub-diagonal: B[j-1] * x_{j-1}
            if(j > 0)
                for(int col = 0; col < p; ++col)
                    for(int row = 0; row < p; ++row)
                        y[j*p + row] += B[j-1](row, col) * x[(j-1)*p + col];
            // super-diagonal: B[j]^H * x_{j+1}  (B[M-1] not used)
            if(j < M-1)
                for(int col = 0; col < p; ++col)
                    for(int row = 0; row < p; ++row)
                        y[j*p + row] += conjugate(B[j](col, row)) * x[(j+1)*p + col];
        }
        return y;
    }

    //! Returns a real-valued copy by taking the real part of every matrix.
    /**
     Converts this MCF to a matrix_continued_fraction<double> by extracting
     the real part of each A[j], B[j], and W.  Useful when the Hamiltonian is
     known to be real and the imaginary parts are negligible numerical noise.
    */
    matrix_continued_fraction<double> real() const {
        matrix_continued_fraction<double> result;
        result.p = p;
        result.A.resize(A.size());
        result.B.resize(B.size());
        for(size_t j = 0; j < A.size(); ++j) result.A[j] = to_real_matrix(A[j]);
        for(size_t j = 0; j < B.size(); ++j) result.B[j] = to_real_matrix(B[j]);
        result.W = to_complex_matrix(to_real_matrix(W));
        result.hermitian_B = hermitian_B;
        return result;
    }

    //! Number of levels (= A.size()).
    int floors() const { return (int)A.size(); }

    //! Convert the off-diagonal blocks B[j] between upper-triangular (QR) and
    //! Hermitian PSD (polar) form, in place.
    /**
     The two representations correspond to different (but physically equivalent)
     choices of orthonormal basis for each Krylov level j >= 1: Q_j' = Q_j * G_j,
     with G_0 = I so the level-0 basis — and hence W and evaluate()'s G(z) — is
     left exactly unchanged.

     Sweeping j = 0 .. M-1 with a running gauge G (G = I initially): A[j] is
     conjugated by the current G, then C = B[j]*G is re-factored — via a polar
     decomposition (C = W_j*P_j, P_j Hermitian PSD) to move towards Hermitian
     form, or via a QR decomposition (C = Q_j*R_j, R_j upper-triangular) to move
     towards upper-triangular form. B[j] becomes the Hermitian/triangular factor
     and G is updated to the unitary factor for use at the next level.

     B[M-1] (the truncation residual, unused by evaluate()/apply()) is converted
     for consistency, but the trailing gauge it would define is discarded.

     Toggles hermitian_B to reflect the new state.
    */
    void convert_B_format()
    {
        int M = (int)A.size();
        if(M == 0){ hermitian_B = !hermitian_B; return; }

        matrix<T> G(p);
        G.identity();
        for(int j = 0; j < M; ++j){
            matrix<T> Gh(G);
            Gh.hermitian_conjugate();
            matrix<T> tmp(p);
            tmp.product(A[j], G);
            A[j].product(Gh, tmp);          // A[j] <- G^H * A[j] * G

            matrix<T> C(p);
            C.product(B[j], G);             // C = B[j] * G

            if(hermitian_B){
                matrix<T> Q, R;
                mcf_qr_decompose(C, Q, R);  // C = Q*R, R upper-triangular
                B[j] = R;
                G = Q;
            } else {
                matrix<T> Wp, P;
                mcf_polar_decompose(C, Wp, P);  // C = Wp*P, P Hermitian PSD
                B[j] = P;
                G = Wp;
            }
        }
        hermitian_B = !hermitian_B;
    }
};


// C++14-compatible helpers: convert a Complex scalar to field T.
inline double      mcf_to_T(Complex z, double)  { return z.real(); }
inline Complex     mcf_to_T(Complex z, Complex) { return z; }


//! Diagonal-matrix operator for block Lanczos.
/**
 Implements the Hamiltonian H = diag(e[0], ..., e[M-1]) used in Q_matrix_to_mcf.
 Satisfies the TYPE concept expected by blockLanczos via mult_add().
*/
template<typename T>
struct diagonal_hamiltonian {
    const vector<double>& evals;
    explicit diagonal_hamiltonian(const vector<double>& _e) : evals(_e) {}
    void mult_add(const vector<T>& x, vector<T>& y) const {
        for(size_t i = 0; i < evals.size(); ++i)
            y[i] += T(evals[i]) * x[i];
    }
};


//! Convert a Q_matrix into a matrix_continued_fraction via block Lanczos.
/**
 The Hamiltonian used is the diagonal matrix H = diag(Q.e[0], ..., Q.e[M-1]).
 The L starting vectors are the L rows of Q.v.  The resulting MCF is equivalent
 to the Lehmann sum G(z)(a,b) = sum_k Q.v(a,k) * conj(Q.v(b,k)) / (z - Q.e[k]).

 The weight matrix W is extracted from Q.v via modified Gram-Schmidt (as in
 combine_via_lanczos), so that G(z) = W^H F_0(z) W.

 M0 is set to Q.M (the Krylov space cannot exceed the eigenvalue count M),
 capped by the global parameter max_iter_BL.

 @param Q  Combined Q_matrix (electron + hole poles, already energy-shifted).
 @return   matrix_continued_fraction equivalent to Q.
*/
template<typename HilbertField>
matrix_continued_fraction<HilbertField> Q_matrix_to_mcf(const Q_matrix<HilbertField>& Q)
{
    const int L = (int)Q.L;
    const int M = (int)Q.M;
    if(M == 0 || L == 0) return matrix_continued_fraction<HilbertField>();

    // Build starting block: phi[a] = row a of Q.v  (length-M vectors)
    vector<vector<HilbertField>> phi(L, vector<HilbertField>(M));
    for(int a = 0; a < L; ++a)
        for(int k = 0; k < M; ++k)
            phi[a][k] = Q.v(a, k);

    // Extract the upper-triangular QR factor W from phi via modified Gram-Schmidt.
    // The working copy q is discarded; blockLanczos re-orthonormalises phi internally.
    matrix<HilbertField> W(L);
    {
        vector<vector<HilbertField>> q(phi);
        for(int l = 0; l < L; ++l){
            for(int k = 0; k < l; ++k){
                HilbertField z = q[k] * q[l];
                W(k, l) = z;
                mult_add(-z, q[k], q[l]);
            }
            double nrm = norm(q[l]);
            W(l, l) = HilbertField(nrm);
            q[l] *= 1.0 / nrm;
        }
    }

    diagonal_hamiltonian<HilbertField> H(Q.e);
    vector<matrix<HilbertField>> A, B;
    int M0 = M;
    bool use_QR = global_bool("block_Lanczos_QR");
    if(use_QR)
        blockLanczos(H, phi, A, B, M0);
    else
        blockLanczosSVD(H, phi, A, B, M0);

    matrix_continued_fraction<HilbertField> mcf(A, B, to_complex_matrix(W));
    mcf.hermitian_B = !use_QR;
    return mcf;
}


//! Combine two MCFs into a single one whose evaluate() gives G_h(z) + G_e(z)^T directly.
/**
 Assembles the direct-sum block-tridiagonal T = T_e ⊕ T_h (same blocks as
 direct_sum()) but uses a non-square weight

   W_combined = [conj(W_e)]   (first  p rows: conjugated electron weight)
                [W_h       ]   (second p rows: hole weight)

 of size 2p × p.  Because W_combined has p columns, evaluate() returns a p×p
 Green function rather than 2p×2p:

   G(z) = W_combined^H F_combined(z) W_combined
        = W_e^T F_e(z) W_e* + W_h^H F_h(z) W_h

 For T = double (real Hamiltonian): F_e is complex-symmetric and W_e is real, so
 W_e^T F_e W_e* = G_e(z) = G_e(z)^T.  The result is exact.

 For T = Complex: exact when time-reversal symmetry ensures G_e = G_e^T (the
 common case).  For Hamiltonians that break time-reversal the approximation
 replaces G_e^T with W_e^T F_e W_e*; in that case use two separate evaluate()
 calls instead.
*/
template<typename T>
matrix_continued_fraction<T> combine_for_gf(
    const matrix_continued_fraction<T>& e,
    const matrix_continued_fraction<T>& h)
{
    const int pe = e.p, ph = h.p;
    QCM_ASSERT(pe == ph);
    QCM_ASSERT(e.hermitian_B == h.hermitian_B);
    const int p = pe + ph;
    const int Me = e.floors(), Mh = h.floors(), M = max(Me, Mh);

    // Direct-sum A and B blocks (identical to direct_sum())
    vector<matrix<T>> A(M), B(M);
    for(int j = 0; j < M; ++j){
        A[j] = matrix<T>(p, p);
        B[j] = matrix<T>(p, p);
        if(j < Me){
            e.A[j].move_sub_matrix(pe, pe, 0, 0, 0,  0,  A[j]);
            e.B[j].move_sub_matrix(pe, pe, 0, 0, 0,  0,  B[j]);
        }
        if(j < Mh){
            h.A[j].move_sub_matrix(ph, ph, 0, 0, pe, pe, A[j]);
            h.B[j].move_sub_matrix(ph, ph, 0, 0, pe, pe, B[j]);
        }
    }

    // Non-square weight W (2p × p): upper block = conj(W_e), lower block = W_h.
    matrix<Complex> W(p, pe);   // zero-initialised
    for(int k = 0; k < pe; ++k)
        for(int i = 0; i < pe; ++i)
            W(k, i) = conj(e.W(k, i));   // conj(W_e) in electron rows
    for(int k = 0; k < ph; ++k)
        for(int i = 0; i < ph; ++i)
            W(pe + k, i) = h.W(k, i);    // W_h in hole rows

    matrix_continued_fraction<T> mcf(A, B, W);
    mcf.hermitian_B = e.hermitian_B;
    return mcf;
}


//! Operator wrapper for the direct-sum T_e ⊕ T_h* acting on the concatenated space.
/**
 The combined space has dimension (Me + Mh)*p with the e-sector occupying the
 first Me*p components and the h-sector occupying the last Mh*p components.
 Both sectors share the same block size p.

 The e-sector applies T_e normally.  The h-sector applies T_h* (element-wise
 complex conjugate of T_h), i.e. T_h* x = conj(T_h conj(x)).  This is needed
 so that the combined MCF (with conj(W_h) starting vectors for the h-sector)
 reproduces G_mcf_h^T rather than G_mcf_h.  For T=double, T_h* = T_h and
 conj is a no-op, so the real case is unaffected.

 Satisfies the TYPE concept expected by blockLanczos:
   void mult_add(const vector<T>& x, vector<T>& y)
*/
template<typename T>
struct combined_sector_operator {
    const matrix_continued_fraction<T>& e;
    const matrix_continued_fraction<T>& h;
    const int Me, Mh, p;

    combined_sector_operator(const matrix_continued_fraction<T>& _e,
                              const matrix_continued_fraction<T>& _h)
        : e(_e), h(_h), Me(_e.floors()), Mh(_h.floors()), p(_e.p)
    { QCM_ASSERT(_e.p == _h.p); }

    void mult_add(const vector<T>& x, vector<T>& y) const {
        // e-sector: y_e += T_e * x_e
        vector<T> xe(x.begin(), x.begin() + Me * p);
        vector<T> ye = e.apply(xe);
        for(int i = 0; i < Me * p; ++i) y[i] += ye[i];

        // h-sector: y_h += T_h* x_h = conj(T_h conj(x_h))
        vector<T> xh(x.begin() + Me * p, x.end());
        for(auto& v : xh) v = conjugate(v);          // conjugate x_h
        vector<T> yh = h.apply(xh);                  // T_h * conj(x_h)
        for(int i = 0; i < Mh * p; ++i) y[Me * p + i] += conjugate(yh[i]);
    }
};


//! Combine two MCFs via a new block Lanczos run on their direct-sum operator.
/**
 Applies blockLanczos to the direct-sum operator T = T_e ⊕ T_h on a space of
 dimension N = (Me + Mh)*p, using p starting vectors whose level-0 components
 are W_e (e-sector) and W_h (h-sector).  The resulting MCF has the same block
 size p as the individual electron and hole MCFs.

 The Green function of the result satisfies:
   G_combined(z) = W_new^H F_new(z) W_new = G⁺(z) + (G⁻)ᵀ(z)

 matching the convention of the default (non-combined) path.  This is achieved
 by running blockLanczos on T_e ⊕ T_h* (T_h* = conj of T_h) and using
 conj(W_h) as h-sector starting vectors, so that the h contribution becomes
   conj(W_h)^H F_{T_h*} conj(W_h) = G_mcf_h^T = (G⁻)ᵀ
 For T=double (real Hamiltonian), T_h* = T_h and the modification is a no-op.

 M0 must be at least Me + Mh to capture the full Krylov space of the
 combined operator (each sector independently contributes Me and Mh steps).

 @param e   Electron MCF (block size p, Me floors).
 @param h   Hole MCF (block size p, Mh floors).
 @param M0  Maximum number of Lanczos steps; updated to actual count on return.
*/
template<typename T>
matrix_continued_fraction<T> combine_via_lanczos(
    const matrix_continued_fraction<T>& e,
    const matrix_continued_fraction<T>& h,
    int M0)
{
    const int p  = e.p;
    QCM_ASSERT(e.p == h.p);
    const int Me = e.floors(), Mh = h.floors();
    const int N  = (Me + Mh) * p;  // total dimension of the combined space

    // Build starting block: p vectors of length N.
    // phi[i]: W_e[:,i]       at e-sector level 0 (positions 0..p-1)
    // phi[i]: conj(W_h[:,i]) at h-sector level 0 (positions Me*p .. Me*p+p-1)
    // The conjugate on the h-sector, combined with the T_h* operator, ensures
    // the h contribution to evaluate() gives (G⁻)ᵀ instead of G⁻.
    vector<vector<T>> phi(p, vector<T>(N, T(0)));
    for(int i = 0; i < p; ++i){
        for(int k = 0; k < p; ++k){
            phi[i][k]          = mcf_to_T(e.W(k, i),       T(0));  // W_e
            phi[i][Me * p + k] = mcf_to_T(conj(h.W(k, i)), T(0));  // conj(W_h)
        }
    }

    // Extract the upper-triangular QR factor W_new from phi via modified
    // Gram-Schmidt (mirrors the procedure in model_instance::build_mcf).
    matrix<T> W_new(p);
    {
        vector<vector<T>> q(phi);   // working copies
        for(int l = 0; l < p; ++l){
            for(int k = 0; k < l; ++k){
                T z = q[k] * q[l];   // inner product <q[k]|q[l]>
                W_new(k, l) = z;
                mult_add(-z, q[k], q[l]);
            }
            double nrm = norm(q[l]);
            W_new(l, l) = T(nrm);
            q[l] *= 1.0 / nrm;
        }
    }

    // Run block Lanczos on T_e ⊕ T_h with the p starting vectors.
    // block_Lanczos_QR=true (default) uses QR; false uses polar decomposition (Hermitian B).
    combined_sector_operator<T> op(e, h);
    vector<matrix<T>> A_new, B_new;
    bool use_QR = global_bool("block_Lanczos_QR");
    if(use_QR)
        blockLanczos(op, phi, A_new, B_new, M0);
    else
        blockLanczosSVD(op, phi, A_new, B_new, M0);

    matrix_continued_fraction<T> mcf(A_new, B_new, to_complex_matrix(W_new));
    mcf.hermitian_B = !use_QR;
    return mcf;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const matrix_continued_fraction<T>& F)
{
    int M = F.floors();
    os << "matrix_continued_fraction: p=" << F.p << "  floors=" << M
       << "  B=" << (F.hermitian_B ? "Hermitian" : "upper-triangular") << '\n';
    for(int j = 0; j < M; ++j){
        os << "A[" << j << "]:\n" << F.A[j];
    }
    for(int j = 0; j < (int)F.B.size(); ++j){
        os << "B[" << j << "]:\n" << F.B[j];
    }
    os << "W:\n" << F.W;
    return os;
}


template<typename T>
void h5_write_mcf(H5::Group& grp, const matrix_continued_fraction<T>& F)
{
    int M = F.floors();
    h5_write_attr(grp, "p",           F.p);
    h5_write_attr(grp, "floors",      M);
    h5_write_attr(grp, "hermitian_B", F.hermitian_B ? 1 : 0);
    for(int j = 0; j < M; ++j){
        H5::Group ag = grp.createGroup("A_" + to_string(j));
        h5_write_mat(ag, "data", F.A[j]);
    }
    for(int j = 0; j < (int)F.B.size(); ++j){
        H5::Group bg = grp.createGroup("B_" + to_string(j));
        h5_write_mat(bg, "data", F.B[j]);
    }
    H5::Group wg = grp.createGroup("W");
    h5_write_mat(wg, "data", F.W);
}


template<typename T>
void h5_read_mcf(H5::Group& grp, matrix_continued_fraction<T>& F)
{
    F.p   = h5_read_attr_int(grp, "p");
    int M = h5_read_attr_int(grp, "floors");
    // hermitian_B is absent in files written before this flag existed;
    // default to false (upper-triangular) for backward compatibility.
    F.hermitian_B = grp.attrExists("hermitian_B") && h5_read_attr_int(grp, "hermitian_B") != 0;
    F.A.resize(M);
    F.B.resize(M);
    for(int j = 0; j < M; ++j){
        H5::Group ag = grp.openGroup("A_" + to_string(j));
        h5_read_mat(ag, "data", F.A[j]);
    }
    for(int j = 0; j < M; ++j){
        H5::Group bg = grp.openGroup("B_" + to_string(j));
        h5_read_mat(bg, "data", F.B[j]);
    }
    H5::Group wg = grp.openGroup("W");
    h5_read_mat(wg, "data", F.W);
}


#endif
