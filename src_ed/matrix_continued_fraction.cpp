#include "matrix_continued_fraction.hpp"
#include "parser.hpp"
#include "hdf5_io.hpp"

/**
 Default constructor.
*/
matrix_continued_fraction::matrix_continued_fraction() : p(0)
{
}


/**
 Evaluate the matrix continued fraction at complex frequency z.

 Backward recursion (deepest level first):

   F = 0_{p×p}                         (truncation: F_M = 0)
   for j = M-1 downto 0:
     Σ = B[j]^H F B[j]                 (if j < M-1; else Σ = 0)
     F = (z·I - A[j] - Σ)^{-1}

 Then return G(z) = W^H F W.

 The last element B[M-1] is the off-diagonal block produced by the final
 QR step of blockLanczos but is not part of the truncated block-tridiagonal;
 it is stored for diagnostic purposes (truncation error) and is not used
 here, exactly as the last element of the b array in continued_fraction.

 @param z  Complex frequency.
 @returns  p×p Green-function matrix.
*/
matrix<Complex> matrix_continued_fraction::evaluate(Complex z) const
{
    int M = (int)A.size();

    // F starts as the M-th floor: F_M = 0
    matrix<Complex> F(p, p);   // zero-initialised

    for(int j = M - 1; j >= 0; --j){

        // Σ = B[j]^H · F · B[j]  (only when j < M-1; otherwise Σ = 0)
        matrix<Complex> sigma(p, p);  // zero-initialised
        if(j < M - 1){
            // step 1: tmp = F · B[j]
            matrix<Complex> tmp(p);
            tmp.product(F, B[j]);
            // step 2: sigma = B[j]^H · tmp = HC(B[j]) · (F · B[j])
            matrix<Complex> Bjh(B[j]);
            Bjh.hermitian_conjugate();
            sigma.product(Bjh, tmp);
        }

        // Build denominator D = z·I - A[j] - Σ
        //   matrix<Complex>::operator+=(Complex) adds a scalar to all diagonal elements
        //   matrix<Complex>::operator-=(matrix<Complex>) subtracts element-wise
        matrix<Complex> D(p, p);    // zero
        D += z;                     // D = z·I
        D -= A[j];                  // D = z·I - A[j]
        D -= sigma;                 // D = z·I - A[j] - Σ

        D.inverse();                // D  →  D^{-1}
        F = D;                      // F_{j} = D^{-1}
    }

    // Apply weight: G = W^H · F_0 · W
    // (When W is the identity this reduces to F_0.)
    matrix<Complex> tmp(p);
    tmp.product(F, W);              // tmp = F · W
    matrix<Complex> Wh(W);
    Wh.hermitian_conjugate();       // Wh  = W^H
    matrix<Complex> G(p);
    G.product(Wh, tmp);             // G   = W^H · F · W

    return G;
}


/**
 Writes a matrix_continued_fraction to an HDF5 group.
 Layout:
   attributes: "p" (int), "floors" (int)
   sub-groups "A_0" … "A_{M-1}" — each a p×p complex matrix via h5_write_mat
   sub-groups "B_0" … "B_{M-1}" — idem
   sub-group  "W"               — p×p complex weight matrix
*/
void h5_write_mcf(H5::Group& grp, const matrix_continued_fraction& F)
{
    int M = F.floors();
    h5_write_attr(grp, "p",      F.p);
    h5_write_attr(grp, "floors", M);
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


/**
 Reads a matrix_continued_fraction from an HDF5 group written by h5_write_mcf.
*/
void h5_read_mcf(H5::Group& grp, matrix_continued_fraction& F)
{
    F.p      = h5_read_attr_int(grp, "p");
    int M    = h5_read_attr_int(grp, "floors");
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