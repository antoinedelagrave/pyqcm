#include "matrix_continued_fraction.hpp"
#include "parser.hpp"

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
 Write to a stream (for debugging / serialisation).
 Format mirrors continued_fraction::operator<<:

   floors: M
   block: p
   A[0]  …  A[M-1]      (each written as a matrix<Complex>)
   B[0]  …  B[M-1]      (idem)
   W                    (weight matrix)
*/
std::ostream& operator<<(std::ostream &flux, const matrix_continued_fraction &F)
{
    flux << "floors: " << F.floors() << "\tblock: " << F.p << "\n";
    for(int j = 0; j < F.floors(); ++j) flux << "A[" << j << "]\n" << F.A[j];
    for(int j = 0; j < (int)F.B.size(); ++j) flux << "B[" << j << "]\n" << F.B[j];
    flux << "W\n" << F.W;
    return flux;
}


/**
 Read from a stream (inverse of operator<<).
*/
std::istream& operator>>(std::istream &flux, matrix_continued_fraction &F)
{
    string tmp;
    int M;
    flux >> tmp >> M >> tmp >> F.p;   // "floors: M  block: p"

    F.A.resize(M);
    F.B.resize(M);
    F.W.set_size(F.p);

    for(int j = 0; j < M; ++j){
        F.A[j].set_size(F.p);
        flux >> tmp >> F.A[j];        // "A[j]" then matrix
    }
    for(int j = 0; j < M; ++j){
        F.B[j].set_size(F.p);
        flux >> tmp >> F.B[j];        // "B[j]" then matrix
    }
    flux >> tmp >> F.W;               // "W" then matrix
    return flux;
}