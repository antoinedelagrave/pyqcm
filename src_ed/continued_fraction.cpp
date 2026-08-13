/**
 * @file continued_fraction.cpp
 * @brief Implementation of the scalar Jacobi continued_fraction class.
 */
#include "continued_fraction.hpp"
#include "matrix_continued_fraction.hpp"

/** default constructor
 */
continued_fraction::continued_fraction()
{
}




/** constructor from data in ready format
 @param _a partial denominators
 @param _b partial numerators
 */
continued_fraction::continued_fraction(const vector<double>& _a, const vector<double>& _b) : a(_a), b(_b) {}




/** constructor
 switch the data from tridiagonal form (obtained from the Lanczos method) to continued fraction form
 @param _a first diagonal
 @param _b second diagonal
 @param e0 Ground state energy
 @param norm norm of the first state of the Lanczos sequence
 @param create true for creation, false for destruction
 */
continued_fraction::continued_fraction(vector<double>& _a, vector<double>& _b, double e0, double norm, bool create) : a(_a), b(_b)
{
  for(size_t i=0; i< b.size(); ++i) b[i] *= b[i];
  b[0] = norm;
  if(create) for(size_t i=0; i< a.size(); ++i) a[i] -= e0;
  else for(size_t i=0; i< a.size(); ++i) a[i]  = -a[i] + e0;
}








/**
 evaluates the continued fraction for a given complex frequency \a z
 @param z complex frequency
 */
Complex continued_fraction::evaluate(Complex z)
{
  Complex G(0.0);
  for(int i=(int)a.size()-1; i>=0 ; i--) G = b[i]/(z-a[i]-G);
  return G;
}




/**
 Frequency-integrated weight: ∫_{-∞}^{0} A(ω) dω, i.e. the contribution of
 this continued fraction's negative-energy poles only.

 A naive "the whole fraction is at negative energy" assumption (equivalent
 to just returning b[0], the total sum rule) is only exact when the
 reference state used to build this continued fraction was a true ground
 state; it can fail for an excited, thermally-weighted reference state. This
 diagonalizes the underlying Jacobi (tridiagonal) matrix instead of assuming
 the sign, by reusing matrix_continued_fraction<double>::integrated_Green_function()
 with block size 1: diagonal a[j], off-diagonal sqrt(b[j+1]), and weight
 sqrt(b[0]) (b[0] holds the sum rule, not a matrix element).
 */
double continued_fraction::integrated_weight() const
{
  int n = (int)a.size();
  if(n == 0) return 0.0;

  vector<matrix<double>> A(n), B(n);
  for(int j = 0; j < n; ++j){
    A[j] = matrix<double>(1,1);
    A[j](0,0) = a[j];
    B[j] = matrix<double>(1,1);
    B[j](0,0) = (j < n-1) ? sqrt(max(b[j+1], 0.0)) : 0.0;
  }
  matrix<Complex> W(1,1);
  W(0,0) = Complex(sqrt(max(b[0], 0.0)), 0.0);

  matrix_continued_fraction<double> mcf(A, B, W);
  matrix<Complex> G = mcf.integrated_Green_function();
  return G(0,0).real();
}





