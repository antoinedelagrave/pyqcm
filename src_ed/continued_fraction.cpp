/**
 * @file continued_fraction.cpp
 * @brief Implementation of the scalar Jacobi continued_fraction class.
 */
#include "continued_fraction.hpp"

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





