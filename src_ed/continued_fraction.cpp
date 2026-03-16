#include "continued_fraction.hpp"
#include "parser.hpp"
#include "symmetry_group.hpp"
#include "sector.hpp"

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
 prints on a stream (for debugging)
 @param flux output stream
 @param F continued fraction to print
 */
std::ostream& operator<<(std::ostream &flux, const continued_fraction &F)
{
  flux << "floors: " << F.a.size() << endl;
  for(size_t i=0; i<F.a.size();++i) flux << F.a[i] << '\t' << F.b[i] << '\n';
  return flux;
}






/**
 reads from a stream
 @param flux input stream
 @param F continued fraction to populate
 */
std::istream& operator>>(std::istream &flux, continued_fraction &F)
{
  string tmp;
  size_t n;
  flux >> tmp >> n; // next_line(flux);
  F.a.resize(n);
  F.b.resize(n);
  for(int i=0; i<n; i++){
    flux >> F.a[i] >> F.b[i];
  }
  return flux;
}


