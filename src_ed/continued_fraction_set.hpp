#ifndef continued_fraction_set_h
#define continued_fraction_set_h

#include "Green_function_set.hpp"
#include "continued_fraction.hpp"

//! set of Jacobi continued fraction for the whole Green function
struct continued_fraction_set : Green_function_set
{
  
  vector<matrix<continued_fraction>> e; 	//!< electron fractions
  vector<matrix<continued_fraction>> h; 	//!< hole fractions
  sector sec; //!< sector
  bool is_complex;

  continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, bool _is_complex);
  continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>> &_a, const vector<vector<double>> &b, bool _is_complex);

  // realizations of base class virtual methods
  void Green_function(const Complex &z, block_matrix<Complex> &G) override;
  void integrated_Green_function(block_matrix<Complex> &M) override;
  void write_hdf5(H5::Group& grp) override;
  void read_hdf5(H5::Group& grp) override;
};


#endif

