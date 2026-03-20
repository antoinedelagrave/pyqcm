/*
 Class for a Jacobi continued fraction
 */

#ifndef continued_fraction_h
#define continued_fraction_h

#include <cstdio>

#include "block_matrix.hpp"
#include "hdf5_io.hpp"

//! Represents a truncated Jacobi continued fraction.
struct continued_fraction
{

	vector<double> a; 	//!< array of partial denominators
	vector<double> b; 	//!< array partial numerators
	
  continued_fraction();
  continued_fraction(const vector<double>& _a, const vector<double>& _b);
  continued_fraction(vector<double>& _a, vector<double>& _b, double e0, double norm, bool create);
  Complex evaluate(Complex z);

  void write_hdf5(H5::Group& grp) const
  {
    h5_write_vec(grp, "a", a);
    h5_write_vec(grp, "b", b);
  }

  void read_hdf5(H5::Group& grp)
  {
    h5_read_vec(grp, "a", a);
    h5_read_vec(grp, "b", b);
  }
};





#endif
