#ifndef lattice_hybrid_h
#define lattice_hybrid_h

/**
 * @file lattice_hybrid.hpp
 * @brief Declaration of the lattice_hybrid struct (external hybridization function).
 */

#include <iostream>

#include <complex>
#include <cstring>
#include <ctype.h>
#include "types.hpp"
#include "vector3D.hpp"

using namespace std;

//! External hybridization function loaded from an HDF5 file (frequencies, k-points, mixing).
struct lattice_hybrid {
  vector<double> w; //!< array of frequencies. Interpreted as imaginary (Matsubara) if eta==0, otherwise as real frequencies (the actual frequency being w + i*eta).
  vector<double> weight; //!< weight of each frequency in an integral over frequencies
	vector<vector3D<double>> k; //!< list of wavevectors (k-points) used with external hybridization. In the superdual basis.
  size_t nw=0; //!< number of frequencies
  size_t nk=0; //!< number of k points
  size_t d=0; //!< dimension of hybridization function
  vector<double> R; //!< array of values (real part, ow-major)
  vector<double> I; //!< array of values (imaginary part, ow-major)
  int mixing; //!< mixing state of the hybridization function
  double eta=0.0; //!< imaginary part added to the frequency on the real axis. If zero, the grid 'w' lies on the imaginary (Matsubara) axis; otherwise 'w' is real and the frequency is w + i*eta.
  lattice_hybrid(const string &filename); //!< constructor from HDF5 file
};

#endif


