#ifndef hdf5_io_h
#define hdf5_io_h

/**
 hdf5_io.hpp
 Utility helpers for reading/writing common qcm_wed types to HDF5 groups.
 Uses the HDF5 C++ API (H5Cpp.h).

 Supported primitives:
   - scalar attributes: double, int, string
   - vector<double>
   - matrix<double>  — written as group {r,c attrs + "data" dataset}
   - matrix<Complex> — written as group {r,c attrs + "re"/"im" datasets}
*/

#include <H5Cpp.h>
#include <string>
#include <vector>
#include <complex>

#include "types.hpp"   // for Complex = complex<double>
#include "matrix.hpp"

using std::string;
using std::vector;

// ---------------------------------------------------------------------------
// Scalar attributes (written on an H5::Group)
// ---------------------------------------------------------------------------
void h5_write_attr(H5::Group& g, const string& name, double v);
void h5_write_attr(H5::Group& g, const string& name, int v);
void h5_write_attr(H5::Group& g, const string& name, const string& v);

double h5_read_attr_double(const H5::Group& g, const string& name);
int    h5_read_attr_int   (const H5::Group& g, const string& name);
string h5_read_attr_string(const H5::Group& g, const string& name);

// ---------------------------------------------------------------------------
// 1-D vector<double>
// ---------------------------------------------------------------------------
void h5_write_vec(H5::Group& g, const string& name, const vector<double>& v);
void h5_read_vec (const H5::Group& g, const string& name, vector<double>& v);

// ---------------------------------------------------------------------------
// matrix<double>  (column-major flat storage, shape r×c)
// Written as a sub-group with attrs "r","c" and dataset "data".
// ---------------------------------------------------------------------------
void h5_write_mat(H5::Group& g, const string& name, const matrix<double>& m);
void h5_read_mat (const H5::Group& g, const string& name, matrix<double>& m);

// ---------------------------------------------------------------------------
// matrix<Complex>  (column-major flat storage)
// Written as a sub-group with attrs "r","c" and datasets "re","im".
// ---------------------------------------------------------------------------
void h5_write_mat(H5::Group& g, const string& name, const matrix<Complex>& m);
void h5_read_mat (const H5::Group& g, const string& name, matrix<Complex>& m);

#endif /* hdf5_io_h */
