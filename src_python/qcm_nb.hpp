#ifndef qcm_nb_h
#define qcm_nb_h

// Shared nanobind helpers used by the QCM and QCM_ED registration functions.

#include <algorithm>
#include <complex>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "matrix.hpp"

namespace nb = nanobind;
using namespace nb::literals;

//------------------------------------------------------------------------------
// Build an owning NumPy array from a contiguous C++ buffer (flat copy, C order).
// The legacy wrappers flat-copied the (column-ordered) matrix<> storage into a
// C-ordered NumPy array; copying the same flat buffer here reproduces the exact
// same bytes (and the historical implicit transpose for square matrices).
//------------------------------------------------------------------------------
template <class T>
inline nb::ndarray<nb::numpy, T> nb_array_(const T *src,
                                           std::initializer_list<size_t> shape) {
  size_t n = 1;
  for (auto s : shape) n *= s;
  T *d = new T[n ? n : 1];
  if (n) std::copy(src, src + n, d);
  nb::capsule owner(d, [](void *p) noexcept { delete[] static_cast<T *>(p); });
  return nb::ndarray<nb::numpy, T>(d, shape, owner);
}

//! (r, c) complex array from a matrix<complex<double>>.
//! (matrix<>::data()/size() are not const, hence the non-const reference.)
inline nb::ndarray<nb::numpy, std::complex<double>>
nb_cmat(matrix<std::complex<double>> &g) {
  return nb_array_<std::complex<double>>(g.data(), {g.r, g.c});
}

//! (N, d, d) complex array from a vector of d x d matrices (flat-stacked).
inline nb::ndarray<nb::numpy, std::complex<double>>
nb_cstack(std::vector<matrix<std::complex<double>>> &g, size_t d) {
  size_t N = g.size();
  size_t tot = N * d * d;
  auto *buf = new std::complex<double>[tot ? tot : 1];
  for (size_t j = 0; j < N; j++)
    std::copy(g[j].data(), g[j].data() + g[j].size(), buf + j * d * d);
  nb::capsule owner(
      buf, [](void *p) noexcept { delete[] static_cast<std::complex<double> *>(p); });
  return nb::ndarray<nb::numpy, std::complex<double>>(buf, {N, d, d}, owner);
}

#endif
