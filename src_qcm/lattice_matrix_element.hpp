#ifndef lattice_matrix_element_h
#define lattice_matrix_element_h

/**
 * @file lattice_matrix_element.hpp
 * @brief Definitions of lattice_matrix_element and related index/GF element structs.
 */

#include "types.hpp"

using namespace std;

//! Matrix element of a lattice operator
struct lattice_matrix_element
{
  size_t site1; //!< row site index
  size_t spin1; //!< rwo spin index
  size_t site2; //!< column site index
  size_t spin2; //!< column spin index
  size_t neighbor; //!< index of the neighboring SUC
  complex<double> v; //!< value of the matrix element
  
  //! constructor
  lattice_matrix_element() : site1(0), site2(0), spin1(0), spin2(0), neighbor(0), v(0.0) {}
  
  //! constructor
  lattice_matrix_element(const int &_site1, const int &_spin1, const int &_site2, const int &_spin2, const int _neighbor, complex<double> _v)
  : site1(_site1), spin1(_spin1), site2(_site2), spin2(_spin2), neighbor(_neighbor), v(_v){}

  friend std::ostream & operator<<(std::ostream &flux, const lattice_matrix_element &s){
    if(s.v.imag() == 0) flux << '(' << s.site1+1 << (s.spin1? '-' : '+') <<  ',' << s.site2+1 << (s.spin2? '-' : '+') << ';' << s.neighbor << ") : " << s.v.real();
    else flux << '(' << s.site1+1 << (s.spin1? '-' : '+') <<  ',' << s.site2+1 << (s.spin2? '-' : '+') << ';' << s.neighbor << ") : " << s.v;
    return flux;
  }
  
  string str() const{
    ostringstream sout;
    sout << *this;
    return sout.str();
  }
};


//! Matrix element indices for a lattice operator, for sorting
struct lattice_index_pair
{
  static size_t Nc; //! number of orbitals in the SUC (from the model)
  size_t site1; //!< row site index
  size_t spin1; //!< rwo spin index
  size_t site2; //!< column site index
  size_t spin2; //!< column spin index
  size_t neighbor; //!< index of the neighboring SUC
  size_t rank; //!< flattened index
  
  //! constructor
  lattice_index_pair() : site1(0), site2(0), spin1(0), spin2(0), neighbor(0), rank(0) {}
  
  //! constructor
  lattice_index_pair(const int &_site1, const int &_spin1, const int &_site2, const int &_spin2, const int _neighbor)
  : site1(_site1), spin1(_spin1), site2(_site2), spin2(_spin2), neighbor(_neighbor){
    rank = site1+Nc*(spin1 + 2*(site2 + Nc*(spin2 + 2*neighbor)));
  }

  friend std::ostream & operator<<(std::ostream &flux, const lattice_index_pair &s){
    flux << '(' << s.site1+1 << (s.spin1? '-' : '+') <<  ',' << s.site2+1 << (s.spin2? '-' : '+') << ';' << s.neighbor << ")";
    return flux;
  }
  
  string str() const{
    ostringstream sout;
    sout << *this;
    return sout.str();
  }

  friend bool operator<(const lattice_index_pair &x, const lattice_index_pair &y){
    return x.rank < y.rank;
  }

};




//! matrix element of the lattice Green function, in a k-independent way
template<typename HilbertField>
struct GF_matrix_element
{
  
  size_t r; //!< row composite index
  size_t c; //!< column composite index
  size_t n; //!< neighbor index
  HilbertField v; //!< value of the matrix element
};

#endif
