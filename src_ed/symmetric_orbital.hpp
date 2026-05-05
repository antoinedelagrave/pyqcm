#ifndef symmetric_orbital_h
#define symmetric_orbital_h

/**
 * @file symmetric_orbital.hpp
 * @brief Definition of the symmetric_orbital struct (orbital labelled by irrep/spin/Nambu).
 */

#include <vector>

using namespace std;

//! represents a symmetrized orbital in the case of a non trivial point group symmetry
struct symmetric_orbital{
  int label; //!< label of symmetric orbital
  int irrep; //!< irreducible representation
  int orb;   //!< label of symmetric orbital within irrep
  int spin;  //!< spin index (0 or 1)
  int nambu; //!< nambu index (0 or 1)
  
  symmetric_orbital() : label(0), irrep(0), orb(0), spin(0), nambu(0) {}
  
  symmetric_orbital(int _irrep, int _orb, int _spin, int _nambu, vector<int> &site_irrep_dim) : label(0), irrep(_irrep), orb(_orb), spin(_spin), nambu(_nambu) {
    for(int i=0; i<irrep; i++) label += site_irrep_dim[i];
    label += orb;
  }
  
  friend std::ostream & operator<<(std::ostream &flux, const symmetric_orbital &s){
    flux << '[' << s.irrep << ':';
    flux << s.orb << ':' << ((s.spin)? '-' : '+') << ']';
    if(s.nambu) flux << "_nambu";
    return flux;
  }
  
  string str(){
    ostringstream sout;
    sout << *this;
    return sout.str();
  }
  
  friend bool operator<(const symmetric_orbital &x, const symmetric_orbital &y){
    if(x.label + x.spin*100 < y.label + y.spin*100) return true;
    else if(x.label + x.spin*100 > y.label + y.spin*100) return false;
    else if(x.nambu < y.nambu) return true;
    else return false;
  }
  friend bool operator>(const symmetric_orbital &x, const symmetric_orbital &y){
    if(x.label + x.spin*100 > y.label + y.spin*100) return true;
    else if(x.label + x.spin*100 < y.label + y.spin*100) return false;
    else if(x.nambu > y.nambu) return true;
    else return false;
  }

};


#endif
