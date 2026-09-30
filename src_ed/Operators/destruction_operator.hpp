/*
 Class for a destruction operator
*/
#ifndef destruction_operator_h
#define destruction_operator_h

#include <memory>
#include <algorithm>

#include "types.hpp"
#include "ED_basis.hpp"
#include "index_pair.hpp"
#include "matrix_element.hpp"
#include "HS_operator.hpp"

using namespace std;

//! identifies a destruction operator based on orbital index and Hilbert space sector
/**
Used for storing the operators in a map
 */
struct destruction_identifier{
  sector secB;
  symmetric_orbital sorb;
  
  destruction_identifier(const sector &_secB, const symmetric_orbital &_sorb) :
  secB(_secB), sorb(_sorb) {}
  
};

// ordering used for storing in a map (recent libc++ versions call operator< directly, bypassing std::less)
inline bool operator<(const destruction_identifier &x, const destruction_identifier &y){
  if(x.sorb < y.sorb) return true;
  else if(x.sorb > y.sorb) return false;
  else{
    if(x.secB > y.secB) return true;
    else return false;
  }
}


namespace std
{
  template<>
  struct less<destruction_identifier>{
    bool operator()(const destruction_identifier &x, const destruction_identifier &y) const{
      return x < y;
    }
  };
}



//==============================================================================

//! destruction operator in a given Hilbert space sector
/**
Destruction operators are stored in memory, unless the Hamiltonian format is "factorized" or "onthefly"
*/
template<typename HilbertField>
struct destruction_operator : HS_operator<HilbertField>
{
  shared_ptr<ED_mixed_basis> B; //!< Basis of domain
  shared_ptr<ED_mixed_basis> T; //!< Basis of image
  symmetric_orbital sorb;

  destruction_operator();
  destruction_operator(shared_ptr<ED_mixed_basis> _B, shared_ptr<ED_mixed_basis> _T, const symmetric_orbital &orb);
};


//==============================================================================
// implementation


template<typename HilbertField>
std::ostream & operator<<(std::ostream &flux, const destruction_operator<HilbertField> &x)
{
  flux << "destruction operator " << x.sorb << " (" << x.r << " x " << x.c << ") from " << x.B->sec << " to " << x.T->sec;
  return flux;
}

#endif
