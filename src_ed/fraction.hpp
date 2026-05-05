#ifndef fraction_h
#define fraction_h

/**
 * @file fraction.hpp
 * @brief Definition of the fraction struct (simple integer fractions for pretty printing).
 */

//! simple fractions
/**
 This is only used in the pretty printing of the basis, for debugging purposes.
 Hence this is not very important.
*/
struct fraction{
  int num;
  int denom;
  const static int primes[];
  const static int nprimes;
  
  fraction(int _n, int _d) : num(_n), denom(_d) {}
  
  void simplify(){
    for(int i=0; i<nprimes; i++){
      if(num%primes[i] == 0 and denom%primes[i] == 0){
        num /= primes[i];
        denom /= primes[i];
        simplify();
      }
    }
  }
  
  friend std::ostream & operator<<(std::ostream &flux, const fraction &s){
    flux << s.num << '/' << s.denom;
    return flux;
  }
};

const int fraction::primes[] = {2,3,5,7,11,13,17,19};
const int fraction::nprimes = 8;



#endif /* fraction_h */
