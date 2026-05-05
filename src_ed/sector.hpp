#ifndef sector_h
#define sector_h

/**
 * @file sector.hpp
 * @brief Definition of the sector struct and HS_mixing namespace (Hilbert-space sectors).
 */

#include <cstring>
#include "parser.hpp"

namespace HS_mixing {
  const int normal=0;
  const int anomalous=1;
  const int spin_flip=2;
  const int full=3;
  const int up_down=4;
}

void qcm_ED_throw(const std::string& s);

//! represents a sector of the Hilbert space
struct sector{
	const static int odd = 9999; //!< value of N or S if not conserved but odd
	const static int even = 10000; //!< value of N or S if not conserved but even
	int N; //!< the total number of particles in the sector (N_1 + N_2).  = even or odd is particle number is not conserved
	int S; //!< the total spin in the sector (N_1 - N_2). = even or odd if spin is not conserved.
	size_t irrep; //!< the label of the point group representation used (from 0 to g-1)
	
	sector(): N(0), S(0), irrep(0) {}
	
	sector(int _N, int _S, size_t _irrep): N(_N), S(_S), irrep(_irrep) {}
	
  // returns the number of up spins
  // N = Nup+Ndw, S = Nup-Ndw ---> Nup = (N+S)/2
  int Nup() const{
    if(N==even or S==even) return even;
    else if(N==odd or S==odd) return odd;
    else return (N+S)/2;
  }

  // returns the number of down spins
  // N = Nup+Ndw, S = Nup-Ndw ---> Ndw = (N-S)/2
  int Ndw() const{
    if(N==even or S==even) return even;
    else if(N==odd or S==odd) return odd;
    else return (N-S)/2;
  }


	/**
   constructor from a string
   */
	sector(const string &str) {
		
		bool valid = true;
    int nel=0;
    string tmp_str;

    // reading N
    int loc = str.find("N");
    if(loc != string::npos){
      tmp_str = str.substr(loc);
      nel = sscanf(tmp_str.c_str(),"N%d",&N);
      if(nel == 0) valid = false;
    }
    else N = even;

    // reading S
    loc = str.find("S");
    if(loc != string::npos){
      tmp_str = str.substr(loc);
      nel = sscanf(tmp_str.c_str(),"S%d",&S);
      if(nel == 0) valid = false;
    }
    else S = even;

    // reading R
    loc = str.find("R");
    if(loc != string::npos){
      tmp_str = str.substr(loc);
      nel = sscanf(tmp_str.c_str(),"R%ld",&irrep);
      if(nel == 0) valid = false;
    }
    else irrep = 0;

    if(!valid){
      qcm_ED_throw("sector string " + str + " does not conform to standard!");
    }

		if(str.find("O") != string::npos) N = S = odd;
    if(S == even && N != even && N%2) S = odd;
    if(N == even && S != even && S%2) N = odd;

  }
		
	
	
	
	friend std::ostream & operator<<(std::ostream &flux, const sector &s){
		if(s.S == s.odd && s.N == s.odd) flux << "O";
		flux << "R" << s.irrep;
		if(s.N != s.even && s.N != s.odd) flux << ":N" << s.N;
		if(s.S != s.even && s.S != s.odd) flux << ":S" << s.S;
		return flux;
	}
	
	
	
	
	
	friend std::istream & operator>>(std::istream &flux, sector &s){
		char tmp_str[32];
		flux >> tmp_str;
		
		if(strchr(tmp_str,'R')!=nullptr){
			if(strchr(tmp_str,'S')==nullptr){
				s.S = s.even;
				if(strchr(tmp_str,'N')==nullptr) s.N = s.even;
				else sscanf(tmp_str,"R:%ld,N:%d",&s.irrep,&s.N);
        if(s.S == even && s.N != even && s.N%2) s.S = odd;
			}
			else{
				if(strchr(tmp_str,'N')==nullptr){
					s.N = s.even;
					sscanf(tmp_str,"R:%ld,S:%d",&s.irrep,&s.S);
          if(s.S%2) s.N = odd;
				}
				else sscanf(tmp_str,"R:%ld,N:%d,S:%d",&s.irrep,&s.N,&s.S);
			}
		}
		else{
			qcm_ED_throw("sector string " + s.name() + " does not conform to standard!");
		}
		return flux;
	}
	
  
  
  
	string name() const{
		ostringstream s;
		s << *this;
		return s.str();
	}
	
};


namespace std
{
	template<>
	struct less<sector>{
		bool operator()(const sector &x, const sector &y) const{
			if(x.S < y.S) return true;
			else if(x.S > y.S) return false;
			else if(x.N < y.N) return true;
			else if(x.N > y.N) return false;
			else if(x.irrep < y.irrep) return true;
			else return false;
		}
	};
}



bool operator!=(const sector &S1, const sector &S2);
bool operator==(const sector &S1, const sector &S2);
bool operator>(const sector &S1, const sector &S2);
bool operator<(const sector &S1, const sector &S2);




#endif
