#ifndef index_pair_h
#define index_pair_h

/**
 * @file index_pair.hpp
 * @brief Definition of the index_pair struct (row/column locator for matrix elements).
 */

#include <iostream>

using namespace std;

//! used to store the location of a matrix element.
struct index_pair
{
	size_t r;
	size_t c;
	
	index_pair() : r(0), c(0) {}
	
	index_pair(size_t _r, size_t _c) : r(_r), c(_c) {}
  
  index_pair swap(){return index_pair(c, r);}
	
	friend std::ostream & operator<<(std::ostream &s, const index_pair &e)
	{
		s << '(' << e.r+1 << ',' << e.c+1 << ")";
		return s;
	}
	
	string str() const{
		ostringstream sout;
		sout << *this;
		return sout.str();
	}

	friend inline bool operator<(const index_pair &x, const index_pair &y){
		if(x.r < y.r) return true;
		else if(x.r > y.r) return false;
		else if(x.c < y.c) return true;
		else if(x.c > y.c) return false;
		else return false;
	}
	
	
};


#endif
