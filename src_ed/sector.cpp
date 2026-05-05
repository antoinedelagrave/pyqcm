/**
 * @file sector.cpp
 * @brief Comparison operators for the sector struct.
 */
#include "sector.hpp"

bool operator!=(const sector &S1, const sector &S2){
	if(S1.N != S2.N or S1.S != S2.S or S1.irrep != S2.irrep) return true;
	else return false;
}


bool operator==(const sector &S1, const sector &S2){
	if(S1 != S2) return false;
	else return true;
}


bool operator>(const sector &S1, const sector &S2){
	if(S1.S < S2.S) return true;
	else if(S1.S > S2.S) return false;
	else if(S1.N > S2.N) return true;
	else if(S1.N < S2.N) return false;
	else if(S1.irrep > S2.irrep) return true;
	else return false;
}


bool operator<(const sector &S1, const sector &S2){
	if(S1.S > S2.S) return true;
	else if(S1.S < S2.S) return false;
	else if(S1.N < S2.N) return true;
	else if(S1.N > S2.N) return false;
	else if(S1.irrep < S2.irrep) return true;
	else return false;
}



