#ifndef qcm_index_h
#define qcm_index_h

/**
 * @file index.hpp
 * @brief Definition of the Index struct (multi-dimensional index labelling).
 */

#include <iostream>

#include "vector_num.hpp"

using namespace std;

//! Used to describe a multi-index labelling, such as in a tensor
struct Index{
	vector<int> dim; // array of dimensions for each index
	vector<int> ind; // array of indices
	int dim_tot; // total number of states (product of all the d[i]'s)
	
	Index(){
		dim_tot = 0;
	}
	
	Index(vector<int> &_dim) : dim(_dim){
		ind.resize(dim.size());
		set_dim_tot();
	}
	
	Index(const Index &x): dim(x.dim), ind(x.ind), dim_tot(x.dim_tot){}
	
	void set_dim_tot(){
		dim_tot = 1;
		for(int j=0; j<dim.size(); ++j) dim_tot *= dim[j];
	}
	
	// returns the compound index
	int operator()() {
		int label = ind[dim.size()-1];
		for(int j=(int)dim.size()-2; j>=0; j--) label = dim[j]*label + ind[j];
		return label;
	}
	
	// builds the indices from the compound index 'label'
	void operator()(int label){
		for(int j=0; j<dim.size(); ++j){
			ind[j] = label%dim[j];
			label /= dim[j];
		}
	}
	
	
	// returns the label associated with group product
	int operator()(int i1, int i2){
		(*this)(i1);
		vector<int> tmp(ind);
		(*this)(i2);
		for(int j=0; j<dim.size(); ++j){
			ind[j] += tmp[j];
			ind[j] %= dim[j];
		}
		return (*this)();
	}
	
	
	// increases the compound index by one and adjusts the indices
	Index & operator++(){
		shift(0);
		return *this;
	}
	
	void shift(int pos){
		if(pos == dim.size()) return;
		ind[pos]++;
		if(ind[pos]%dim[pos] == 0){
			ind[pos]=0;
			shift(pos+1);
		}
	}
	
	// hermitian product useful if the index represent a group element
	int operator*(const Index &x){
		int prod = 0;
		for(int j=0; j<dim.size(); ++j){
			prod += (ind[j]*x.ind[j]*dim_tot)/dim[j];
		}
		return prod%dim_tot;
	}
	
	//! writes to a stream
	friend std::ostream & operator<<(std::ostream &flux, const Index &A){
		flux << '[' << A.ind[0];
		for(int i=1; i<A.dim.size(); ++i) flux << ", " << A.ind[i];
		flux << ']';
		return flux;
	}
	
	string str(){
		ostringstream sout;
		sout << *this;
		return sout.str();
	}
};


#endif
