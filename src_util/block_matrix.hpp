#ifndef block_matrix_h
#define block_matrix_h

/**
 * @file block_matrix.hpp
 * @brief Definition of the block_matrix template (block-diagonal matrix storage).
 */

#include <iostream>
#include <vector>

#include "matrix.hpp"

//! lock-diagonal matrix (of ints, double or Complex types)
template<typename T>
struct block_matrix
{
	size_t r;	//!< total number of rows
	size_t c;	//!< total number of rows
	vector<matrix<T>> block; //!< blocks
	vector<size_t> cumul_r; //!< total number of rows before each block
	vector<size_t> cumul_c; //!< total number of columns before each block
	
	//! default constructor
	block_matrix(): r(0),c(0) 
	{
		set_size();
	}
	
	//! constructor from a number of blocks
	block_matrix(size_t n)
	{
		block.assign(n, matrix<T>());
	}

	//! constructor from a vector of dimensions
	block_matrix(vector<size_t> dim)
	{
    block.assign(dim.size(), matrix<T>());
    for(size_t i=0; i<dim.size(); i++) block[i].set_size(dim[i]);
    set_size();
	}
	
	//! constructor from a vector of dimensions
	block_matrix(vector<int> dim)
	{
    block.assign(dim.size(), matrix<T>());
    for(size_t i=0; i<dim.size(); i++) block[i].set_size(dim[i]);
    set_size();
	}
	
	//! constructor from a vector of matrices
  block_matrix(vector<matrix<T> > &_block) : block(_block)
	{
    set_size();
	}
	
	//! assignation operator overloading
	block_matrix& operator=(const block_matrix<T>& A)
	{
		r = A.r;
		c = A.c;
		cumul_r = A.cumul_r;
		cumul_c = A.cumul_c;
		block = A.block;
		return(*this);
	}
	
	
	//! computes the size
	void set_size(){
		cumul_r.clear();
		cumul_r.reserve(block.size());
		r = 0;
		for(size_t i=0; i<block.size(); ++i){
			cumul_r.push_back(r);
			r += block[i].r;
		}
		cumul_c.clear();
		cumul_c.reserve(block.size());
		c = 0;
		for(size_t i=0; i<block.size(); ++i){
			cumul_c.push_back(c);
			c += block[i].c;
		}
	}
	
	//! element access (for rhs)
	inline T operator()(const size_t &i, const size_t &j)const{
		size_t I= block.size()-1; while(cumul_r[I]>i) I--;
		size_t J= block.size()-1; while(cumul_c[J]>i) J--;
		if(I != J) return(0.0);
		else return (block[I])(i-cumul_r[I],j-cumul_c[I]);
	}
	
	//! construct a Matrix from a block_matrix
	matrix<T> build_matrix(){
    matrix<T> A(r,c);
		for(size_t i=0; i<block.size(); i++){
			for(size_t j=0; j<block[i].r; j++){
				for(size_t k=0; k<block[i].c; k++){
					A(cumul_r[i]+j,cumul_c[i]+k) = block[i](j,k);
				}
			}
		}
    return A;
	}
	
	
	//! subtraction of another block matrix
	template <typename S>
	block_matrix<T>& operator-=(const block_matrix<S> &A){
		for(size_t i=0; i<block.size(); ++i) block[i].v -= A.block[i].v;
		return *this;
	}
	
	//! addition of another block matrix
	template <typename S>
	block_matrix<T>& operator+=(const block_matrix<S> &A){
		for(size_t i=0; i<block.size(); ++i) block[i].v += A.block[i].v;
		return *this;
	}
	
	//! adding to a matrix, times a constant: Y = Y + z*this
	template <typename S>
	void add(matrix<S> &Y, T z) const
	{
		for(size_t ib=0; ib<block.size(); ++ib){ // loop over sets of rows
			size_t br = block[ib].r;
			size_t bc = block[ib].c;
			size_t R = cumul_r[ib];
			size_t C = cumul_c[ib];
			for(size_t i=0; i<br; ++i){
				for(size_t j=0; j<bc; ++j){
					Y(R+i,C+j) += z*block[ib](i,j);
				}
			}
		}
	}
	
	//! adds a number times the identity matrix
	void add(T d){
		for(size_t i=0; i<block.size(); ++i) block[i].add(d);
	}

	
	//! clears to zero (but keep allocated)
	void clear(){
		for(size_t i=0; i<block.size(); ++i) block[i].clear();
	}
	
	//! computes the square difference with another matrix A
	double diff_sq(const block_matrix<T> &A)
	{
		double z=0.0;
		for(size_t i = 0; i < block.size(); ++i) z += block[i].diff_sq(*A.block[i]);
		return(z);
	}
	
	//! replaces the matrix by its inverse
	void inverse()
	{
		for(size_t i = 0; i < block.size(); ++i) block[i].inverse();
	}
	
	
	//! returns the trace of the matrix
	T trace(){
		T z(0.0);
		for(size_t i = 0; i < block.size(); ++i) z += block[i].trace();
		return z;
	}
	
	//! addition of a number times the unit matrix
	block_matrix<T>& operator+=(const double a){for(size_t i=0; i<block.size(); ++i) block[i].add(a); return *this;}
	block_matrix<T>& operator+=(const Complex a){for(size_t i=0; i<block.size(); ++i) block[i].add(a); return *this;}
	block_matrix<T>& operator-=(const double a){for(size_t i=0; i<block.size(); ++i) block[i].add(-a); return *this;}
	block_matrix<T>& operator-=(const Complex a){for(size_t i=0; i<block.size(); ++i) block[i].add(-a); return *this;}
	
	
	
	//! multiplies a block matrix with a dense matrix : Y = this * X
	template <typename S, typename U>
	void mult_left(const matrix<S> &X, matrix<U> &Y) const
	{
		for(size_t ib=0; ib<block.size(); ++ib){ // loop over sets of rows of Y
			for(size_t i=0; i<block[ib].r; ++i){
				for(size_t j=0; j<X.c; ++j){
					U z(0.0);
					for(size_t k=0; k<block[ib].c; ++k){
						z += block[ib](i,k)*X(k+cumul_c[ib],j);
					}
					Y(i+cumul_r[ib],j) = z;
				}
			}
		}
	}
	
	//! multiplies a dense matrix with a block matrix : Y = X * this
	template <typename S, typename U>
	void mult_right(const matrix<S> &X, matrix<U> &Y) const
	{
		for(size_t ib=0; ib<block.size(); ++ib){ // loop over sets of rows of Y
			for(size_t i=0; i<block[ib].c; ++i){
				for(size_t j=0; j<X.r; ++j){
					U z(0.0);
					for(size_t k=0; k<block[ib].r; ++k){
						z += block[ib](k,i)*X(j,k+cumul_r[ib]);
					}
					Y(j,i+cumul_c[ib]) = z;
				}
			}
		}
	}
	
	
	
	
	//! performs a similarity transformation : A = hermitian(this) * B * this
	template<typename S, typename U>
	void simil(matrix<S>  &A, const matrix<U> &B)
	{
		A.zero();
		for(size_t ij=0; ij<block.size(); ij++){
			size_t J_r = (int)cumul_r[ij];
			size_t J_c = (int)cumul_c[ij];
			for(size_t j = 0; j < block[ij].c; j++){
				for(size_t ii=0; ii<block.size(); ii++){
					size_t I_r = (int)cumul_r[ii];
					size_t I_c = (int)cumul_c[ii];
					for(size_t i = 0; i < block[ii].c; ++i){
						S z = 0.0;
						for(size_t k = 0; k < block[ii].r; ++k){
							S zz = 0.0;
							for(size_t l = 0; l < block[ij].r; ++l){
								zz += block[ij](l,j)*B(k+I_r,l+J_r);
							}
							z += zz*conjugate(block[ii](k,i));
						}
						A(i+I_c,j+J_c) += z;
					}
				}
			}
		}
	}

	
	
	
	//! performs a similarity transformation : A = this * B * hermitian(this)
	template<typename S, typename U>
	void simil_conjugate(matrix<S>  &A, const matrix<U> &B)
	{
		A.zero();
		for(size_t ij=0; ij<block.size(); ij++)
		{
			size_t J_r = (int)cumul_r[ij];
			size_t J_c = (int)cumul_c[ij];
			for(size_t j = 0; j < block[ij].c; j ++)
			{
				for(size_t ii=0; ii<block.size(); ii++)
				{
					size_t I_r = (int)cumul_r[ii];
					size_t I_c = (int)cumul_c[ii];
					for(size_t i = 0; i < block[ii].c; ++i)
					{
						S z = 0.0;
						for(size_t k = 0; k < block[ii].r; ++k)
						{
							S zz = 0.0;
							for(size_t l = 0; l < block[ij].r; ++l)
							{
								zz += conjugate(block[ij](j,l))*B(k+I_r,l+J_r);
							}
							z += zz*conjugate(block[ii](i,k));
						}
						A(i+I_c,j+J_c) += z;
					}
				}
			}
		}
	}

	
	
};

//! writes to a stream
template <typename T>
std::ostream & operator<<(std::ostream &flux, block_matrix<T> &A){
	for(size_t i=0; i<A.block.size(); ++i) flux << A.block[i];
	return flux;
}


#endif
