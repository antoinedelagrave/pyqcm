#ifndef basis3D_h
#define basis3D_h

/**
 * @file basis3D.hpp
 * @brief Declaration of the basis3D struct (3D vector basis with dual/inverse).
 */

#include <iostream>

#include "vector3D.hpp"

//! represent a basis of 3D vectors
/**
 The use of a "working basis" (e.g. Cartesian) is assumed throughout, and all vectors of the basis3D object
 are expressed in terms of this working basis.
 */
struct basis3D
{
	vector<vector3D<double>> e; //!< basis vectors
	matrix<double> M; //!< inverse of the matrix whose rows are the basis vectors
	double vol; //!< volume of the triad (triple product of the basis vectors)
	
	basis3D(){e.reserve(3);}
  basis3D(vector<double> _e);

	void trivial(){
		e.push_back(vector3D<double>(1.0,0.0,0.0));
		e.push_back(vector3D<double>(0.0,1.0,0.0));
		e.push_back(vector3D<double>(0.0,0.0,1.0));
		init();
	}
	void init();
	void inverse();
	vector3D<double> to(const vector3D<double>& v);
	vector3D<double> from(const vector3D<double>& v);
	void dual(basis3D &D);
	
	
	friend std::istream & operator>>(std::istream &flux, basis3D &x);
	friend std::ostream & operator<<(std::ostream &flux, basis3D &x);
};

#endif
