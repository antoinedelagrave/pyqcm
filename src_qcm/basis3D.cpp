/*! \file
 \brief Methods of the struct \a basis3D
 */


#include "basis3D.hpp"
#include "matrix.hpp"



basis3D::basis3D(vector<double> _e)
{
  trivial();
  if(_e.size()%3 or _e.size()>9) qcm_throw("the basis array should have 0, 3, 6 or 9 elements");
  size_t D = _e.size()/3;
  int j=0;
  for(int i=0; i<D; i++){
    e[i].x = _e[j++];
    e[i].y = _e[j++];
    e[i].z = _e[j++];
  }
  init();
}



//------------------------------------------------------------------------------
/**
 initialises the basis once its vectors have been defined.
 Calculates the matrix M necessary for basis changes, and the volume.
 */
void basis3D::init()
{
	M.set_size(3);
	
	M(0,0) = e[0].x;
	M(1,0) = e[0].y;
	M(2,0) = e[0].z;
	
	M(0,1) = e[1].x;
	M(1,1) = e[1].y;
	M(2,1) = e[1].z;
	
	M(0,2) = e[2].x;
	M(1,2) = e[2].y;
	M(2,2) = e[2].z;
	
	M.inverse();
	
	vol = triple_product(e[0], e[1], e[2]);
}






//------------------------------------------------------------------------------
/**
 Used when the basis vectors provided, instead of being expressed in the working basis, are rather the
 working basis vectors expressed in the basis we are interested in. In other words, the basis needs to
 be inverted.
 The matrix M is read directly (without need to invert) and then a copy of it is inverted in order to read
 the components of the vectors.
 */
void basis3D::inverse()
{
	M.set_size(3);
	
	M(0,0) = e[0].x;
	M(1,0) = e[0].y;
	M(2,0) = e[0].z;
	
	M(0,1) = e[1].x;
	M(1,1) = e[1].y;
	M(2,1) = e[1].z;
	
	M(0,2) = e[2].x;
	M(1,2) = e[2].y;
	M(2,2) = e[2].z;
	
	matrix<double> Q(M);
	Q.inverse();
	
	e[0].x = Q(0,0);
	e[0].y = Q(1,0);
	e[0].z = Q(2,0);
	
	e[1].x = Q(0,1);
	e[1].y = Q(1,1);
	e[1].z = Q(2,1);
	
	e[2].x = Q(0,2);
	e[2].y = Q(1,2);
	e[2].z = Q(2,2);
	
	vol = triple_product(e[0], e[1], e[2]);
}






//------------------------------------------------------------------------------
/**
 from a vector V expressed in the working basis, provides as output its components in the current basis
 @param V vector expressed in the working (Cartesian) basis
 */
vector3D<double> basis3D::to(const vector3D<double>& V)
{
	vector3D<double> W;
	
	W = V;
	W.transform(M);
	return W;
}






//------------------------------------------------------------------------------
//! from a vector V expressed in the current basis, provides as output its components in the working basis
vector3D<double> basis3D::from(const vector3D<double>& V)
{
	vector3D<double> W;
	
	W.x = V.x*e[0].x + V.y*e[1].x + V.z*e[2].x;
	W.y = V.x*e[0].y + V.y*e[1].y + V.z*e[2].y;
	W.z = V.x*e[0].z + V.y*e[1].z + V.z*e[2].z;
	
	return W;
}






//------------------------------------------------------------------------------
//! Returns another basis, 2PI dual of the current basis
void basis3D::dual(basis3D &D)
{
	double L1 = 2.0*M_PI/vol;
	
	D.e[0] = vector_product(e[1],e[2])*L1;
	D.e[1] = vector_product(e[2],e[0])*L1;
	D.e[2] = vector_product(e[0],e[1])*L1;
}






//------------------------------------------------------------------------------
//! reads the vectors of the basis from a stream
std::istream & operator>>(std::istream &flux, basis3D &B){
	flux >> B.e[0] >> B.e[1] >> B.e[2];
	B.init();
	return flux;
}




//------------------------------------------------------------------------------
//! writes the vectors of the basis from a stream
std::ostream & operator<<(std::ostream &flux, basis3D &B){
	flux << B.e[0] << '\n' << B.e[1] << '\n'  << B.e[2] << '\n';
	return flux;
}
