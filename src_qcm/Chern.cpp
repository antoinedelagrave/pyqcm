#include <iostream>
#include <fstream>

#define MAX_K_SIDE 5000

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lattice_model_instance.hpp"
#include "integrate.hpp"
#include "parser.hpp"

bool recursive=false;
const double threshold = 0.1*M_PI;
vector3D<double> increment(double d, int no, int dir);

template <typename T>
void gauge_field(matrix<T> &A, matrix<T> &B, vector<T> &u){
	for(int i=0; i<A.c; ++i){
		u[i] = 0.0;
		for(int j=0; j<A.r; ++j){
			u[i] += conjugate(A(j,i))*B(j,i);
		}
	}
}

/**
Computes the eigenvalues and eigenvectors of a Green function at zero frequency
@param G [in] Green function object
@param k0 [in] wave vector
@param e [out] eigenvalues
@param U [out] matrix whose columns are the eigenvectors
@param opt [in] option:  
  opt = 0 : physical basis (periodized Green function)
  opt = 1 : dual_basis (periodized Green function)
  opt = 2 : non periodized Green function
 */
void lattice_model_instance::Green_eigensystem(Green_function &G, const vector3D<double> &k0, vector<double> &e, matrix<Complex> &U, int opt)
{
  vector3D<double> k = k0;
  if(opt&1) k = model->superdual.to(model->dual.from(k));
  else if(opt&2){
    // do nothing : leave k as is
  }
  else k = model->superdual.to(model->physdual.from(k));
  
  Green_function_k K(G,k);
  if(opt&2){
    set_Gcpt(K);
    K.Gcpt.eigensystem(e, U);
  }
  else{
    periodized_Green_function(K);
    K.g.eigensystem(e, U);
  }
}

/*
returns an increment vector 
*/
vector3D<double> increment(double d, int no, int dir)
{
  switch(dir){
    case  3:
      switch(no){
        case 0: return { d, 0.0, 0.0};
        case 1: return { 0.0, d, 0.0};
        case 2: return {-d, 0.0, 0.0};
      }
    case  -3:
      switch(no){
        case 0: return { d, 0.0, 0.0};
        case 1: return { 0.0,-d, 0.0};
        case 2: return {-d, 0.0, 0.0};
      }
    case  2:
      switch(no){
        case 0: return {-d, 0.0, 0.0};
        case 1: return { 0.0, 0.0, d};
        case 2: return { d, 0.0, 0.0};
      }
    case  -2:
      switch(no){
        case 0: return {-d, 0.0, 0.0};
        case 1: return { 0.0, 0.0, -d};
        case 2: return { d, 0.0, 0.0};
      }
    case  1:
      switch(no){
        case 0: return {0.0, d, 0.0};
        case 1: return {0.0, 0.0, d};
        case 2: return {0.0, -d, 0.0};
      }
    case  -1:
      switch(no){
        case 0: return {0.0, d, 0.0};
        case 1: return {0.0, 0.0,-d};
        case 2: return {0.0,-d, 0.0};
      }
  }
  return {0.0, 0.0, 0.0};
}




/**
Computes the integral of the Berry connexion along a closed contour, specified as a list of wavevectors
@param k [in] array of wavevectors
@param orb [in] orb=0 means a sum over all lattice orbitals, otherwise the lattice orbital specified
@param spin_down [in] true if the spin down sector is considered
*/
double lattice_model_instance::Berry_flux(const vector<vector3D<double>> &k, int orb, bool spin_down)
{
  check_signals();
  double eta = global_double("eta");
  Green_function G = cluster_Green_function(Complex(0, eta), false, spin_down);

  size_t ng = model->dim_reduced_GF;
  if(orb > ng) qcm_throw("the orbital number specified in Berry curvature computations is beyond the number of orbitals in the lattice model");
	
  matrix<Complex> U(ng), U0(ng), U1(ng);
	vector<Complex> u(ng);
	vector<double> e(ng), e0(ng), e1(ng);

  // loop over the wavevectors of the path
  Green_eigensystem(G, k[0], e1, U1, 0); for(auto& x : e0) x = -1.0/x;
  U0 = U1;
  e0 = e1;
  Complex z(1.0);
  for(int i=1; i<k.size(); i++){
    Green_eigensystem(G, k[i], e, U, 0); for(auto& x : e) x = -1.0/x;
    gauge_field(U0, U, u);
    if (orb==0){
      for(int b=0; b<ng; b++) if(e[b] < 0.0) z *= u[b];
    }
    else z *= u[orb-1];
    U0 = U;
    e0 = e;
  }
  gauge_field(U0, U1, u);
  if (orb==0){
      for(int b=0; b<ng; b++) if(e0[b]+e[b] < 0.0) z *= u[b];
  }
  else z *= u[orb-1];
  double flux = -arg(z)/(2*M_PI);

  if(model->mixing == HS_mixing::normal) flux *= 2.0;
  if(model->mixing == HS_mixing::up_down and spin_down == false) flux += Berry_flux(k, true, 0);
  return flux;
}




/**
Computes the flux of the Berry curvature through a square plaquette with a corner at k1 and wavevector increments deltax and deltay.
@param G [in] Green function objet
@param k1 [in] wavevector (corner of the plaquette)
@param deltax [in] wavevector step in direction 1
@param deltay [in] wavevector step in direction 2
@param opt [in] option, like in function Green_eigensystem above
@param dir [in] direction of perpendicular to plaquette (x=1, y=2, z=3)
@param orb [in] orb=0 means a sum over all lattice orbitals, otherwise the orbital specified
*/
double lattice_model_instance::Berry_plaquette(Green_function &G, const vector3D<double> &k1, const double deltax, const double deltay, const int opt, int dir, int orb)
{
  check_signals();

  size_t ng;
  if(opt&2) ng = model->dim_GF;
  else ng = model->dim_reduced_GF;
  if(orb > ng) qcm_throw("the orbital label specified in Berry curvature computations is beyond the number of orbitals in the lattice model");

	matrix<Complex> U1(ng), U2(ng), U3(ng), U4(ng);
	vector<Complex> u1(ng), u2(ng), u3(ng), u4(ng);
	vector<double> e1(ng), e2(ng), e3(ng), e4(ng);
  vector3D<double> k(k1);

  Green_eigensystem(G, k, e1, U1, opt);
  for(auto& x : e1) x = -1.0/x;

  k += increment(deltax, 0, dir);
  Green_eigensystem(G, k, e2, U2, opt);
  for(auto& x : e2) x = -1.0/x;
  
  k += increment(deltay, 1, dir);
  Green_eigensystem(G, k, e3, U3, opt);
  for(auto& x : e3) x = -1.0/x;

  k += increment(deltax, 2, dir);
  Green_eigensystem(G, k, e4, U4, opt);
  for(auto& x : e4) x = -1.0/x;

	gauge_field(U1, U2, u1);
	gauge_field(U2, U3, u2);
	gauge_field(U3, U4, u3);
	gauge_field(U4, U1, u4);
	Complex z(1.0);
  if (orb==0){
    for(int b=0; b<ng; b++){
      if(e1[b]+e2[b]+e3[b]+e4[b] < 0.0) z *= u1[b]*u2[b]*u3[b]*u4[b];
    }
  }
  else{
    int b = orb-1;
    z *= u1[b]*u2[b]*u3[b]*u4[b];
  }
  double phase = arg(z);
  if(abs(phase) > threshold && recursive){
    phase = 0.0;
    double dx = deltax/2;
    double dy = deltay/2;
    k = k1;
    phase += Berry_plaquette(G, k, dx, dy, opt, dir, orb);
    k += increment(dx, 0, dir);
    phase += Berry_plaquette(G, k, dx, dy, opt, dir, orb);
    k += increment(dy, 1, dir);
    phase += Berry_plaquette(G, k, dx, dy, opt, dir, orb);
    k += increment(dx, 2, dir);
    phase += Berry_plaquette(G, k, dx, dy, opt, dir, orb);
  }


	return phase;
}




/**
 Plots the interacting Berry curvature within the Brillouin zone
 See:
 Z. Wang and S.C. Zhang, Physical Review X 2 (2012), no. 3, 031008. AND
 Takahiro Fukui, Yasuhiro Hatsugai, and Hiroshi Suzuki, Journal of the Physical Society of Japan 74 (2005), no. 6, 1674– 1677.
 
 @param k1 [in] lower left corner of the wavevector domain
 @param k2 [in] upper right of the wavevector domain
 @param nk [in] number of wavevectors on the side of the Brillouin zone (nk x nk grid) 
 @param orb [in] orbital label (0 means a sum over all lattice orbitals)
 @param rec [in] if true, refines reursively the grid if needed
 @param dir [in] direction of perpendicular to plaquette (x=1, y=2, z=3)
 */
vector<double> lattice_model_instance::Berry_curvature(vector3D<double>& k1, vector3D<double>& k2, int nk, int orb, bool rec, int dir)
{
  recursive = rec;

  if(model->spatial_dimension < 2) qcm_throw("'Berry_curvature' can only be applied to a 2D or 3D Brillouin zone!");
  if(nk > MAX_K_SIDE) qcm_throw("The wavevector grid has too many points. nk should be <= "+to_string<int>(MAX_K_SIDE));
  if(nk < 10) qcm_throw("The wavevector grid has too few points. nk should be >= 10");
  
  int opt=0;
  if(global_bool("dual_basis")) opt += 1;
  if(global_char("periodization") == 'N') opt += 2;

	
  vector<double> B(nk*nk, 0.0);
  vector3D<double> delta1;
  vector3D<double> delta2;
  double d1, d2;
  if(dir==1){
    if(fabs(k1.x-k2.x)>1e-10) qcm_throw("in Berry curvature, k1 and k2 must have the same 3rd component");
    d1 = (k2.y-k1.y)/nk;
    d2 = (k2.z-k1.z)/nk;
    delta1.y = d1;
    delta2.z = d2;
  }
  else if(dir==2){
    if(fabs(k1.y-k2.y)>1e-10) qcm_throw("in Berry curvature, k1 and k2 must have the same 3rd component");
    d1 = (k2.x-k1.x)/nk;
    d2 = (k2.z-k1.z)/nk;
    delta1.x = d1;
    delta2.z = d2;
  }
  else{
    if(fabs(k1.z-k2.z)>1e-10) qcm_throw("in Berry curvature, k1 and k2 must have the same 3rd component");
    d1 = (k2.x-k1.x)/nk;
    d2 = (k2.y-k1.y)/nk;
    delta1.x = d1;
    delta2.y = d2;
  }

  double eta = global_double("eta");
  Green_function G = cluster_Green_function(Complex(0, eta), false, false);
		
	// loop over the plaquettes
  // #pragma omp parallel for
	for(int j=0; j<nk; j++){
		for(int i=0; i<nk; i++){
			B[i+nk*j] = Berry_plaquette(G, k1 + i*delta1 + j*delta2, d1, d2, opt, dir, orb);
		}
	}
	
	if(model->mixing == HS_mixing::up_down){
		G = cluster_Green_function(Complex(0, 1e-8), false, true);
    for(int j=0; j<nk; j++){
      for(int i=0; i<nk; i++){
        B[i+nk*j] = Berry_plaquette(G, k1 + i*delta1 + j*delta2, d1, d2, opt, dir, orb);
      }
    }
  }
  else if(model->mixing == HS_mixing::normal) B *= 2.0;
  // else if(model->mixing == HS_mixing::full) B *= 0.5;
  B *= -1.0/(2*M_PI*d1*d2);
 
  return B;
}





/**
 Computes the contribution of a plaque to the Chern number
 @param k [in] node wavevector
 @param a [in] half side of the cube
 @param nk [in] number of wavevectors on the side of the Brillouin zone 
 @param orb [in] orbital label (0 means a sum over all lattice orbitals)
 @param rec [in] if true, refines reursively the grid if needed
 @param dir [in] direction of perpendicular to plaquette (x=1, y=2, z=3)
 */
double lattice_model_instance::monopole_part(vector3D<double>& k, double a, int nk, int orb, bool rec, int dir, bool spin_down)
{
  recursive = rec;

  int opt=0;

  double charge = 0.0;
  double d = 2*a/nk;
  
  double eta = global_double("eta");
  // double eta = 1e-6;
  Green_function G = cluster_Green_function(Complex(0, eta), false, spin_down);
		
	// loop over the plaquettes
  // #pragma omp parallel for
	for(int j=0; j<nk; j++){
		for(int i=0; i<nk; i++){
      vector3D<double>kb = k;
      switch(dir){
        case 1 :
          kb.x += a;
          kb.y += i*d - a;
          kb.z += j*d - a;
          break;
        case -1 :
          kb.x -= a;
          kb.y -= i*d - a;
          kb.z -= j*d - a;
          break;
        case 2 :
          kb.y += a;
          kb.z += i*d - a;
          kb.x += j*d - a;
          break;
        case -2 :
          kb.y -= a;
          kb.z -= i*d - a;
          kb.x -= j*d - a;
          break;
        case 3 :
          kb.z += a;
          kb.x += i*d - a;
          kb.y += j*d - a;
          break;
        case -3 :
          kb.z -= a;
          kb.x -= i*d - a;
          kb.y -= j*d - a;
          break;
      }
			charge += Berry_plaquette(G, kb, d, d, opt, dir, orb);
		}
	} 
  return charge;
}



/**
 Computes the Chern number integral on the surface of a cube
 @param k [in] wavevector at the center of the cube
 @param a [in] half side of the cube
 @param nk [in] number of wavevectors on the side of the Brillouin zone 
 @param orb [in] orbital label (0 means a sum over all lattice orbitals)
 @param rec [in] if true, refines reursively the grid if needed
 */
double lattice_model_instance::monopole(vector3D<double>& k, double a, int nk, int orb, bool rec)
{
  recursive = rec;

  if(model->spatial_dimension != 3) qcm_throw("'monopole' can only be applied to a 3D model!");
  if(nk > 1000) qcm_throw("The wavevector grid has too many points. nk should be <= 1000");
  if(nk < 2) qcm_throw("The wavevector grid has too few points. nk should be >= 2");
  
	double charge = 0.0;

  charge += monopole_part(k, a, nk, orb, rec, 1, false);
  charge += monopole_part(k, a, nk, orb, rec,-1, false);
  charge += monopole_part(k, a, nk, orb, rec, 2, false);
  charge += monopole_part(k, a, nk, orb, rec,-2, false);
  charge += monopole_part(k, a, nk, orb, rec, 3, false);
  charge += monopole_part(k, a, nk, orb, rec,-3, false);
 
  if(model->mixing == HS_mixing::up_down){
    charge += monopole_part(k, a, nk, orb, rec, 1, true);
    charge += monopole_part(k, a, nk, orb, rec,-1, true);
    charge += monopole_part(k, a, nk, orb, rec, 2, true);
    charge += monopole_part(k, a, nk, orb, rec,-2, true);
    charge += monopole_part(k, a, nk, orb, rec, 3, true);
    charge += monopole_part(k, a, nk, orb, rec,-3, true);
  }
  else if(model->mixing == HS_mixing::normal) charge *= 2.0;
  charge *= -1.0/(2*M_PI);
  return charge;
}
