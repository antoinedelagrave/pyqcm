 /*! \file
 \brief Integration routines, based on the Cubature library (pcubature_v), or home-made (gauss_kronrod).
 They provide resources to integrate over frequency and wavevector, or over wavevector alone.
 */

#include <chrono>
#include "global_parameter.hpp"
#include "integrate.hpp"
#include "vector3D.hpp"
#include "QCM.hpp"
#include "qcm_ED.hpp"
#include "cubature.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#define GLS_INTERVALS 128


double small_scale;
double large_scale;
bool integrand_verb = false;
int integrand_count = 0;


//------------------------------------------------------------------------------
// declarations local to this file

bool sliced = false;

int k_cb_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int low_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int mid_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int high_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int low_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int mid_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
int high_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval);
//------------------------------------------------------------------------------
// variables local to this file

double w_domain; //! domain of the frequency integral
double iw_cutoff;
Complex frequency; //! frequency of a particular k-integral

function<void (Complex w, const int *nv, double I[])> w_integrand;
function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> wk_integrand;
function<void (vector3D<double> &k, const int *nv, double I[])> k_integrand;

double GK_x[8] = { // Gauss-Kronrod rule abscissas
	0.000000000000000,
	0.949107912342759,
	0.741531185599394,
	0.405845151377397,
	0.991455371120813,
	0.864864423359769,
	0.586087235467691,
	0.207784955007898
};

double GK_w[8] = { // Gauss-Kronrod rule weights for the 15-point rule
	0.209482141084728,
	0.063092092629979,
	0.140653259715525,
	0.190350578064785,
	0.022935322010529,
	0.104790010322250,
	0.169004726639267,
	0.204432940075298
};

double GK_gw[4] = { // Gauss-Kronrod rule weights for the 7-point rule
	0.417959183673469,
	0.129484966168870,
	0.279705391489277,
	0.381830050505119
};







/**
 Performs an integral over frequencies and wavevectors
 Uses the CUBA library
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param dim spatial dimension (1 to 3)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properlyl initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::wk_integral(int dim, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
	int ndim = dim+1;
	int cuba_maxpoints;

	auto cubature = global_bool("use_pcubature") ? pcubature_v : hcubature_v;
	if(verb) cout << "Cubature integration (" << (global_bool("use_pcubature") ? "p" : "h") << "-adaptive)" << endl;

	int ncomp = (int)Iv.size();
	vector<double> value(ncomp,0);
	vector<double> err(ncomp,0);

	wk_integrand = f; // sets the file-wide pointer to the integrand

	cuba_maxpoints=10000;
	if(ndim==2){
		cuba_maxpoints=500000;
	}
	else if(ndim==3){
		cuba_maxpoints=10000000;
	}
	else{
		cuba_maxpoints=20000000;
	}

	int fail;
	const double xmin[] = {0,0,0,0};
	const double xmax[] = {1,1,1,1};

  small_scale = global_double("small_scale");
  large_scale = global_double("large_scale");
  double cutoff_scale = global_double("cutoff_scale");
  iw_cutoff = 1.0/cutoff_scale;
  integrand_verb = verb;
	integrand_count = 0;
	//------------------------------------------------------------------------------
	// first region : frequencies below small_scale
	w_domain = small_scale;
	double accur = accuracy*M_PI/w_domain;
	double fac = w_domain*M_1_PI;
	auto t1 = std::chrono::high_resolution_clock::now();
	fail = cubature(ncomp, low_freq_integrand, nullptr, ndim, xmin, xmax, (size_t)cuba_maxpoints, accur, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
	if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
	auto t2 = std::chrono::high_resolution_clock::now();
	double seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "region 1 : " << seconds << " seconds" << endl;

	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;

	//------------------------------------------------------------------------------
	// second region : frequencies between small_scale and large_scale

	w_domain = large_scale-small_scale;
	accur = accuracy*M_PI/w_domain;
	fac = w_domain*M_1_PI;
  to_zero(value);
	t1 = std::chrono::high_resolution_clock::now();
	fail = cubature(ncomp, mid_freq_integrand, nullptr, ndim, xmin, xmax, (size_t)cuba_maxpoints, accur, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
	if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
	t2 = std::chrono::high_resolution_clock::now();
	seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "region 2 : " << seconds << " seconds" << endl;

	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;

	//------------------------------------------------------------------------------
	// third region : inverse frequencies below 1/large_scale

	w_domain = 1.0/large_scale;
	accur = accuracy*M_PI/w_domain;
	fac = w_domain*M_1_PI;
	to_zero(value);
	t1 = std::chrono::high_resolution_clock::now();
	fail = cubature(ncomp, high_freq_integrand, nullptr, ndim, xmin, xmax, (size_t)cuba_maxpoints, accur, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
	if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
	t2 = std::chrono::high_resolution_clock::now();
	seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "region 3 : " << seconds << " seconds" << endl;

	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
}







/**
 Performs an integral over frequencies and wavevectors on a predetermined grid
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param w array of frequencies along the positive imaginary axis (double)
 @param weight array of weights associated with the frequencies
 @param dim spatial dimension (1 to 3)
 @param nk_side number of wavevector on each side of the momentum integration domain (Brillouin zone)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properlyl initialized)
 */
void QCM::wk_integral_grid(const vector<double> &w, const vector<double> &weight, int dim, int nk_side, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv)
{
	cout << "computing integrals from a grid of " << w.size() << " frequencies and " << nk_side << "**" << dim << " k-points" << endl; 
	int nv = (int)Iv.size();
	vector<double> I(nv);
	vector<double> Ik(nv);
	for(int iw=0; iw<w.size(); iw++){
		Complex wc(0, w[iw]);
		to_zero(I);
		double ikside = 1.0/nk_side;
		if(dim==1){
			for(int ikx = 0; ikx < nk_side; ikx++){
				vector3D<double> k(ikx*ikside, 0.0, 0.0);
				f(wc, k, &nv, Ik.data());
				I += Ik;
			}
			I *= ikside;
		}
		else if (dim==2){
			for(int ikx = 0; ikx < nk_side; ikx++){
				for(int iky = 0; iky < nk_side; iky++){
					vector3D<double> k(ikx*ikside, iky*ikside, 0.0);
					f(wc, k, &nv, Ik.data());
					I += Ik;
				}
			}
			I *= ikside*ikside;
		}
		else if (dim==3){
			for(int ikx = 0; ikx < nk_side; ikx++){
				for(int iky = 0; iky < nk_side; iky++){
					for(int ikz = 0; ikz < nk_side; ikz++){
						vector3D<double> k(ikx*ikside, iky*ikside, ikz*ikside);
						f(wc, k, &nv, Ik.data());
						I += Ik;
					}
				}
			}
			I *= ikside*ikside*ikside;
		}
		Iv += I*weight[iw];
	}	
}




/**
 Performs an integral over wavevectors
 Uses the CUBA library
 @param f		function to integrate (may be multi-component)
 @param Iv		value of the integral (adds to previous value: must be properlyl initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::k_integral(int dim, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
  int cuba_maxpoints;
  if(dim==1)       cuba_maxpoints=100000;
  else if(dim==2)  cuba_maxpoints=300000;
  else             cuba_maxpoints=4000000;

  int ncomp = (int)Iv.size();
  vector<double> value(ncomp,0);
  vector<double> err(ncomp,0);
  double accur = accuracy;
  int fail;

  k_integrand = f;

  auto cubature = global_bool("use_pcubature") ? pcubature_v : hcubature_v;
  const double xmin[] = {0,0,0,0};
  const double xmax[] = {1,1,1,1};

  auto t1 = std::chrono::high_resolution_clock::now();
  fail = cubature(ncomp, k_cb_integrand, nullptr, dim, xmin, xmax, (size_t)cuba_maxpoints, accur, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
  auto t2 = std::chrono::high_resolution_clock::now();
  double seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
  if(verb) cout << "Cubature : done in " << seconds << " seconds" << endl;
  for(int i=0; i<ncomp; ++i) Iv[i] += value[i];
}


//******************************************************************************
// THE ROUTINES BELOW THIS LINE ARE FOR INTERNAL USE TO THIS FILE ONLY


/**
 Low-frequency integrand in frequency-wavevector integrals
 @param ndim  dimension of the integral
 @param npts  number of points to evaluate (vectorized batch)
 @param x     point(s) of evaluation, layout x[i*ndim+j]
 @param fdim  number of components of the integrand
 @param fval  values of the integrand (to be calculated), layout fval[i*fdim+k]
 */
int low_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;

  if(ndim==1){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[i]*w_domain);
      vector3D<double> k(0.0,0.0,0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==2){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[2*i]*w_domain);
      vector3D<double> k(x[2*i+1],0.0,0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==3){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[3*i]*w_domain);
      vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==4){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[4*i]*w_domain);
      vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  integrand_count += npts;
  if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;
  return 0;
}


/**
 mid-frequency integrand in frequency-wavevector integrals.
 The integral is performed over inverse frequency between 0 and 1/w_domain
 */
int mid_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;

  if(ndim==1){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[i]*w_domain+small_scale);
      vector3D<double> k(0.0,0.0,0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==2){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[2*i]*w_domain+small_scale);
      vector3D<double> k(x[2*i+1],0.0,0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==3){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[3*i]*w_domain+small_scale);
      vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==4){
    for(int i=0; i<(int)npts; i++){
      Complex w(0,x[4*i]*w_domain+small_scale);
      vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
      wk_integrand(w,k,nv,&fval[fdim*i]);
    }
  }
  integrand_count += npts;
  if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;
  return 0;
}



/**
 high-frequency integrand in frequency-wavevector integrals.
 The relation between iw and w is  iw = w_domain*(1/w - 1/w_cutoff)
 The integral is performed over inverse frequency between 0 and 1/w_domain
 */
int high_freq_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;

  if(ndim==1){
    for(int i=0; i<(int)npts; i++){
      double iw = x[i]*w_domain;
      if(iw < iw_cutoff){
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(0.0, 0.0, 0.0);
        wk_integrand(w,k,nv,&fval[fdim*i]);
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] /= iw*iw;
      }
    }
  }
  else if(ndim==2){
    for(int i=0; i<(int)npts; i++){
      double iw = x[2*i]*w_domain;
      if(iw < iw_cutoff){
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[2*i+1],0.0,0.0);
        wk_integrand(w,k,nv,&fval[fdim*i]);
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] /= iw*iw;
      }
    }
  }
  else if(ndim==3){
    for(int i=0; i<(int)npts; i++){
      double iw = x[3*i]*w_domain;
      if(iw < iw_cutoff){
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
        wk_integrand(w,k,nv,&fval[fdim*i]);
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] /= iw*iw;
      }
    }
  }
  else if(ndim==4){
    for(int i=0; i<(int)npts; i++){
      double iw = x[4*i]*w_domain;
      if(iw < iw_cutoff){
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
        wk_integrand(w,k,nv,&fval[fdim*i]);
        for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] /= iw*iw;
      }
    }
  }
  integrand_count += npts;
  if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;
  return 0;
}




int k_cb_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;

  if(ndim==1){
    for(int i=0; i<(int)npts; i++){
      vector3D<double> k(x[i],0.0,0.0);
      k_integrand(k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==2){
    for(int i=0; i<(int)npts; i++){
      vector3D<double> k(x[2*i],x[2*i+1],0.0);
      k_integrand(k,nv,&fval[fdim*i]);
    }
  }
  else if(ndim==3){
    for(int i=0; i<(int)npts; i++){
      vector3D<double> k(x[3*i],x[3*i+1],x[3*i+2]);
      k_integrand(k,nv,&fval[fdim*i]);
    }
  }
  return 0;
}


/**
 Low-frequency integrand (frequency only, 1D)
 */
int low_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;
  for(int i=0; i<(int)npts; i++){
    Complex w(0,x[i]*w_domain);
    w_integrand(w,nv,&fval[fdim*i]);
  }
  return 0;
}


/**
 mid-frequency integrand (frequency only, 1D).
 */
int mid_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;
  for(int i=0; i<(int)npts; i++){
    Complex w(0,x[i]*w_domain+small_scale);
    w_integrand(w,nv,&fval[fdim*i]);
  }
  return 0;
}


/**
 high-frequency integrand (frequency only, 1D).
 The relation between iw and w is  iw = w_domain*(1/w - 1/w_cutoff)
 */
int high_freq_w_integrand(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  const int nv_int = (int)fdim;
  const int *nv = &nv_int;
  for(int i=0; i<(int)npts; i++){
    double iw = x[i]*w_domain;
    if(iw < iw_cutoff){
      for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] = 0.0;
    }
    else{
      Complex w(0,1.0/iw);
      w_integrand(w,nv,&fval[fdim*i]);
      for(unsigned j=0; j<fdim; ++j) fval[fdim*i+j] /= iw*iw;
    }
  }
  return 0;
}













