/*! \file
 \brief Methods of the classes \a vector
 */


#include "vector_num.hpp"


/**
 returns the sum of all components
 @param x input vector
 */
double sum(vector<double> &x) {
  double z = 0.0;
  for (size_t i = 0; i < x.size(); ++i) z += x[i];
  return (z);
}

/**
 returns the sum of the negative components
 @param x input vector
 */
double sum_negative(vector<double> &x) {
  double z = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    if (x[i] < 0.0) z += x[i];
  return (z);
}


/**
 performs the multiply-add operation y += a*x
 @param a scalar multiplier
 @param x input vector to scale and add
 @param y [in,out] accumulation vector
 */
void mult_add(double a, const vector<double> &x, vector<double> &y) { cblas_daxpy((int)y.size(), a, x.data(), 1, y.data(), 1); }



/**
 performs the multiply-add operation y += a*x
 Complex version
 @param a scalar multiplier
 @param x input vector to scale and add
 @param y [in,out] accumulation vector
 */
void mult_add(Complex a, const vector<Complex> &x, vector<Complex> &y) {
  cblas_zaxpy((int)x.size(), (void *)&a, (void *)x.data(), 1, (void *)y.data(), 1);
}



/**
 dot product (double version)
 @param x first vector
 @param y second vector
 */
double operator*(const vector<double> &x, const vector<double> &y) {
  double z = 0.0;
  z = cblas_ddot((int)x.size(), x.data(), 1, y.data(), 1);
  return (z);
}


/**
 dot product (complex version)
 @param x first vector
 @param y second vector
 */
Complex operator*(const vector<Complex> &x, const vector<Complex> &y) {
  Complex z = 0.0;
  cblas_zdotc_sub((int)x.size(), x.data(), 1, y.data(), 1, &z);
  return (z);
}


/**
 norm of a vector
 @param x input vector
 */
double norm(const vector<double> &x) { return cblas_dnrm2((int)x.size(), x.data(), 1); }
double norm(const vector<Complex> &x) { return cblas_dznrm2((int)x.size(), x.data(), 1); }


/**
 square norm of a vector
 @param x input vector
 */
double norm2(const vector<double> &x) {
  double z = cblas_dnrm2((int)x.size(), x.data(), 1);
  return (z * z);
}
double norm2(const vector<Complex> &x) {
  double z = cblas_dznrm2((int)x.size(), x.data(), 1);
  return (z * z);
}


/**
 multiplies the vector by a constant: x *= a
 @param x [in,out] vector to scale
 @param c scalar multiplier
 */
void operator*=(vector<double> &x, const double &c) { cblas_dscal((int)x.size(), c, x.data(), 1); }
void operator*=(vector<Complex> &x, const Complex &c) { cblas_zscal((int)x.size(), &c, x.data(), 1); }


/**
 Adds to vector random components between -1 and 1, then normalizes
 @param x [in,out] vector to randomize and normalize
 @param ran normal distribution object used to generate random values
 */
bool random(vector<double> &x, std::normal_distribution<double> &ran) {
  std::default_random_engine generator((unsigned)global_int("seed"));

  for (size_t i = 0; i < x.size(); ++i) x[i] += ran(generator);
  return normalize<double>(x);
}

bool random(vector<Complex> &x, std::normal_distribution<double> &ran) {
  std::default_random_engine generator((unsigned)global_int("seed"));

  for (size_t i = 0; i < x.size(); ++i) x[i] += Complex(ran(generator), ran(generator));
  return normalize<Complex>(x);
}



/**
 changes the components of x so as to make the first non-negligible component real and positive
 */
void fix_phase(vector<Complex> &x) {
  Complex fac(1.0);
  for (size_t i = 0; i < x.size(); i++) {
    if (abs(x[i]) > 1e-10) {
      fac = x[i];
      break;
    }
  }
  fac = abs(fac) / fac;
  x *= fac;
  for (size_t i = 0; i < x.size(); i++) x[i] = chop(x[i]);
}



/**
 Finds the largest component in absolute value
 */
double max_abs(vector<double> &x) {
  double max = 0.0;
  for (size_t i = 0; i < x.size(); i++)
    if (fabs(x[i]) > max) max = fabs(x[i]);
  return max;
}



/**
 Finds the smallest component in absolute value
 */
double min_abs(vector<double> &x) {
  double min = fabs(x[0]);
  for (size_t i = 1; i < x.size(); i++)
    if (fabs(x[i]) < min) min = fabs(x[i]);
  return min;
}



/*
 transforms a real vector into a complex one
 **/
vector<Complex> to_complex(const vector<double> &x) {
  vector<Complex> xc(x.size());
  for (size_t i = 0; i < x.size(); i++) xc[i] = x[i];
  return xc;
}


/**
 returns the estimated average and standard deviation of a sample x by the jack-knife method
 */
pair<double,double> jackknife(const vector<double>& x)
{
  int N = x.size();
  vector<double> xb(N); // subsample average
  for(int i=0; i<N; i++){ // loop over subsamples
    double sx=0.0; // computing the subsample average
    for(int j=0; j<i; j++) sx += x[j];
    for(int j=i+1; j<N; j++) sx += x[j];
    xb[i] = sx/(N-1);
  }

  // computing the average 'ave' and variance 'var' of xb
  double ave=0.0;
  for(int i=0; i<N; i++) ave += xb[i];
  ave /= N;

  double var=0.0;
  for(int i=0; i<N; i++){
    xb[i]  -=  ave;
    var += xb[i]*xb[i];
  }
  var *= 1.0*(N-1)/N;
  return {ave, sqrt(var)}; // average and standard deviation
}