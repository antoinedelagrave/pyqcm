#ifndef VDVH_kernel_h
#define VDVH_kernel_h

#include "types.hpp"
#include <vector>

//double version
void VDVH_matmul(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

//complex version
void VDVH_matmul(std::vector<Complex> &G, std::vector<Complex> &V, std::vector<Complex> &D, const int L, const int M);
void kernel(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L);


//Naive version
template<typename HilbertField>
void VDVH_naive(std::vector<Complex> &G, const std::vector<HilbertField> &V, const std::vector<Complex> &D, const int L, const int M) {  
  for(size_t i=0; i<M; ++i){
    for(size_t a=0; a<L; ++a){
      Complex vai = V[a+i*L]*D[i];
      for(size_t b=0; b<L; ++b){
        G[b + a*L] += vai*conjugate(V[b+i*L]); // original was G(a,b) but was wrong with complex operators
      }
    }
  }
}

#endif
