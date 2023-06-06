#include <VDVH_kernel.hpp>
#include <immintrin.h>
#include <vector>
#include <cstring>
#include "types.hpp"


/*
How I designed this kernel ?
I used AVX2 processor which contained 16 vector registers (256 bits)
I would like to maximize their use.
https://en.algorithmica.org/hpc/algorithms/matmul/
TODO: I did not use the fact that G is symetric
*/


#define KERNEL_HEIGH_D 4
#define KERNEL_WIDTH_D 3
void kernel(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_D][KERNEL_WIDTH_D]{}; //hold two complex type
  __m256d reg_temp, reg_temp2;
  Complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_D; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm256_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real());
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_D; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = y+2*j + k*L;
        reg_temp2 = _mm256_set_pd(V[index+1],V[index+1],V[index],V[index]);
        //res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  
  // write the results back to G
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D; i++) { //heigh inner loop becase G is column oriented
      _G[2*(x+i + (y+2*j)*L)] += res[i][j][0];
      _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
      _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2];
      _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
    }
  }
}

//https://www.officedaytime.com/simd512e/
void VDVH_matmul(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  //note Mpad is the size of the inner padded matrix
  //G is LpadH * LpadV
  const int LpadH = (L + 2*KERNEL_WIDTH_D-1) / KERNEL_WIDTH_D * KERNEL_WIDTH_D;
  const int LpadV = (L + KERNEL_HEIGH_D-1) / KERNEL_HEIGH_D * KERNEL_HEIGH_D;
  const int Mpad = (M + KERNEL_HEIGH_D-1) / KERNEL_HEIGH_D * KERNEL_HEIGH_D;
  
  //padding the input matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(LpadV * LpadH);
  std::vector<double> _V; _V.resize(LpadV * Mpad);
  std::vector<Complex> _D; _D.resize(Mpad);
  
  std::copy(D.begin(), D.end(), _D.begin());
  for (int i = 0; i < M; i++) {
    std::copy(V.begin() + i*L, V.begin() + (i+1)*L, _V.begin() + i * LpadV);
  }
  
  for (int x = 0; x < LpadV; x += KERNEL_HEIGH_D)
    for (int y = 0; y < LpadH; y += 2*KERNEL_WIDTH_D)
      kernel(_G.data(), _V.data(), _D.data(), x, y, 0, M, Mpad, LpadV);
 
  for (int i = 0; i < L; i++) std::copy(_G.begin()+ i*LpadV, _G.begin()+i*LpadV+L, G.begin()+i*L);
  
  _G.resize(0); _V.resize(0); _D.resize(0);
}




//complex version
void VDVH_matmul(std::vector<Complex> &G, std::vector<Complex> &V, std::vector<Complex> &D, const int L, const int M){};
void kernel(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L) {};


/*
#define KERNEL_HEIGH_C 4
#define KERNEL_WIDTH_C 3
void kernel(Complex* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L)
{
  __m256d res[KERNEL_HEIGHT][KERNEL_WIDTH]{}; //hold two complex type
  __m256d reg_temp, reg_temp2;
  //double VaiD_r, VaiD_i;
  Complex temp;
  
  for(int k=l; k<r; k++) { //k line in V^T, column in V
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGHT; i++) {
      //broadcast lines of V(k,x+i) * D(k) into a register
      //VaiD_r = V[x+i,k] * D[2*k]; //in RAX
      //VaiD_i = V[x+i,k] * D[2*k+1]; //in RAX
      temp = V[x+i,k] * D[k];
      reg_temp = _mm256_set_pd(temp.real(),temp.imag(),temp.real(),temp.imag());
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH; j++) {
        reg_temp2 = _mm256_set_pd(V(y+j,k),V(y+j,k),V(y+j+1,k),V(y+j+1,k)); //V^T(k,y+j) = V(y+j,k)
        res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
      }
    }
  }
  
  // write the results back to C
  for (int i = 0; i < KERNEL_HEIGHT; i++)
    for (int j = 0; j < KERNEL_WIDTH; j++)
      G(i,j) += t[i][j];
}*/
