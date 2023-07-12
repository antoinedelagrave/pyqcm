#ifndef symmetry_group_h
#define symmetry_group_h

#include <iostream>
#include <vector>
#include <typeinfo>
#include <cstdint>

#include "qcm_ED.hpp"
#include "matrix.hpp"
#include "binary_state.hpp"
#include "sector.hpp"
#include "parser.hpp"
#include "index.hpp"
#include "symmetric_orbital.hpp"
#include "block_matrix.hpp"
#include "index_pair.hpp"

using namespace std;

#define MAX_GENERATORS 4 //!< maximum number of commuting generators.

struct rep_map
{
  binary_state b;
  int phase;
  int length;
};

//! contains the symmetry operations of the cluster and methods to take advantage of them
/**
 The symmetry group is supposed Abelian.
 It is defined by a small set of generators, each being a "signed permutation": the combination of a permutation of
 sites with possibly sign changes of the correponding annihilation operators
 
 each element of the group is thus defined by a permutation 'e' and a sign mask 'sign_mask' (the set bits of the mask
 are associated with the orbitals that change sign under the element).
 
 The generators are defined by a permutation 'generator' and a sign mask 'generator_sign_mask'
 */
struct symmetry_group
{
  bool has_complex_irrep;
  bool bath_irrep; //!< if true, the bath orbitals belong to specific representations
  int N; //!< total number of sites (i.e. including bath sites)
	int n_sites; //!< number of orbitals used in Green functions (real sites)
	int g; //!< number of group elements
  int L; 
	vector<vector<int>> generator; //!< generators
	vector<vector<int>> e; //!< group elements (cluster representation)
	matrix<int> phi; //!< phases of the group characters (if multiplied by 2\pi/g)
	matrix<Complex> chi; //!< character table (transformation matrix)
	matrix<Complex> S; //!< transforms symmetric orbitals into site orbitals
  matrix<double> S_real; //!< transforms symmetric orbitals into site orbitals (real version)
  vector<bool> complex_irrep; //!< true if the irrep requires complex numbers (has at least one complex-valued character)
  vector<int> site_irrep_dim; //! number of cluster symmetric orbitals in each irrep
	vector<int> site_irrep_dim_cumul; //! number of cluster symmetric orbitals before each irrep
	matrix<int> tensor; //!< matrix of the tensor products
	vector<int> conjugate; //!< label of conjugate irrep
  vector<int> phaseR;
  vector<Complex> phaseC;


  symmetry_group(int _N, int _n_sites, const vector<vector<int>> &gen, bool bath_irrep);
  bool is_valid_element(const vector<int>& v, bool full=false);
  bool is_identity(const vector<int>& v, bool full=false);
  vector<int> product(const vector<int>& x, const vector<int>& y);
  vector<int> inverse(const vector<int>& x);
  vector<int> identity();
  int sign(const vector<int>& x);
  uint32_t apply_single(const vector<int>& x, const uint32_t b);
  pair<uint32_t, int> apply(const vector<int>& x, const uint32_t b);
  template<typename T>
  vector<T> apply_to_vector(const vector<int>& v, const vector<T>& x);  
  pair<int, int> map_orbital(const vector<int>& v, int a);
  bool diff(const vector<int>& x, const vector<int>& y, bool full=false);
  void build();
  pair<binary_state, int> apply(const int &elem, const binary_state &b);
  pair<binary_state, int> apply(const int &elem, const int &irrep, const binary_state &b);
  rep_map Representative(const binary_state &b, int irrep);
  void to_site_basis(int r, vector<Complex> &x, vector<Complex> &y, int n_mixed);
  void to_site_basis(int r, vector<double> &x, vector<double> &y, int n_mixed);
  void to_site_basis(block_matrix<Complex> &B, matrix<Complex> &G, int n_mixed);
  bool sector_is_valid(const sector &sec);
  sector shift_sector(const sector &sec, int pm, int spin, int _irrep);
  bool operator!=(const symmetry_group &y);

  template<typename HilbertField>
  inline HilbertField phaseX(int i);

  template<typename HilbertField>
  void check_invariance(const map<index_pair,HilbertField> &elements, const string& name, bool interaction);
  vector<vector<symmetric_orbital>> build_symmetric_orbitals(int mixing);
};


std::ostream & operator<<(std::ostream &s, const symmetry_group &x);


// implementation


template<typename HilbertField>
void symmetry_group::check_invariance(const map<index_pair,HilbertField> &elements, const string& name, bool interaction)
{
  if(g==1) return;
  
  for(int i=0; i<generator.size(); i++){
    for(auto &x : elements){
      auto A = map_orbital(generator[i], x.first.r);
      A.second *= -1; // sign because c_a^\dg is a creation operator
      auto B = map_orbital(generator[i], x.first.c);
      int f = (A.second + B.second + 4*g)%(2*g); // positive modulo
      index_pair xp(A.first, B.first);
      auto y = elements.find(xp);
      if(y == elements.end() and interaction) y = elements.find(index_pair(B.first, A.first));
      if(y == elements.end()){
        ostringstream sout;
        sout << "Term " << x.first << " of parameter " << name << " is incompatible with symmetry generator " << i+1 << ", which maps it into " << xp;
        qcm_ED_throw(sout.str());
      }
      Complex ratio = x.second*phaseC[f]/y->second;
      if(abs(ratio-1.0) > 1e-8){
        ostringstream sout;
        sout << "Term " << x.first << " of parameter " << name << " is incompatible with symmetry generator " << i+1 << ", which maps it into " << xp << " but with value " << phaseC[f]*y->second << " instead of " << x.second;
        qcm_ED_throw(sout.str());
      }
    }
  }
}

template<>
inline double symmetry_group::phaseX<double>(int i){return phaseR[i];}

template<>
inline Complex symmetry_group::phaseX<Complex>(int i){return phaseC[i];}

#endif






