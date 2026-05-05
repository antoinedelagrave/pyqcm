#ifndef Q_matrix_set_h
#define Q_matrix_set_h

/**
 * @file Q_matrix_set.hpp
 * @brief Definition of the Q_matrix_set template (Lehmann form of the full Green function).
 */

#include "Green_function_set.hpp"
#include "Q_matrix.hpp"

//!  Used to store the Lehman representation of a Green function
template<typename HilbertField>
struct Q_matrix_set : Green_function_set
{
  vector<Q_matrix<HilbertField>> q; //!< QMatrices

  Q_matrix_set(const Q_matrix_set &Q);
  Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing);
  Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>>& w, const vector<matrix<HilbertField>> &_q);
  Q_matrix<HilbertField> consolidated_qmatrix();
  void append(Q_matrix_set &Q);
  void check_norm(double threshold, double norm);
  void streamline(bool verb=false);
  void merge(vector<multimap<double, vector<HilbertField>>>& M);

  // realizations of base class virtual methods
  void Green_function(const Complex &z, block_matrix<Complex> &G) override;
  void integrated_Green_function(block_matrix<Complex> &M) override;
  void write_hdf5(H5::Group& grp) override;
  void read_hdf5(H5::Group& grp) override;
};


//==============================================================================
// implementation


/**
 Constructor
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing) : Green_function_set(_group, mixing)
{
  q.assign(group->g, Q_matrix<HilbertField>());
}






/**
 deep copy Constructor
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(const Q_matrix_set &Q)
{
  L = Q.L;
  group = Q.group;
  q = Q.q;
}






/**
 constructor from data from a previous computation
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>>& _e, const vector<matrix<HilbertField>> &_q) : Green_function_set(_group, mixing)
{
  group = _group;
  q.resize(group->g);
  for(int i=0; i<group->g; i++) q[i] = Q_matrix<HilbertField>(_e[i], _q[i]);
}


/**
 partial Green function evaluation
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::Green_function(const Complex &z, block_matrix<Complex> &G)
{
  for(size_t r=0; r<q.size(); ++r){
    if(q[r].L == 0) continue;
    q[r].Green_function(z,G.block[r]);
  }
}


/**
 partial integrated Green function evaluation
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::integrated_Green_function(block_matrix<Complex> &G)
{
  for(size_t r=0; r<q.size(); ++r){
    if(q[r].L == 0) continue;
    q[r].integrated_Green_function(G.block[r]);
  }
}


/**
 Appends to the Q_matrix another one (increasing the number of rows for the same number of columns
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::append(Q_matrix_set &Q)
{
  QCM_ASSERT(Q.q.size() == q.size());
  for(size_t r=0; r<q.size(); ++r) q[r]->append(*Q.q[r]);
}


/**
 Checking normalization
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::check_norm(double threshold, double norm)
{
  for(size_t r=0; r<q.size(); ++r) q[r]->check_norm(threshold, norm);
}


/**
 Eliminating the small contributions
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::streamline(bool verb)
{
  for(size_t r=0; r<q.size(); ++r){
    q[r].streamline(verb);
  }
}


/**
 Puts together the elements into a single Q_matrix, in the cluster site basis (i.e. not symmetric operators)
 */
template<typename HilbertField>
Q_matrix<HilbertField> Q_matrix_set<HilbertField>::consolidated_qmatrix()
{
  size_t i_cum = 0, i_tot = 0;

  for(size_t r=0; r<q.size(); ++r) i_tot += q[r].M;
  Q_matrix<HilbertField> Q(L, i_tot);

  vector<HilbertField> Y(L);
  for(size_t r=0; r<q.size(); ++r){
    vector<HilbertField> X(q[r].L);
    size_t imax = q[r].M;
    for(size_t i=0; i<imax; ++i){
      q[r].v.extract_column(i,X);
      group->to_site_basis(r, X, Y, n_mixed);
      Q.v.insert_column(i+i_cum, Y);
      Q.e[i+i_cum] = q[r].e[i];
    }
    i_cum += imax;
  }
  return Q;
}


/**
Merging into a multimap
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::merge(vector<multimap<double, vector<HilbertField>>>& M)
{
  for(size_t r=0; r<q.size(); ++r){
    for(int j=0; j< q[r].M; j++){
      vector<HilbertField> v(q[r].L);
      for(int k=0; k<q[r].L; k++) v[k] = q[r].v(k,j);
      M[r].insert({q[r].e[j], v});
    }
  }
}

/**
 Writes the Q_matrix_set to an HDF5 group.
 Layout: attribute "nblocks"; sub-groups "block_0" … "block_{g-1}".
*/
template<typename HilbertField>
void Q_matrix_set<HilbertField>::write_hdf5(H5::Group& grp)
{
  h5_write_attr(grp, "nblocks", (int)q.size());
  for(size_t r = 0; r < q.size(); ++r){
    H5::Group bg = grp.createGroup("block_" + to_string(r));
    q[r].write_hdf5(bg);
  }
}


/**
 Reads the Q_matrix_set from an HDF5 group written by write_hdf5.
*/
template<typename HilbertField>
void Q_matrix_set<HilbertField>::read_hdf5(H5::Group& grp)
{
  int nblocks = h5_read_attr_int(grp, "nblocks");
  q.resize(nblocks);
  for(int r = 0; r < nblocks; ++r){
    H5::Group bg = grp.openGroup("block_" + to_string(r));
    q[r].read_hdf5(bg);
  }
}

#endif /* Q_matrix_set_h */
