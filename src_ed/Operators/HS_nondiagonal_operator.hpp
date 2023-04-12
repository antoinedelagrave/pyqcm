#ifndef HS_nondiagonal_operator_h
#define HS_nondiagonal_operator_h

#include "HS_Hermitian_operator.hpp"

//! Represents a non-diagonal Hermitian operator in a sector of the Hilbert space
template<typename HS_field>
struct HS_nondiagonal_operator : HS_Hermitian_operator
{
  shared_ptr<ED_mixed_basis> B; //!< basis of the sector
  vector<pair<HS_field, vector<pair<uint32_t, uint32_t>>>> v; //! elements outside of the diagonal
  vector<diagonal_matrix_element> diag_elem; //!< list of diagonal elements
  
  HS_nondiagonal_operator(shared_ptr<model> _the_model, const string &_name, sector _sec);
  ~HS_nondiagonal_operator();
  
  void multiply_add(const vector<double> &x, vector<double> &y, double z);
  void multiply_add(const vector<Complex> &x, vector<Complex> &y, double z);
  void dense_form(matrix<Complex> &h, double z);
  void dense_form(matrix<double> &h, double z);
  void CSR_map(map<index_pair,double> &elem, vector<double> &diag, double z);
  void CSR_map(map<index_pair,Complex> &elem, vector<double> &diag, double z);
  void diag(vector<double> &Y, double z);
  void Triplet_COO_map(vector<matrix_element<double>>& E, double z, bool sym_store);
  void Triplet_COO_map(vector<matrix_element<Complex>>& E, double z, bool sym_store);
  void insert(uint32_t I, uint32_t J, HS_field z);
  void sort_elements();
};


//==============================================================================
// implementation

/**
 Constructor for an operator acting in a given sector
 */
template<typename HS_field>
HS_nondiagonal_operator<HS_field>::HS_nondiagonal_operator(shared_ptr<model> _the_model, const string &_name, sector _sec)
: HS_Hermitian_operator(_the_model, _name, _sec)
{
  B = the_model->provide_basis(sec);
}



/**
 destructor
 */
template<typename HS_field>
HS_nondiagonal_operator<HS_field>::~HS_nondiagonal_operator()
{
  for(auto& w : v) erase(w.second);
  erase(v);
  erase(diag_elem);
}

//! inserts an element 
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::insert(uint32_t I, uint32_t J, HS_field z){
  if(abs(z) < SMALL_VALUE) return;
  else if(J>I) return; // stores only the lower diagonal + diagonal
  else if(J==I) diag_elem.push_back(diagonal_matrix_element(I, real(z)));
  else{
    size_t i;
    for(i=0; i<v.size(); i++){
      if(z == v[i].first){
        v[i].second.push_back({I,J});
        break;
      }
    }
    if(i==v.size()){
      v.push_back({z, vector<pair<uint32_t, uint32_t>>()});
      v.back().second.push_back({I,J});
    }
  }
}

template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::sort_elements()
{
  for(auto& x : v) std::sort(x.second.begin(), x.second.end());
  std::sort(diag_elem.begin(), diag_elem.end());
}


/**
 Multiplies by the operator
 @param y		output vector
 @param x		input vector
 @param z		multiplies the operator
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::multiply_add(const vector<double> &x, vector<double> &y, double z)
{
  for(auto& w : v){
    double z2=real(w.first*z);
    for(auto& e : w.second) y[e.first] += z2*x[e.second];
    for(auto& e : w.second) y[e.second] += z2*x[e.first];
  }
  for(auto& e : diag_elem) y[e.r] += z*e.v*x[e.r];
}
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::multiply_add(const vector<Complex> &x, vector<Complex> &y, double z)
{
  for(auto& w : v){
    HS_field z2=w.first*z;
    for(auto& e : w.second) y[e.first] += z2*x[e.second];
    z2=conjugate(z2);
    for(auto& e : w.second) y[e.second] += z2*x[e.first];
  }
  for(auto& e : diag_elem) y[e.r] += z*e.v*x[e.r];
}




/**
 produces a dense form of the operator
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::dense_form(matrix<double> &h, double z)
{
  if(h.v.size() == 0) return;
  for(auto& w : v){
    double z2=real(w.first*z);
    for(auto& e : w.second) h(e.first, e.second) += z2;
    for(auto& e : w.second) h(e.second, e.first) += z2;
  }
  for(auto& e : diag_elem) h(e.r, e.r) += z*e.v;
}
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::dense_form(matrix<Complex> &h, double z)
{
  if(h.v.size() == 0) return;
  for(auto& w : v){
    HS_field z2=w.first*z;
    for(auto& e : w.second) h(e.first, e.second) += z2;
    z2=conjugate(z2);
    for(auto& e : w.second) h(e.second, e.first) += z2;
  }
  for(auto& e : diag_elem) h(e.r, e.r) += z*e.v;
}





/**
 fills a map, in order to construct the CSR form of the Hamiltonian
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::CSR_map(map<index_pair,double> &E, vector<double> &D, double z)
{
  for(auto& w : v){
    double z2=real(w.first*z);
    for(auto& e : w.second) E[{e.first, e.second}] += z2;
  }
  for(auto& e : diag_elem) D[e.r] += z*e.v;
}
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::CSR_map(map<index_pair,Complex> &E, vector<double> &D, double z)
{
  for(auto& w : v){
    HS_field z2=w.first*z;
    for(auto& e : w.second) E[{e.first, e.second}] += z2;
  }
  for(auto& e : diag_elem) D[e.r] += z*e.v;
}


/**
 populates an array Y with the diagonal elements of the operator
 useful for prepraring the Hamiltonian for the Davidson  method
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::diag(vector<double> &Y, double z)
{
  for(auto& e : diag_elem) Y[e.r] += z*e.v;
}


/**
 populates a vector of matrix_element to build the Hamiltonian
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::Triplet_COO_map(vector<matrix_element<double>>& E, double z, bool sym_store)
{
    //diag element
    for(auto& e : diag_elem) {
        matrix_element<double> T(e.r,e.r,z*e.v);
        E.push_back(T);
    }
    //off-diag element
    for(auto& w : v){
        double z2=real(w.first*z);
        for(auto& e : w.second) {
            matrix_element<double> T(e.first,e.second,z2);
            E.push_back(T);
        }
        if(sym_store){
            for(auto& e : w.second) {
                matrix_element<double> T(e.second,e.first,z2);
                E.push_back(T);
            }
        }
    }
}

/**
 populates a vector of matrix_element to build the Hamiltonian
 */
template<typename HS_field>
void HS_nondiagonal_operator<HS_field>::Triplet_COO_map(vector<matrix_element<Complex>>& E, double z, bool sym_store)
{
    //diag element
    for(auto& e : diag_elem) {
        matrix_element<Complex> T(e.r,e.r,z*e.v);
        E.push_back(T);
    }
    //off-diag element
    for(auto& w : v){
        Complex z2=w.first*z;
        for(auto& e : w.second) {
            matrix_element<Complex> T(e.first,e.second,z2);
            E.push_back(T);
        }
        if(sym_store){
            z2 = conjugate(z2);
            for(auto& e : w.second) {
                matrix_element<Complex> T(e.second,e.first,z2);
                E.push_back(T);
            }
        }
    }
}


#endif
