#ifndef HS_interaction_operator_h
#define HS_interaction_operator_h

#include "HS_Hermitian_operator.hpp"

//! Represents an interaction operator (density-density) in a sector of the Hilbert space
struct HS_interaction_operator : HS_Hermitian_operator
{
  vector<double> elem;
  shared_ptr<ED_basis> B;
  
  HS_interaction_operator(shared_ptr<model> _the_model, const string &_name, sector _sec, const vector<matrix_element<double>> &elements);

  void multiply_add(const vector<double> &x, vector<double> &y, double z);
  void multiply_add(const vector<Complex> &x, vector<Complex> &y, double z);
  void dense_form(matrix<double> &h, double z);
  void dense_form(matrix<Complex> &h, double z);
  void CSR_map(map<index_pair,double> &E, vector<double> &D, double z);
  void CSR_map(map<index_pair,Complex> &E, vector<double> &D, double z);
  void diag(vector<double> &Y, double z);
  void Triplet_COO_map(vector<matrix_element<double>>& E, double z, bool sym_store);
  void Triplet_COO_map(vector<matrix_element<Complex>>& E, double z, bool sym_store);
};


//==============================================================================
// implementation

/**
 constructor
 */
HS_interaction_operator::HS_interaction_operator(shared_ptr<model> _the_model, const string &_name, sector _sec, const vector<matrix_element<double>> &elements)
: HS_Hermitian_operator(_the_model, _name, _sec)
{
  size_t dim;
  if(the_model->is_factorized){
    B = the_model->provide_factorized_basis(sec);
    dim = B->dim;
  }
  else{
    dim = the_model->provide_basis(sec)->dim;
    B = the_model->provide_basis(sec);
  }
  elem.resize(dim);
  
  for(size_t I=0; I<B->dim; ++I){
    double val(0.0);
    for(auto &x : elements){
      uint64_t mask = binary_state::mask(x.r,B->L) + binary_state::mask(x.c,B->L);
      if((B->bin(I).b & mask) == mask) val += x.v;
    }
    elem[I] = val;
  }
}




/**
 applies the operator on the vector x and adds the results (times z) to the vector y
 */
void HS_interaction_operator::multiply_add(const vector<double> &x, vector<double> &y, double z)
{
  for(size_t i = 0; i < B->dim; ++i) y[i] += x[i]*elem[i]*z;
}

void HS_interaction_operator::multiply_add(const vector<Complex> &x, vector<Complex> &y, double z)
{
  for(size_t i = 0; i < B->dim; ++i) y[i] += x[i]*elem[i]*z;
}




/**
 produces a dense form of the operator
 */
void HS_interaction_operator::dense_form(matrix<double> &h, double z)
{
  if(h.v.size() == 0) return;
  for(size_t i = 0; i < B->dim; ++i) h(i,i) += elem[i]*z;
}
void HS_interaction_operator::dense_form(matrix<Complex> &h, double z)
{
  if(h.v.size() == 0) return;
  for(size_t i = 0; i < B->dim; ++i) h(i,i) += elem[i]*z;
}



/**
 fills a map, in order to construct the CSR form of the Hamiltonian
 */
void HS_interaction_operator::CSR_map(map<index_pair,double> &E, vector<double> &D, double z)
{
  mult_add(z, elem, D);
}
void HS_interaction_operator::CSR_map(map<index_pair,Complex> &E, vector<double> &D, double z)
{
  mult_add(z, elem, D);
}



void HS_interaction_operator::diag(vector<double> &d, double z)
{
  for(size_t i=0; i<elem.size(); i++) d[i] += z*elem[i];
}

/**
 populates a vector of matrix_element to build the Hamiltonian
 */
void HS_interaction_operator::Triplet_COO_map(vector<matrix_element<double>>& E, double z, bool sym_store)
{
    //diag element
    for(size_t i=0; i<elem.size(); i++) {
        matrix_element<double> T(i,i,z*elem[i]);
        E.push_back(T);
    }
}

/**
 populates a vector of matrix_element to build the Hamiltonian
 */
void HS_interaction_operator::Triplet_COO_map(vector<matrix_element<Complex>>& E, double z, bool sym_store)
{
    //diag element
    for(size_t i=0; i<elem.size(); i++) {
        matrix_element<Complex> T(i,i,z*elem[i]);
        E.push_back(T);
    }
}


#endif /* interaction_operator_h */
