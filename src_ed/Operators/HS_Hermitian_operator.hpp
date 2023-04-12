/*
 Abstract class representing a Hermitian operator (Hermitian_operator)
 and its realization in the Hilbert space (HS_Hermitian_operator)
 */

#ifndef HS_Hermitian_operator_h
#define HS_Hermitian_operator_h

#include "model.hpp"

//! Abstract class for a Hermitian operator in a sector of the Hilbert space
struct HS_Hermitian_operator
{
  string name; //!< name of the operator
  sector sec; //!< sector of the HS on which it acts
  shared_ptr<model> the_model; //!< backtrace to the model

  HS_Hermitian_operator(shared_ptr<model> _the_model, const string &_name, sector _sec) 
  : the_model(_the_model), name(_name), sec(_sec) {}
  
  virtual void multiply_add(const vector<double> &x, vector<double> &y, double z) = 0;
  virtual void multiply_add(const vector<Complex> &x, vector<Complex> &y, double z) = 0;
  virtual void diag(vector<double> &d, double z) = 0;
  virtual void dense_form(matrix<double> &h, double z) = 0;
  virtual void dense_form(matrix<Complex> &h, double z) = 0;
  virtual void CSR_map(map<index_pair,double> &elem, vector<double> &diag, double z) = 0;
  virtual void CSR_map(map<index_pair,Complex> &elem, vector<double> &diag, double z) = 0;
  //The two below functions can be parallelize but does it worth it ?
  virtual void Triplet_COO_map(vector<matrix_element<double>>& E, double z, bool sym_store) {};
  virtual void Triplet_COO_map(vector<matrix_element<Complex>>& E, double z, bool sym_store) {};

  template<typename HS_field>
  void expectation_value(const state<HS_field> &gs, double &average, double &average2);

};


//==============================================================================
// implementation


/**
 Computes the expectation value and variance of the operator in the state gs
 */
template<typename HS_field>
void HS_Hermitian_operator::expectation_value(const state<HS_field> &gs, double &average, double &average2)
{
  vector<HS_field> tmp_gs(gs.psi.size());
  multiply_add(gs.psi,tmp_gs,1.0);
  average += gs.weight*real(tmp_gs*gs.psi);
  average2 += gs.weight*norm2(tmp_gs);
}


#endif
