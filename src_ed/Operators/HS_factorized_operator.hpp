/*
 Abstract class representing a Hermitian operator (Hermitian_operator)
 and its realization in the Hilbert space (HS_Hermitian_operator)
 */

#ifndef HS_factorized_operator_h
#define HS_factorized_operator_h

#include "HS_Hermitian_operator.hpp"
#include "HS_half_operator.hpp"

template<typename op_field>
struct one_body_operator;

//! Represents a factorizable (up x down) one-body operator in a sector of the Hilbert space
template<typename op_field>
struct HS_factorized_operator : HS_Hermitian_operator
{
  shared_ptr<ED_factorized_basis> B;
  shared_ptr<HS_half_operator<op_field>> up;
  shared_ptr<HS_half_operator<op_field>> down;
  
  HS_factorized_operator(shared_ptr<model> _the_model, const string &_name, sector _sec, one_body_operator<op_field> *op);
  
  void multiply_add(const vector<double> &x, vector<double> &y, double z);
  void multiply_add(const vector<Complex> &x, vector<Complex> &y, double z);
  void dense_form(matrix<Complex> &h, double z){}
  void dense_form(matrix<double> &h, double z){}
  void CSR_map(map<index_pair,double> &elem, vector<double> &diag, double z){}
  void CSR_map(map<index_pair,Complex> &elem, vector<double> &diag, double z){}
  void diag(vector<double> &Y, double z){}
};

//==============================================================================
// implementation

template<typename op_field>
HS_factorized_operator<op_field>::HS_factorized_operator(shared_ptr<model> _the_model, const string &_name, sector _sec, one_body_operator<op_field> *op)
: HS_Hermitian_operator(_the_model, _name, _sec)
{
  B = the_model->provide_factorized_basis(sec);
  int Nup = sec.Nup();
  int Ndw = sec.Ndw();

  if(op->half_operators.find(Nup) == op->half_operators.end()){
    op->half_operators[Nup] = make_shared<HS_half_operator<op_field>>(HS_half_operator<op_field>(op->elements, B->up, false));
  }
  up = op->half_operators[Nup];
  if(op->half_operators_dw.find(Ndw) == op->half_operators_dw.end()){
    op->half_operators_dw[Ndw] = make_shared<HS_half_operator<op_field>>(HS_half_operator<op_field>(op->elements, B->down, true));
  }
  down = op->half_operators_dw[Ndw];
}


template<typename op_field>
void HS_factorized_operator<op_field>::multiply_add(const vector<double> &x, vector<double> &y, double z)
{
  auto Nup = sec.Nup();
  auto Ndw = sec.Ndw();
  auto dimup = up->dim;
  auto dimdw = down->dim;

  // spin down
  for(auto& w : down->v){
    double z2=real(w.first*z);
    for(auto& e : w.second) 
      for(int i=0; i<dimup; i++) y[e.first*dimup+i] += z2*x[e.second*dimup+i];
    for(auto& e : w.second)
      for(int i=0; i<dimup; i++) y[e.second*dimup+i] += z2*x[e.first*dimup+i];
  }
  for(auto& e : down->diag_elem) 
    for(int i=0; i<dimup; i++) y[e.r*dimup+i] += z*e.v*x[e.r*dimup+i];

  // spin up
  for(int i=0; i<dimdw; i++){
    for(auto& w : up->v){
      double z2=real(w.first*z);
      for(auto& e : w.second) y[e.first+dimup*i] += z2*x[e.second+dimup*i];
      for(auto& e : w.second) y[e.second+dimup*i] += z2*x[e.first+dimup*i];
    }
    for(auto& e : up->diag_elem) y[e.r+dimup*i] += z*e.v*x[e.r+dimup*i];
  }
}


template<typename op_field>
void HS_factorized_operator<op_field>::multiply_add(const vector<Complex> &x, vector<Complex> &y, double z)
{
  auto Nup = sec.Nup();
  auto Ndw = sec.Ndw();
  auto dimup = up->dim;
  auto dimdw = down->dim;

  // spin down
  for(auto& w : down->v){
    Complex z2= w.first*z;
    for(auto& e : w.second) 
      for(int i=0; i<dimup; i++) y[e.first*dimup+i] += z2*x[e.second*dimup+i];
    for(auto& e : w.second)
      for(int i=0; i<dimup; i++) y[e.second*dimup+i] += z2*x[e.first*dimup+i];
  }
  for(auto& e : down->diag_elem) 
    for(int i=0; i<dimup; i++) y[e.r*dimup+i] += z*e.v*x[e.r*dimup+i];

  // spin up
  for(int i=0; i<dimdw; i++){
    for(auto& w : up->v){
      Complex z2= w.first*z;
      for(auto& e : w.second) y[e.first+dimup*i] += z2*x[e.second+dimup*i];
      for(auto& e : w.second) y[e.second+dimup*i] += z2*x[e.first+dimup*i];
    }
    for(auto& e : up->diag_elem) y[e.r+dimup*i] += z*e.v*x[e.r+dimup*i];
  }
}

#endif
