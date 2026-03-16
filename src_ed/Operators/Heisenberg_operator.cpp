#include "Heisenberg_operator.hpp"
#include "HS_Heisenberg_operator.hpp"



/**
 Constructor from name and matrix elements
 @param _name   name of the operator
 @param _the_model   pointer to model
 @param _elements   nonzero one-body matrix elements
 */
Heisenberg_operator::Heisenberg_operator(const string &_name, shared_ptr<model> _the_model, const vector<matrix_element<double>>& _elements, char _dir)
: Hermitian_operator(_name, _the_model), elements(_elements), dir(_dir)
{
  is_interaction = true;
  for(auto& x : elements) element_map[{x.r, x.c}] = x.v;
  the_model->group->check_invariance<double>(element_map, name, true);
  element_map.clear();
}



/**
 set the target of an operator
 1 : cluster
 2 : bath only
 3 : hybridization
 @param in_bath vector of bool defining the status of each site
 */
void Heisenberg_operator::set_target(vector<bool> &in_bath){
  this->target = 1;
  for(auto& x : elements){
    if(in_bath[x.r] or in_bath[x.c])
      qcm_ED_throw("Heisenberg operator "+this->name+" must be defined on the cluster only");
  }
}

/**
 returns a pointer to, and constructs the associated HS operator in the sector with basis B.
 @param sec sector of the Hilbert space in which to build the operator
 @param complex_Hilbert_space if true, builds the operator in a complex Hilbert space
 */
shared_ptr<HS_Hermitian_operator> Heisenberg_operator::build_HS_operator(sector sec, bool complex_Hilbert_space)
{
  shared_ptr<ED_mixed_basis> B = the_model->provide_basis(sec);
  if(B->group->complex_irrep[B->sec.irrep]) return make_shared<HS_Heisenberg_operator<Complex>>(the_model, name, sec, elements, dir);
  else return make_shared<HS_Heisenberg_operator<double>>(the_model, name, sec, elements, dir);
}



/**
 prints definition to a file
 @param fout output stream
 */
void Heisenberg_operator::print(ostream& fout)
{
  fout << "\nHeisenberg operator (type " << dir << ") " << name << endl;
  for(auto& x : elements) fout << x.r+1 << '\t' << x.c+1 << '\t' << x.v << endl;
}

/**
 returns a list of complexified matrix elements
 */
vector<matrix_element<Complex>> Heisenberg_operator::matrix_elements()
{
  vector<matrix_element<Complex>> celem(elements.size());
  for(int i=0; i<elements.size(); i++){
    auto el = elements[i];
    celem[i] = {el.r, el.c, Complex(el.v)};
  }
  return celem;
}


void Heisenberg_operator::multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  qcm_ED_throw("on the fly computation imnpossible with a Heisenberg operator");
}

void Heisenberg_operator::multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  qcm_ED_throw("on the fly computation imnpossible with a Heisenberg operator");
}
