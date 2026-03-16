/*
template specializations of destruction_operator
 */

#include "destruction_operator.hpp"

/**
 constructor
 @param _B source (bra) basis
 @param _T target (ket) basis after annihilation
 @param orb symmetric orbital on which the destruction operator acts
 */
template<>
destruction_operator<double>::destruction_operator(shared_ptr<ED_mixed_basis> _B, shared_ptr<ED_mixed_basis> _T, const symmetric_orbital &orb)
: HS_operator<double>(_T->dim, _B->dim), B(_B), T(_T), sorb(orb)
{
  int n = B->L;
  shared_ptr<symmetry_group> group = B->group;
  ostringstream sout; sout << *this; 
  if(global_bool("verb_ED")) cout << sout.str() << endl;
  for(uint32_t I=0; I<B->dim; ++I){
    auto R = group->Representative(B->bin(I), B->sec.irrep); // just to get 'length'
    for(int site=0; site < group->n_sites; ++site){
      if(group->S(orb.label,site) == 0.0) continue;
      auto P = Destroy(site + orb.spin*n, I, *B, *T);
      if(!get<3>(P)) continue;
      get<1>(P) = get<1>(P)%(2*group->g);
      double X =  group->phaseR[get<1>(P)] * group->S_real(orb.label,site) * sqrt((1.0*R.length)/get<2>(P));
      insert(X, get<0>(P), I);
    }
  }
  consolidate();
}




/**
 constructor
 @param _B source (bra) basis
 @param _T target (ket) basis after annihilation
 @param orb symmetric orbital on which the destruction operator acts
 */
template<>
destruction_operator<Complex>::destruction_operator(shared_ptr<ED_mixed_basis> _B, shared_ptr<ED_mixed_basis> _T, const symmetric_orbital &orb)
: HS_operator<Complex>(_T->dim, _B->dim), B(_B), T(_T), sorb(orb)
{
  int n = B->L;
  shared_ptr<symmetry_group> group = B->group;
  ostringstream sout; sout << *this;
  if(global_bool("verb_ED")) cout << sout.str() << endl;
  for(uint32_t I=0; I<B->dim; ++I){
    auto R = group->Representative(B->bin(I), B->sec.irrep); // just to get 'length'
    for(int site=0; site < group->n_sites; ++site){
      if(group->S(orb.label,site) == Complex(0.0)) continue;
      auto P = Destroy(site + orb.spin*n, I, *B, *T);
      if(!get<3>(P)) continue;
      get<1>(P) = get<1>(P)%(2*group->g);
      Complex X = group->phaseC[get<1>(P)] * conjugate(group->S(orb.label,site)) * sqrt((1.0*R.length)/get<2>(P));
      insert(X, get<0>(P), I);
    }
  }
  consolidate();
}
