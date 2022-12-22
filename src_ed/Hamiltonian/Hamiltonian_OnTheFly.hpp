/*
 Implementation of the Hamiltonian in CSR format (own implementation)
*/

#ifndef Hamiltonian_otf
#define Hamiltonian_otf

#include "Hamiltonian_base.hpp"


template<typename HilbertField>
class Hamiltonian_OnTheFly: public Hamiltonian<HilbertField>
{
    public:
    
        Hamiltonian_OnTheFly(
            shared_ptr<model> the_model, 
            const map<string, double> &value,
            sector _sec
        );
        void mult_add(vector<HilbertField> &x, vector<HilbertField> &y);
    
    private:
    
        map<shared_ptr<Hermitian_operator>, double> ops; //!< correpondence between terms in H and their coefficients
        void ops_map(const map<string, double> &value);

};

/**
 constructor
 */
template<typename HilbertField>
Hamiltonian_OnTheFly<HilbertField>::Hamiltonian_OnTheFly(
    shared_ptr<model> _the_model,
    const map<string, double> &value, 
    sector _sec
) {
    this->the_model = _the_model;
    this->sec = _sec;
    this->B = _the_model->provide_basis(_sec);
    this->dim = this->B->dim;
    ops_map(value);
}


/**
 Applies the Hamiltonian: y = y +H.x
 @param y vector to which H.x is added to
 @param x input vector
 */
template<typename HilbertField>
void Hamiltonian_OnTheFly<HilbertField>::mult_add(vector<HilbertField> &x, vector<HilbertField> &y)
{
    for(auto& h : ops){
        h.first->multiply_add_OTF(x, y, h.second, this->B);
    }
    return;
}

//GS_energy
//states
//print
//to_dense


/**
 returns a map of Hermitian operators to value
 */
template<typename HilbertField>
void Hamiltonian_OnTheFly<HilbertField>::ops_map(const map<string, double> &value)
{
    for(auto& x : value){
      ops[this->the_model->term.at(x.first)] = value.at(x.first);
    }
    return;
}

#endif
