/*
 Implementation of the Hamiltonian in CSR format (own implementation)
*/

#ifndef Hamiltonian_operator
#define Hamiltonian_operator

#include "Hamiltonian_base.hpp"


template<typename HilbertField>
class Hamiltonian_Operator: public Hamiltonian<HilbertField>
{
    public:
    
        Hamiltonian_Operator(
            shared_ptr<model> the_model, 
            const map<string, double> &value,
            sector _sec
        );
        void mult_add(vector<HilbertField> &x, vector<HilbertField> &y);
    
    private:
    
        map<shared_ptr<HS_Hermitian_operator>, double> sparse_ops; //!< correpondence between terms in H and their coefficients
        void HS_ops_map(const map<string, double> &value);

};

/**
 constructor
 */
template<typename HilbertField>
Hamiltonian_Operator<HilbertField>::Hamiltonian_Operator(
    shared_ptr<model> _the_model,
    const map<string, double> &value, 
    sector _sec
) {
    this->the_model = _the_model;
    this->sec = _sec;
    this->B = _the_model->provide_basis(_sec);
    this->dim = this->B->dim;
    
    HS_ops_map(value);
}

//GS_energy
//states
//print
//to_dense

/**
 Applies the Hamiltonian: y = y +H.x
 @param y vector to which H.x is added to
 @param x input vector
 */
template<typename HilbertField>
void Hamiltonian_Operator<HilbertField>::mult_add(vector<HilbertField> &x, vector<HilbertField> &y)
{
    for(auto& h : sparse_ops) {
        h.first->multiply_add(x, y, h.second);
    }
}


/**
 builds HS_operators as needed
 */
template<typename HilbertField>
void Hamiltonian_Operator<HilbertField>::HS_ops_map(const map<string, double> &value)
{
    bool is_complex = false;
    if(typeid(HilbertField) == typeid(Complex)) is_complex = true;
    for(auto& x : value){
        Hermitian_operator& op = *this->the_model->term.at(x.first);
        if(op.HS_operator.find(this->sec) == op.HS_operator.end()){
            op.HS_operator[this->sec] = op.build_HS_operator(this->sec, is_complex); // ***TEMPO***
        }
        sparse_ops[op.HS_operator.at(this->sec)] = value.at(x.first);
    }
}


#endif
