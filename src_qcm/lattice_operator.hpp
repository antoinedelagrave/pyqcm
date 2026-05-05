#ifndef latticeOperator_h
#define latticeOperator_h

/**
 * @file lattice_operator.hpp
 * @brief Declaration of the lattice_operator struct (terms in the lattice Hamiltonian).
 */

#include <iomanip>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iterator>

#include "vector3D.hpp"
#include "types.hpp"
#include "qcm_ED.hpp"
#include "lattice_matrix_element.hpp"
#include "sector.hpp"

enum class latt_op_type {one_body, singlet, dz, dy, dx, Hubbard, Hund, Heisenberg, X, Y, Z};

template<typename T>
T native(Complex x);

struct lattice_model;


//!store information about an operator of the model (a term in the Hamiltonian)
struct lattice_operator
{
	bool is_active; //!< true if the operator is activated (nonzero or tagged)
	bool is_closed; //!< true if the operator cannot accept new matrix elements as it is declared complete
	bool is_complex; //!< true is the operator has complex matrix elements, which requires a complex Hilbert space
	bool is_density_wave; //!< true if the operator is a density wave, and thus may be different from one cluster to the next
	bool is_interaction; //!< true if the operator is activated in one of the "lattice_model_instances", even with value zero
	double average; //!< average (from the Green function)
	double nambu_correction_full; //!< correction to averages needed in the full Nambu formalism
	double nambu_correction; //!< correction to averages needed in the simple Nambu formalism
	double norm; //!< multiplier from the raw average to the printed average
	int mixing; //!< mixing caused by this operator (default = normal)
	lattice_model &model; //!< backtrace to the lattice model
	latt_op_type type; //!< type of the operator (see enum defined above)
	static map<string,latt_op_type> op_type_map;
	string name; //!< name of the operator
	vector<bool> in_cluster; //!< true if the operator exists in a given cluster.
	vector<GF_matrix_element<Complex>> IGF_elem_down;  //!< list of inter-SUC matrix elements for the GF (takes mixing into account)
	vector<GF_matrix_element<Complex>> IGF_elem;  //!< list of inter-SUC matrix elements for the GF (takes mixing into account)
	vector<lattice_matrix_element> elements;
	vector<matrix_element<Complex>> GF_elem_down; //!< list of intra-SUC matrix elements for the GF (takes mixing into account)
	vector<matrix_element<Complex>> GF_elem; //!< list of intra-SUC matrix elements for the GF (takes mixing into account)
	
	
	lattice_operator(lattice_model& _model, const string& _name, latt_op_type _type = latt_op_type::one_body);
	lattice_operator(lattice_model& _model, const string& _name, const string& _type);
	void add_matrix_element(const vector3D<int64_t>& pos1, const vector3D<int64_t>& link, Complex v, const string& opt="");
	void close();
	void consolidate();
	void check_spin_symmetry();
	void one_body_matrix(bool spin_down);
};


#endif






