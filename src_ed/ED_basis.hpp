#ifndef ED_basis_h
#define ED_basis_h

#include <cstdio>
#include <string>
#include <map>
#include <memory>
#include <tuple>

#include "parser.hpp"
#include "symmetry_group.hpp"
#include "sector.hpp"
#include "binary_state.hpp"

#define PARAMETER_ZERO 1e-10 //!< absolute value below which a parameter is considered exactly zero
#define MIN_DUMP_SIZE 13	//!< minimum number of orbitals beyond which the operators are dumped on file
#define MIN_BASIS_DUMP_SIZE 13
#define MAX_ORBITAL 32

//! abstract class for a basis in a Hilbert space sector
struct ED_basis
{
	int L; //!< number of orbitals of a given spin
	sector sec; //!< sector of the Hilbert space
	std::string name; //!< Name of basis (identifies the sector and the irrep)
	uint32_t dim; //!< dimension of the Hilbert space
	
	static bool verb;
	
	ED_basis(const sector &_sec, int L);
	virtual binary_state bin(uint32_t I) const = 0;
	virtual uint32_t index(const binary_state &b) const = 0;
	virtual void print_state(std::ostream &flux, uint32_t i) const = 0;

	template<typename T>
	void print_state(std::ostream &flux, const vector<T>& psi) const
	{
		if(psi.size() != dim) qcm_ED_throw("dimensions of vector psi and basis do not match!");
		if(psi.size() > global_int("max_dim_print")) return;
		for(int i=0; i<psi.size(); i++){
			flux << psi[i] << "\t: ";
			print_state(flux, i);
			flux << '\n';
		}
	}
};


//! Basis in a tensor factor of the Hilbert space (spin up or spin down)
/**
It is used in the "factorized format" of the Hamiltonian, which applies only in mixing cases 0 and 4.
*/
struct ED_halfbasis
{
	int N; //!< number of particles
	int L; //!< number of orbitals (sites)
	uint32_t dim; //!< dimension of the Hilbert space
	vector<uint32_t> bin; //!< binary representation of states
	
	ED_halfbasis(int L, int N);
	uint32_t index(const uint32_t &b) const;
	void print_state(std::ostream &flux, int i) const;
};


//! Basis in a sector of the Hilbert space when the latter is a tensor product of up and down spins
/**
It is used in the "factorized format" of the Hamiltonian, which applies only in mixing cases 0 and 4.
*/
struct ED_factorized_basis : ED_basis
{
	shared_ptr<ED_halfbasis> up;
	shared_ptr<ED_halfbasis> down;

	ED_factorized_basis(const sector &_sec, int _L);
	binary_state bin(uint32_t I) const;
	uint32_t index(const binary_state &b) const;
	void print_state(std::ostream &flux, uint32_t i) const;
	tuple<uint32_t, uint32_t, uint32_t, uint32_t> components(uint32_t label)  const;
};


//! Basis in a sector of the Hilbert space, but not described as a tensor product. Applies to all mixings
struct ED_mixed_basis : ED_basis
{
	shared_ptr<symmetry_group> group; //!< pointer to symmetry group
	vector<binary_state> binlist; //!< binary representation of states
		
	ED_mixed_basis(const sector &_sec, shared_ptr<symmetry_group> _group);
	~ED_mixed_basis();
	binary_state bin(uint32_t I) const;
	uint32_t index(const binary_state &b) const;
	void print_state(std::ostream &flux, uint32_t i) const;
};


std::ostream & operator<<(std::ostream &flux, const ED_basis &B);
tuple<uint32_t, int, int, bool> Destroy(const int a, const uint32_t label, const ED_mixed_basis &Bx, const ED_mixed_basis &By);
tuple<uint32_t, int, int, bool>  Destroy(const int a, const uint32_t label, const ED_factorized_basis &Bx, const ED_factorized_basis &By);

#endif
