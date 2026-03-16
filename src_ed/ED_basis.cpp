/*! \file
	\brief Construction of the Hilbert space basis for the exact diagonalization solver
 */
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <set>
#include <algorithm>

#include "ED_basis.hpp"
#include "symmetry_group.hpp"
#include "fraction.hpp"

//------------------------------------------------------------------------------
// external variables or declarations

using namespace std;

//------------------------------------------------------------------------------
// variables local to this file

map<pair<int, int>, shared_ptr<ED_halfbasis>> halfbasis; //!< list of bases of the factorized space

//------------------------------------------------------------------------------
// declarations local to this file

bool ED_basis::verb = false;

/**
 Constructor of the basis in the sector the_sec
 @param _sec sector of the Hilbert space considered
 @param _L number of orbitals (sites)
 */
ED_basis::ED_basis(const sector &_sec, int _L): sec(_sec) , L(_L), dim(0)
{
	if(verb) cout << "construction of basis in sector " << sec.name() << endl;
	name = sec.name();
}




/**
 Applies the creation (\a op = 1) or destruction (\a op = -1) operator on a state
 of Basis Bx.
 Will necessarily give a state of Basis By, or zero.
 returns false if the state is null
 
 @param a orbital at which the creation/annihilation takes place
 @param label basis state label on which this is applied in basis \a Bx
 @param labelp	label of the resulting basis state in basis \a By
 @param Bx	initial basis
 @param By target basis
 returns {labelp, phase, length, flag}
 phase : phase in factor exp(i phase pi /g) produced by applying the operator.
 length : length of the cycle of the created state (useful to calculate its normalization)
 flag : true if resulting state is nonzero

 */
tuple<uint32_t, int, int, bool>  Destroy(const int a, const uint32_t label, const ED_mixed_basis &Bx, const ED_mixed_basis &By)
{
	tuple<uint32_t, int, int, bool> T(0,0,0,false);
	uint64_t mask = (a >= Bx.group->N) ? 1UL << (a - Bx.group->N + 32) : 1UL << a; // the literal '1UL' is crucial (not '1')
	binary_state ss = Bx.binlist[label];
	if(!(ss.b&mask)) return T;
	auto X = Bx.group->Representative(ss.b^mask, By.sec.irrep);
	if(X.length==0) return T;
	uint32_t labelp = By.index(X.b);
	if(labelp == By.dim) return T; // replace by exception raise
	if(bitcount64(ss.b & (mask-1)) & 1) X.phase += Bx.group->g;
	get<3>(T) = true;
	get<0>(T) = labelp;
	get<1>(T) = X.phase;
	get<2>(T) = X.length;
	// cout << a << '\t' << label << '\t' << labelp << endl; // tempo
	return T;
}

tuple<uint32_t, int, int, bool>  Destroy(const int a, const uint32_t label, const ED_factorized_basis &Bx, const ED_factorized_basis &By)
{
	tuple<uint32_t, int, int, bool> T(0,0,0,false);
	uint32_t label_up = label%Bx.up->dim;
	uint32_t label_down = label/Bx.up->dim;
	uint32_t ss_up = Bx.up->bin[label_up];
	uint32_t ss_down = Bx.down->bin[label_down];

	// tuple<uint32_t, uint32_t, uint32_t, uint32_t> C = Bx.components(label);

	uint32_t ssp, labelp;
	if(a >= Bx.L){ // down orbital
		uint32_t mask = 1U << (a - Bx.L);
		if(mask&ss_down) ssp = ss_down^mask;
		else return T;
		get<1>(T) = (bitcount32(ss_down&(mask-1)) + bitcount32(ss_up))%2;
		labelp = By.down->index(ssp);
		get<0>(T) = labelp*By.up->dim + label_up;
	}
	else{
		uint32_t mask = 1U << a;
		if(mask&ss_up) ssp = ss_up^mask;
		else return T;
		get<1>(T) = (bitcount32(ss_up&(mask-1)))%2;
		labelp = By.up->index(ssp);
		get<0>(T) = label_down*By.up->dim + labelp;
	}
	get<2>(T) = 1;
	get<3>(T) = true;
	// cout << a << '\t' << label << '\t' << get<0>(T) << endl; // tempo
	return T;
}

//==============================================================================
// class ED_halfbasis

ED_halfbasis::ED_halfbasis(int _L, int _N): L(_L), N(_N)
{
	size_t max_dim = 1<<L; // = 2^L
	bin.reserve(1024);

	for(uint32_t b=0; b < max_dim; ++b){
		if(bitcount32(b) != N) continue;
		bin.push_back(b);
	}
	dim = bin.size();
}
		
/**
 finds the index of a state by binary search
 */
uint32_t ED_halfbasis::index(const uint32_t &b) const
{
	uint32_t I = lower_bound(bin.begin(), bin.end(), b) - bin.begin();
	if(I < dim and bin[I]==b) return I;
	else return dim;
}



//==============================================================================
// class ED_factorized_basis

ED_factorized_basis::ED_factorized_basis(const sector &_sec, int _L)
: ED_basis(_sec, _L)
{
	int Nup = sec.Nup();
	int Ndw = sec.Ndw();
	if(Nup > 32 or Ndw > 32) qcm_ED_throw("construction of factorized basis impossible : wrong mixing or N too large");

    if(halfbasis.find({L,Nup}) == halfbasis.end()) halfbasis[{L,Nup}] = make_shared<ED_halfbasis>(L, Nup);
    if(halfbasis.find({L,Ndw}) == halfbasis.end()) halfbasis[{L,Ndw}] = make_shared<ED_halfbasis>(L, Ndw);

	up = halfbasis[{L,Nup}];
	down = halfbasis[{L,Ndw}];
	dim = up->dim*down->dim;
}

binary_state ED_factorized_basis::bin(uint32_t I) const
{
	return binary_state(up->bin[I%up->dim], down->bin[I/up->dim]);
}

uint32_t ED_factorized_basis::index(const binary_state &b) const
{
	return up->index(b.left()) + up->dim*down->index(b.right());
}

void ED_factorized_basis::print_state(std::ostream &flux, uint32_t i) const
{
	PrintBinaryDouble(flux, bin(i).b, L);
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t> ED_factorized_basis::components(uint32_t label) const
{
	uint32_t label_up = label%(up->dim);
	uint32_t label_down = label/up->dim;
	uint32_t ss_up = up->bin[label_up];
	uint32_t ss_down = down->bin[label_down];
	return tuple<uint32_t, uint32_t, uint32_t, uint32_t>(label_up, label_down, ss_up, ss_down);
}


//==============================================================================
// class ED_mixed_basis

ED_mixed_basis::ED_mixed_basis(const sector &_sec, shared_ptr<symmetry_group> _group): 
ED_basis(_sec, _group->N), group(_group)
{	
	size_t n_orb = group->N;
	
	if(n_orb > MAX_ORBITAL){
		qcm_ED_throw("Number of orbitals (sites) exceeds limit of " + to_string(MAX_ORBITAL));
	}
	binlist.reserve(4096);
	
	
	//..............................................................................
	// Builds the basis in the completely mixed case
	
	if(sec.N == sector::even and sec.S == sector::even){
		size_t max_dimup = 1<<n_orb; // = 2^n
		binlist.reserve(4096);
		dim=0;
		// Putting together
		int phase;
		int length;
		for(uint32_t i=0; i<max_dimup; ++i){
			for(uint32_t j=0; j<max_dimup; ++j){
				binary_state b(i,j);
				if(b.count()%2) continue; // reject if odd number of particles
				binary_state b0=b;
				auto R = group->Representative(b, sec.irrep);
				if(R.length and  R.b == b0) binlist.push_back(R.b);
			}
		}
	}
	if(sec.N == sector::odd and sec.S == sector::odd){
		size_t max_dimup = 1<<n_orb; // = 2^n
		binlist.reserve(4096);
		dim=0;
		// Putting together
		int phase;
		int length;
		for(uint32_t i=0; i<max_dimup; ++i){
			for(uint32_t j=0; j<max_dimup; ++j){
				binary_state b(i,j);
				if(b.count()%2 == 0) continue; // reject if even number of particles
				binary_state b0=b;
				auto R = group->Representative(b, sec.irrep);
				if(R.length and  R.b == b0) binlist.push_back(R.b);
			}
		}
	}
	
	//..............................................................................
	// Builds the basis in the anomalous case (no spin flip)
	
	else if(sec.N >= sector::odd and sec.S < sector::odd){
		size_t max_dimup = 1<<n_orb; // = 2^n
		dim=0;
		for(size_t Nup=0; Nup <= n_orb; Nup++){
			size_t Ndw = Nup - sec.S;
			if(Ndw > n_orb) continue;
			vector<uint32_t> bin_up; // list of binary up states with Nup electrons
			vector<uint32_t> bin_dw; // list of binary up states with Ndw electrons
			bin_up.reserve(1024);
			bin_dw.reserve(1024);
			// construction of bin_up
			for(uint32_t bup=0; bup < max_dimup; ++bup){
				if(bitcount32(bup) != Nup) continue;
				bin_up.push_back(bup);
			}
			size_t nbup = bin_up.size();
			// construction of bin_dw
			for(uint32_t bdw=0; bdw < max_dimup; ++bdw){
				if(bitcount32(bdw) != Ndw) continue;
				bin_dw.push_back(bdw);
			}
			size_t nbdw = bin_dw.size();
			
			// Putting together
			int phase;
			int length;
			for(size_t i=0; i<nbup; ++i){
				for(size_t j=0; j<nbdw; ++j){
					binary_state b(bin_dw[j],bin_up[i]);
					binary_state b0=b;
					auto R = group->Representative(b, sec.irrep);
					if(R.length and  R.b == b0) binlist.push_back(R.b);
				}
			}
		}
	}
	
	//..............................................................................
	// Builds the basis in the spin flip case (no anomalous terms)
	
	else if(sec.N < sector::odd and sec.S >= sector::odd){
		uint32_t max_dimup = 1<<n_orb; // = 2^n
		dim=0;
		for(size_t Nup=0; Nup <= n_orb; Nup++){
			size_t Ndw = -Nup + sec.N;
			if(Ndw > n_orb) continue;
			vector<uint32_t> bin_up; // list of binary up states with Nup electrons
			vector<uint32_t> bin_dw; // list of binary up states with Ndw electrons
			binlist.reserve(4096);
			bin_up.reserve(1024);
			bin_dw.reserve(1024);
			// construction of bin_up
			for(uint32_t bup=0; bup < max_dimup; ++bup){
				if(bitcount32(bup) != Nup) continue;
				bin_up.push_back(bup);
			}
			size_t nbup = bin_up.size();
			// construction of bin_dw
			for(uint32_t bdw=0; bdw < max_dimup; ++bdw){
				if(bitcount32(bdw) != Ndw) continue;
				bin_dw.push_back(bdw);
			}
			size_t nbdw = bin_dw.size();
			
			// Putting together
			int phase;
			int length;
			for(size_t i=0; i<nbup; ++i){
				for(size_t j=0; j<nbdw; ++j){
					binary_state b(bin_dw[j],bin_up[i]);
					binary_state b0=b;
					auto R = group->Representative(b, sec.irrep);
					if(R.length and  R.b == b0) binlist.push_back(R.b);
				}
			}
		}
	}
	
	//..............................................................................
	// Builds the basis in the unmixed case
	
	else {
		size_t max_dimup = 1<<n_orb; // = 2^n
		size_t Nup = (sec.N + sec.S)/2; // number of up electrons
		size_t Ndw = (sec.N - sec.S)/2; // number of down electrons
		vector<uint32_t> bin_up; // list of binary up states with Nup electrons
		vector<uint32_t> bin_dw; // list of binary up states with Ndw electrons
		bin_up.reserve(1024);
		bin_dw.reserve(1024);
		
		// construction of bin_up
		for(uint32_t bup=0; bup < max_dimup; ++bup){
			if(bitcount32(bup) != Nup) continue;
			bin_up.push_back(bup);
		}
		size_t nbup = bin_up.size();
		
		// construction of bin_dw
		for(uint32_t bdw=0; bdw < max_dimup; ++bdw){
			if(bitcount32(bdw) != Ndw) continue;
			bin_dw.push_back(bdw);
		}
		size_t nbdw = bin_dw.size();
		
		dim=0;
		// Putting together
		int phase;
		int length;
		for(size_t i=0; i<nbup; ++i){
			for(size_t j=0; j<nbdw; ++j){
				binary_state b(bin_dw[j],bin_up[i]);
				binary_state b0=b;
				auto R = group->Representative(b, sec.irrep);
				if(R.length and  R.b == b0) binlist.push_back(R.b);
			}
		}
	}

  //..............................................................................
  // sorting
  
  dim = binlist.size();
  sort(binlist.begin(), binlist.end());
  if(verb) cout << "dimension = " << dim << endl;
}

/**
 Destructor
 */
ED_mixed_basis::~ED_mixed_basis()
{
	binlist.clear();
}


inline binary_state ED_mixed_basis::bin(uint32_t I) const
{
	return binlist[I];
}

/**
 finds the index of a state by binary search
 */
uint32_t ED_mixed_basis::index(const binary_state &b) const
{
  uint32_t I = lower_bound(binlist.begin(), binlist.end(), b) - binlist.begin();
  if(I < dim and binlist[I]==b) return I;
  else return dim;
}


/**
 Prints a basis state
 */
void ED_mixed_basis::print_state(std::ostream &flux, uint32_t i) const
{
	int phase;
	auto R = group->Representative(binlist[i], sec.irrep);
	if(R.length == 0) return;
	if(group->g > 1) flux << '[' << R.length << "]\t";
	PrintBinaryDouble(flux, binlist[i].b, group->N);
	set<binary_state> s;
	for(size_t j=1; j<group->g; ++j){
		s.insert(binlist[i]);
		auto Y = group->apply(j, sec.irrep, binlist[i]);
		if(s.count(Y.first)==0){
			s.insert(Y.first);
			if(Y.second==0) flux << " + ";
			else if(Y.second==group->g) flux << " - ";
			else{
				fraction f((int)Y.second,(int)group->g); f.simplify();
				flux << " + exp(i pi " << f << ") ";
			}
			PrintBinaryDouble(flux, Y.first.b, group->N);
		}
	}
}




/**
 Prints the basis (if the dimension is small enough)
 */
std::ostream & operator<<(std::ostream &flux, const ED_basis &B)
{
	size_t n_orb = B.L;
	
	flux << "---------------------------------------------------------------------------------\n";
	flux << "Basis " << B.name << "\t(" << B.dim << " states)\n";
	if(B.dim > 1024) return flux;
	flux << "List of basis states\n";
	for(size_t i=0; i<B.dim; ++i){
		flux << i+1 << '\t';
		B.print_state(flux, i);
		flux << '\n';
	}
	return flux;
}




