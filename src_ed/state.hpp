#ifndef state_h
#define state_h

#include <cstdio>

#include "sector.hpp"
#include "Q_matrix_set.hpp"
#include "continued_fraction_set.hpp"
#include "mcf_set.hpp"
#include "parser.hpp"
#include "ED_basis.hpp"

//! state (e.g. the ground state) from which a contribution to the Green function is computed
template<typename HilbertField>
struct state
{
	sector sec; //!< sector to which the state belongs
	vector<HilbertField> psi; 	//!< the Hilbert-space vector representing the state
	double energy; //!< energy of the state
	double weight; //!< its weight in the density matrix
	
  shared_ptr<Green_function_set> gf;
  shared_ptr<Green_function_set> gf_down;
  
  /**
   default constructor
   */
  state() : sec("R0") {}

  
  
  /**
	 constructor from sector and dimension
	 */
	state(sector _sec, size_t dim) : sec(_sec)
	{
		psi.resize(dim);
	}
  
  
  /**
   writes state to an HDF5 group.
   Layout: attributes "sec","energy","weight","gf_format";
           sub-group "gf"; optionally sub-group "gf_down".
   The GF format is deduced from the runtime type of the gf pointer.
  */
  void write_hdf5(H5::Group& grp)
  {
    h5_write_attr(grp, "sec",    sec.name());
    h5_write_attr(grp, "energy", energy);
    h5_write_attr(grp, "weight", weight);
    // Deduce the GF format from the runtime type of gf
    string fmt = "bl";
    if(gf != nullptr){
      if(dynamic_pointer_cast<continued_fraction_set>(gf) != nullptr) fmt = "cf";
      else if(dynamic_pointer_cast<mcf_set<HilbertField>>(gf) != nullptr) fmt = "mcf";
    }
    h5_write_attr(grp, "gf_format", fmt);
    if(gf != nullptr){
      H5::Group gg = grp.createGroup("gf");
      gf->write_hdf5(gg);
    }
    if(gf_down != nullptr){
      H5::Group gg = grp.createGroup("gf_down");
      gf_down->write_hdf5(gg);
    }
  }


  /**
   constructs state from an HDF5 group (written by write_hdf5).
   @param grp     HDF5 group containing the state data
   @param group   symmetry group (needed to construct GF representations)
   @param mixing  mixing state
  */
  state(H5::Group& grp, shared_ptr<symmetry_group> sym_group, int mixing)
  {
    sec    = sector(h5_read_attr_string(grp, "sec"));
    energy = h5_read_attr_double(grp, "energy");
    weight = h5_read_attr_double(grp, "weight");
    string fmt = h5_read_attr_string(grp, "gf_format");
    GF_FORMAT gf_fmt;
    if(fmt == "bl")       gf_fmt = GF_format_BL;
    else if(fmt == "mcf") gf_fmt = GF_format_MCF;
    else                  gf_fmt = GF_format_CF;

    bool is_complex = (typeid(HilbertField) == typeid(Complex));

    auto read_gf = [&](const string& gname) -> shared_ptr<Green_function_set> {
      H5::Group gg = grp.openGroup(gname);
      if(gf_fmt == GF_format_BL){
        auto qs = make_shared<Q_matrix_set<HilbertField>>(sym_group, mixing);
        qs->read_hdf5(gg);
        return qs;
      }
      else if(gf_fmt == GF_format_MCF){
        auto ms = make_shared<mcf_set<HilbertField>>(sym_group, mixing);
        ms->read_hdf5(gg);
        return ms;
      }
      else {
        auto cfs = make_shared<continued_fraction_set>(sec, sym_group, mixing, is_complex);
        cfs->read_hdf5(gg);
        return cfs;
      }
    };

    if(grp.nameExists("gf"))      gf      = read_gf("gf");
    if(grp.nameExists("gf_down")) gf_down = read_gf("gf_down");
  }


  /**
   writing the wavefunction to an ASCII file
   */
  void write_wavefunction(ostream& fout, const ED_basis &B)
  {
    fout << "state\n" << sec << '\t' << energy << '\t' << weight << endl;
    if(B.dim <= global_int("max_dim_print")){
      for(int i=0; i<B.dim; i++){
        fout << abs(psi[i])*abs(psi[i]) << '\t' << psi[i] << '\t';
        B.print_state(fout,i);
        fout << '\n';
      }
    }
  }


};

template<typename HilbertField>
std::ostream& operator<<(std::ostream &flux, const state<HilbertField> &x)
{
  flux << "E = " << x.energy << " (" << x.sec << ") weight = " << x.weight << endl;
  return flux;
}



#endif

