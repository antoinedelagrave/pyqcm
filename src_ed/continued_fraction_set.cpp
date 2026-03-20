#include "continued_fraction_set.hpp"
#include "hdf5_io.hpp"

/**
 Constructor
 */
continued_fraction_set::continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, bool _is_complex) :
Green_function_set(_group, mixing), sec(_sec), is_complex(_is_complex)
{
  e.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  h.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  for(size_t r=0; r<group->site_irrep_dim.size(); ++r){
    size_t n = group->site_irrep_dim[r];
    if(sec.S >= sector::odd) n *= 2;
    if(sec.N >= sector::odd) n *= 2;
    e[r].set_size(n);
    h[r].set_size(n);
  }
}

/**
 Constructor from arrays
 */
continued_fraction_set::continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>> &A, const vector<vector<double>> &B, bool _is_complex) :
Green_function_set(_group, mixing), sec(_sec), is_complex(_is_complex)
{
  QCM_ASSERT(group->site_irrep_dim.size() == 2*A.size());
  QCM_ASSERT(group->site_irrep_dim.size() == 2*B.size());
  e.resize(group->site_irrep_dim.size());
  h.resize(group->site_irrep_dim.size());
  for(size_t r=0; r<group->site_irrep_dim.size(); ++r){
    size_t n = group->site_irrep_dim[r];
    if(sec.S >= sector::odd) n *= 2;
    if(sec.N >= sector::odd) n *= 2;
    e[r].set_size(n);
    h[r].set_size(n);
  }
  int j=0;
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a=0; a<nr; ++a){
      int bmax = (is_complex)? nr : a+1;
      for(size_t b=0; b<bmax; ++b){
        e[r](a, b) = continued_fraction(A[j], B[j]);
        j++;
        h[r](a, b) = continued_fraction(A[j], B[j]);
        j++;
      }
    }
  }

}



/**
 constructs the cluster Green function matrix at frequency z
 */
void continued_fraction_set::Green_function(const Complex &z, block_matrix<Complex> &G)
{
  // loop over irreps
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    if(nr==0) continue;
    vector<Complex> Gtmp(nr);

    // diagonal terms (direct frequency)
    for(size_t o1 = 0; o1 < nr; o1++){
      Gtmp[o1] = e[r](o1,o1).evaluate(z) + h[r](o1,o1).evaluate(z);
      G.block[r](o1,o1) += Gtmp[o1];
    }

    // off diagonal terms
    if(is_complex){
      for(size_t o1 = 0; o1 < nr; o1++){
        for(size_t o2 = 0; o2 <o1; o2++){
          Complex tmp1 = e[r](o1,o2).evaluate(z) +  h[r](o1,o2).evaluate(z);
          Complex tmp2 = Complex(0,1)*(e[r](o2,o1).evaluate(z) +  h[r](o2,o1).evaluate(z));
          G.block[r](o2,o1) += 0.5*(tmp1 + tmp2 - Complex(1, 1)*(Gtmp[o1] + Gtmp[o2]));
          G.block[r](o1,o2) += 0.5*(tmp1 - tmp2 - Complex(1,-1)*(Gtmp[o1] + Gtmp[o2]));
        }
      }
    }
    else{
      for(size_t o1 = 0; o1 < nr; o1++){
        for(size_t o2 = 0; o2 <o1; o2++){
          Complex tmp = 0.5*(e[r](o1,o2).evaluate(z) + h[r](o1,o2).evaluate(z) - Gtmp[o1] - Gtmp[o2]);
          G.block[r](o1,o2) += tmp;
          G.block[r](o2,o1) += tmp;
        }
      }
    }
  }
}


/**
 frequency-integrated Green function
 */
void continued_fraction_set::integrated_Green_function(block_matrix<Complex> &G)
{
  // loop over irreps
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    if(nr==0) continue;

    // diagonal terms
    for(size_t o1 = 0; o1 < nr; o1++){
      G.block[r](o1,o1) += h[r](o1,o1).b[0];
    }

    // off diagonal terms
    for(size_t o1 = 0; o1 < nr; o1++){
      for(size_t o2 = 0; o2 <o1; o2++){
        Complex tmp = 0.5*(h[r](o1,o2).b[0] - G.block[r](o1,o1) - G.block[r](o2,o2));
        G.block[r](o1,o2) += tmp;
        G.block[r](o2,o1) += tmp;
      }
    }
  }
}


/**
 Writes the continued_fraction_set to an HDF5 group.
 Layout (within grp):
   attribute "is_complex" : int
   For each (r, a, b): sub-group "r{r}_a{a}_b{b}" containing:
     - sub-group "e" with datasets "a","b" for the electron fraction
     - sub-group "h" with datasets "a","b" for the hole fraction
*/
void continued_fraction_set::write_hdf5(H5::Group& grp)
{
  h5_write_attr(grp, "is_complex", (int)is_complex);
  for(size_t r = 0; r < e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a = 0; a < nr; ++a){
      int bmax = is_complex ? (int)nr : (int)(a + 1);
      for(int b = 0; b < bmax; ++b){
        string gname = "r" + to_string(r) + "_a" + to_string(a) + "_b" + to_string(b);
        H5::Group sg = grp.createGroup(gname);
        H5::Group eg = sg.createGroup("e");
        e[r](a, (size_t)b).write_hdf5(eg);
        H5::Group hg = sg.createGroup("h");
        h[r](a, (size_t)b).write_hdf5(hg);
      }
    }
  }
}


/**
 Reads the continued_fraction_set from an HDF5 group (written by write_hdf5).
*/
void continued_fraction_set::read_hdf5(H5::Group& grp)
{
  is_complex = (bool)h5_read_attr_int(grp, "is_complex");
  for(size_t r = 0; r < e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a = 0; a < nr; ++a){
      int bmax = is_complex ? (int)nr : (int)(a + 1);
      for(int b = 0; b < bmax; ++b){
        string gname = "r" + to_string(r) + "_a" + to_string(a) + "_b" + to_string(b);
        H5::Group sg = grp.openGroup(gname);
        H5::Group eg = sg.openGroup("e");
        e[r](a, (size_t)b).read_hdf5(eg);
        H5::Group hg = sg.openGroup("h");
        h[r](a, (size_t)b).read_hdf5(hg);
      }
    }
  }
}
