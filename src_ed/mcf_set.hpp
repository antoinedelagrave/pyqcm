#ifndef mcf_set_h
#define mcf_set_h

#include "Green_function_set.hpp"
#include "matrix_continued_fraction.hpp"
#include "global_parameter.hpp"

/**
 @brief Set of matrix-valued continued fractions for the full Green function.

 Stores one electron MCF (pm=+1, creation) and one hole MCF (pm=-1,
 annihilation) per irreducible representation block.  These are obtained from
 the block Lanczos method (blockLanczos) applied to the block of starting
 vectors phi[i] = c_i^†|GS> (electron) or phi[i] = c_i|GS> (hole).

 Template parameter T is the Hilbert-space field (double or Complex).

 ### Conventions (matching Q_matrix_set / continued_fraction_set)

 The Q_matrix_set applies v.cconjugate() to the electron eigenvectors after
 the band Lanczos, which effectively transposes the electron contribution to
 the Green function:
   G_Q^+(a,b) = sum_n conj(<phi_a|n>) <phi_b|n> / (z - e_n)
              = [W^H F_0(z) W]^T_{a,b}        (where F_0 is the block resolvent)

 The hole contribution is NOT conjugated, so:
   G_Q^-(a,b) = sum_n <phi_a|n> conj(<phi_b|n>) / (z - e_n)
              = [W^H F_0(z) W]_{a,b}

 Consequently:
 - Hole part:     G.block[r] +=      h[r].evaluate(z)
 - Electron part: G.block[r] += TRANSPOSE(e[r].evaluate(z))

 For real Hamiltonians (T = double) the GF is symmetric, so the
 transpose is a no-op.

 ### Integrated Green function

 For the hole MCF all poles lie at negative energies, so the spectral integral
 equals W_h^H W_h (the squared weight matrix, which includes the sqrt of the
 state weight Omega.weight).  Following the same G(b,a) convention as
 Q_matrix::integrated_Green_function, we add (W_h^H W_h)^T to G.block[r].
*/
template<typename T>
struct mcf_set : Green_function_set
{
    vector<matrix_continued_fraction<T>> e;        //!< electron MCFs (one per irrep)
    vector<matrix_continued_fraction<T>> h;        //!< hole MCFs (one per irrep)
    vector<matrix_continued_fraction<T>> combined; //!< pre-built combined MCFs (G_h + G_e^T)

    //! Constructor: allocates empty MCFs for each irrep.
    mcf_set(shared_ptr<symmetry_group> _group, int mixing)
        : Green_function_set(_group, mixing)
    {
        e.resize(group->g);
        h.resize(group->g);
        combined.resize(group->g);
    }

    //! Build combined MCFs from the electron and hole MCFs.
    /**
     Must be called after all e[r] and h[r] have been filled (e.g. at the end
     of build_mcf).  For each irrep block r both sectors must be present; if
     only one is non-empty the combined MCF defaults to that sector alone.

     When the global option "combine_mcf" is true, the combined MCF is obtained
     via a new block Lanczos run on the direct-sum operator T_e ⊕ T_h
     (combine_via_lanczos).  Otherwise (default) the electron and hole MCFs are
     combined analytically via a non-square weight matrix (combine_for_gf).
    */
    void build_combined()
    {
        bool use_lanczos = global_bool("combine_mcf");
        for(size_t r = 0; r < group->g; ++r){
            bool has_e = e[r].floors() > 0;
            bool has_h = h[r].floors() > 0;
            if(has_e && has_h){
                if(use_lanczos){
                    int M0 = max(e[r].floors(), h[r].floors());
                    combined[r] = combine_via_lanczos(e[r], h[r], M0);
                } else {
                    combined[r] = combine_for_gf(e[r], h[r]);
                }
            } else if(has_h){
                combined[r] = h[r];          // hole only: W is already correct
            } else if(has_e){
                // Electron only: conjugate W_e so evaluate() returns G_e^T
                auto tmp = e[r];
                for(size_t k = 0; k < tmp.W.v.size(); ++k)
                    tmp.W.v[k] = conj(tmp.W.v[k]);
                combined[r] = tmp;
            }
        }
    }

    // Virtual method implementations
    void Green_function(const Complex &z, block_matrix<Complex> &G) override;
    void integrated_Green_function(block_matrix<Complex> &G) override;
    void write_hdf5(H5::Group& grp) override;
    void read_hdf5(H5::Group& grp) override;
};


//==============================================================================
// Inline implementations


/**
 Evaluates the matrix continued fraction Green function at frequency z
 and adds the result to G.

 The hole part is added directly.  The electron part is TRANSPOSED before
 addition (see class-level convention note).
*/
template<typename T>
inline void mcf_set<T>::Green_function(const Complex &z, block_matrix<Complex> &G)
{
    for(size_t r = 0; r < group->g; ++r) {
        if(combined[r].floors() > 0)
            G.block[r] += combined[r].evaluate(z);
    }
}


/**
 Computes the frequency-integrated Green function (occupation matrix).

 For the hole MCF all poles are at negative energies.  The exact spectral
 weight is W_h^H W_h.  Following the Q_matrix convention G(b,a) += M(a,b),
 we add (W_h^H W_h)^T to G.block[r].
*/
template<typename T>
inline void mcf_set<T>::integrated_Green_function(block_matrix<Complex> &G)
{
    for(size_t r = 0; r < group->g; ++r) {
        if(h[r].floors() == 0 || h[r].p == 0) continue;
        int p = h[r].p;
        matrix<Complex> Wh = h[r].W;
        Wh.hermitian_conjugate();
        matrix<Complex> M(p);
        M.product(Wh, h[r].W);  // M = W^H * W
        M.transpose();           // (W^H W)^T: cf. G(b,a) += M(a,b) convention
        G.block[r] += M;
    }
}


/**
 Writes the mcf_set to an HDF5 group.
 Layout: attribute "nblocks"; for each r, sub-group "block_r" containing "e" and "h".
*/
template<typename T>
inline void mcf_set<T>::write_hdf5(H5::Group& grp)
{
    h5_write_attr(grp, "nblocks", (int)group->g);
    for(size_t r = 0; r < group->g; ++r){
        H5::Group bg = grp.createGroup("block_" + to_string(r));
        H5::Group eg = bg.createGroup("e");
        h5_write_mcf(eg, e[r]);
        H5::Group hg = bg.createGroup("h");
        h5_write_mcf(hg, h[r]);
    }
}


/**
 Reads the mcf_set from an HDF5 group written by write_hdf5.
*/
template<typename T>
inline void mcf_set<T>::read_hdf5(H5::Group& grp)
{
    int nblocks = h5_read_attr_int(grp, "nblocks");
    e.resize(nblocks);
    h.resize(nblocks);
    for(int r = 0; r < nblocks; ++r){
        H5::Group bg = grp.openGroup("block_" + to_string(r));
        H5::Group eg = bg.openGroup("e");
        h5_read_mcf(eg, e[r]);
        H5::Group hg = bg.openGroup("h");
        h5_read_mcf(hg, h[r]);
    }
}


#endif /* mcf_set_h */
