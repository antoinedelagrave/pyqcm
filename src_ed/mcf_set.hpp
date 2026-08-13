#ifndef mcf_set_h
#define mcf_set_h

/**
 * @file mcf_set.hpp
 * @brief Definition of the mcf_set template (set of matrix-valued continued fractions).
 */

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

 Both the Q_matrix_set (VDVH kernel, with v.cconjugate() on the electron
 eigenvectors) and the continued_fraction_set produce the same output:

   G_output(a,b) = G⁺(a,b) + G⁻(b,a) = G⁺(a,b) + (G⁻)ᵀ(a,b)

 where G⁺ and G⁻ are the physical electron and hole Green functions.

 The MCF evaluate() gives directly:
   e[r].evaluate(z)(a,b) = G⁺(a,b)   (no transformation needed)
   h[r].evaluate(z)(a,b) = G⁻(a,b)   (must be transposed before adding)

 Consequently:
 - Electron part: G.block[r] +=           e[r].evaluate(z)
 - Hole part:     G.block[r] += TRANSPOSE(h[r].evaluate(z))

 For real Hamiltonians (T = double) both G⁺ and G⁻ are symmetric, so the
 transpose is a no-op and the distinction does not matter.

 ### Integrated Green function

 integrated_Green_function() does NOT assume that every pole of the hole MCF
 lies at negative energy (that assumption only holds when the reference state
 is the/a true ground state; it can fail for an excited, thermally-weighted
 reference state at finite temperature). Instead it diagonalizes the
 block-tridiagonal projected Hamiltonian of h[r] (or of combined[r] when the
 combine_mcf path is used) and sums only the negative-eigenvalue poles — see
 matrix_continued_fraction::integrated_Green_function(). The result is
 transposed before being added to G.block[r], following the same convention
 as Green_function()'s hole-part transpose; combined[r] already includes the
 transpose internally (see combine_via_lanczos), so no further transpose is
 applied in that case.
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
     of build_mcf).

     When the global option "combine_mcf" is true, the electron and hole MCFs
     are combined into a single MCF via a new block Lanczos run on the
     direct-sum operator T_e ⊕ T_h (combine_via_lanczos), and stored in
     combined[r].  Green_function() then evaluates combined[r] alone.

     Otherwise (default), combined[r] is left empty and Green_function()
     evaluates e[r] and h[r] separately, adding G_h + G_e^T to G.block[r].
    */
    void build_combined()
    {
        if(!global_bool("combine_mcf")) return;
        for(size_t r = 0; r < group->g; ++r){
            bool has_e = e[r].floors() > 0;
            bool has_h = h[r].floors() > 0;
            if(has_e && has_h){
                int M0 = e[r].floors() + h[r].floors();
                combined[r] = combine_via_lanczos(e[r], h[r], M0);
            }
            // single-sector blocks: combined stays empty; Green_function handles them.
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

 When "combine_mcf" is true, combined[r] holds a single MCF for G_h + G_e
 and is evaluated directly.

 Otherwise (default), the electron and hole MCFs are evaluated separately:
 the electron part is added directly (e[r].evaluate(z) = G⁺);
 the hole part is transposed before addition (h[r].evaluate(z) = G⁻,
 and the output convention requires G⁻ transposed = (G⁻)ᵀ).
*/
template<typename T>
inline void mcf_set<T>::Green_function(const Complex &z, block_matrix<Complex> &G)
{
    if(global_bool("combine_mcf")){
        for(size_t r = 0; r < group->g; ++r)
            if(combined[r].floors() > 0)
                G.block[r] += combined[r].evaluate(z);
        return;
    }
    for(size_t r = 0; r < group->g; ++r){
        if(e[r].floors() > 0)
            G.block[r] += e[r].evaluate(z);
        if(h[r].floors() > 0){
            matrix<Complex> Gh = h[r].evaluate(z);
            Gh.transpose();
            G.block[r] += Gh;
        }
    }
}


/**
 Computes the frequency-integrated Green function (occupation matrix).

 Per irrep block: if combined[r] is populated (combine_mcf path, or the
 Q_matrix-to-MCF conversion which only ever fills combined[r]), its poles
 already merge the electron and hole channels in the final output
 convention (see combine_via_lanczos), so its filtered integral is added
 directly, without transposition. Otherwise, the hole MCF h[r] is used: its
 filtered integral (only the negative-eigenvalue poles, see
 matrix_continued_fraction::integrated_Green_function()) is transposed
 before being added, matching Green_function()'s hole-part convention.
*/
template<typename T>
inline void mcf_set<T>::integrated_Green_function(block_matrix<Complex> &G)
{
    for(size_t r = 0; r < group->g; ++r) {
        if(combined[r].floors() > 0){
            G.block[r] += combined[r].integrated_Green_function();
        }
        else if(h[r].floors() > 0 && h[r].p > 0){
            matrix<Complex> Gi = h[r].integrated_Green_function();
            Gi.transpose();
            G.block[r] += Gi;
        }
    }
}


/**
 Writes the mcf_set to an HDF5 group.
 Layout: attribute "nblocks"; for each r, sub-group "block_r" containing "e"
 and "h", plus "combined" when combined[r] is populated (combine_mcf path,
 including the Q_matrix-to-MCF conversion which only fills combined[r]).
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
        if(combined[r].floors() > 0){
            H5::Group cg = bg.createGroup("combined");
            h5_write_mcf(cg, combined[r]);
        }
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
    combined.resize(nblocks);
    for(int r = 0; r < nblocks; ++r){
        H5::Group bg = grp.openGroup("block_" + to_string(r));
        H5::Group eg = bg.openGroup("e");
        h5_read_mcf(eg, e[r]);
        H5::Group hg = bg.openGroup("h");
        h5_read_mcf(hg, h[r]);
        if(bg.nameExists("combined")){
            H5::Group cg = bg.openGroup("combined");
            h5_read_mcf(cg, combined[r]);
        }
    }
}


#endif /* mcf_set_h */
