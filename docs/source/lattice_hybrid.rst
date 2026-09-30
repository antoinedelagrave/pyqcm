######################
Lattice hybridizations
######################

When using **pyqcm** in the context of *ab initio* computations, the set of uncorrelated orbitals affect the correlated orbitals through a lattice hybridization function :math:`\Gamma(\mathbf k,\omega)`  that depends both on wavevector and frequency. This lattice hybridization will appear in the correlated subspace Green function as added to the self-energy and to the bath hybridization used in DMFT. The information contained in :math:`\Gamma(\mathbf k,\omega)` is numerical, and must be stored on a grid of frequencies and wavevectors, in HDF5 format. 

When defining a ``lattice_model`` object, pyqcm can be told (via the optional argument ``hybrid_file``) to read such a file.
When that happens, it signals *pyqcm* that Green functions can only be computed on this grid, not at an arbitrary frequency or wavevector. It can then perform CDMFT or compute lattice averages, as usual.

File structure
==============

The frequency grid is independent of the wavevector grid; in other words, the full grid is a Cartesian product of a frequency grid (:math:`N_\omega` frequencies) and of a wavevector grid (:math:`M` wavevectors). The HDF5 file contains the following datasets:

* ``w`` : a real array of :math:`N_\omega` frequencies. They lie on the imaginary axis (the frequency is :math:`i\omega_n`), unless ``eta`` is nonzero (see below).
* ``weight`` : a real array of :math:`N_\omega` weights :math:`W_n` associated with the frequencies. Lattice averages are computed as :math:`\sum_n W_n \frac1M\sum_{\mathbf k}`, i.e., with the given frequency weights and a uniform average over the wavevectors.
* ``k`` : a real :math:`M\times3` array of wavevectors, expressed in the basis of the reciprocal superlattice (i.e., in units of the vectors dual to the superlattice vectors, typically in the interval :math:`[0,1)`), **not** in the physical basis. The two coincide only when the super unit cell is the unit cell. Since the wavevector sums are uniform averages, the grid should sample the reduced Brillouin zone uniformly.
* ``hybrid_real`` and ``hybrid_imag`` : two real arrays of shape :math:`N_\omega\times M\times d\times d` containing the real and imaginary parts of :math:`\Gamma(\mathbf k,\omega)`: the element ``[n, m, a, b]`` is :math:`\Gamma_{ab}(\mathbf k_m,\omega_n)`.
* ``mixing`` : an integer, the mixing state of :math:`\Gamma` (0 : normal; 1 : simple Nambu; 2 : spin-flip), which determines its dimension :math:`d` (see below).
* ``eta`` (optional) : a real number. If absent or zero, the frequencies ``w`` lie on the imaginary axis. Otherwise they are real frequencies and the actual frequency is :math:`\omega+i\eta`. This only applies to the evaluation of the Green function on the grid (e.g. :func:`~pyqcm.model_instance.CPT_Green_function_grid` or :func:`~pyqcm.model_instance.spectral_function`); lattice averages and the CDMFT host function require the imaginary axis, and raise an error if ``eta`` is nonzero.

The indices :math:`a,b` of :math:`\Gamma` follow the ordering of the physical orbitals of the super unit cell described in :doc:`models` (bath orbitals excluded). The mixing states that can be combined are the following:

* ``mixing = 0`` : :math:`\Gamma` is defined in the normal state, and :math:`d` is the number of physical orbitals :math:`N_s` in the super unit cell. The model itself may be in the normal state, in the simple Nambu mixing or in the spin-flip mixing; :math:`\Gamma` is then upgraded automatically (Nambu doubling or spin doubling).
* ``mixing = 1`` : :math:`\Gamma` is already defined in the simple Nambu mixing, :math:`d=2N_s`, and the model must also be in the simple Nambu mixing.
* ``mixing = 2`` : :math:`\Gamma` is already defined in the spin-flip mixing (e.g. because of spin-orbit couplings in the lattice environment), :math:`d=2N_s`, and the model must also be in the spin-flip mixing.

Other combinations (in particular the full Nambu mixing and the spin asymmetric mixing) are not currently supported, and are rejected with an error when the file is read.

The function ``write_hybrid_file()`` of the module ``pyqcm.TRIQS_converter`` writes a file in this format from numpy arrays.

Updating the lattice hybridization
==================================

The lattice hybridization can be replaced at any time by that contained in another file, with the ``lattice_model``
member function ``update_extern_hybrid(filename)``, for instance in a loop in which :math:`\Gamma(\mathbf k,\omega)`
is recomputed by an external code::

    model = pyqcm.lattice_model('model', clus, superlattice, lattice, hybrid_file='hybrid_0.h5')
    ...
    model.update_extern_hybrid('hybrid_1.h5')

The new file has the same structure as the original one, but its frequency and wavevector grids may be different.
The same function can also add a lattice hybridization to a model that was defined without one.
The file is read and checked immediately (unless the model has not yet been used, in which case it is read when the
first model instance is created); if it is invalid (e.g. if the dimension of the matrices is wrong), an error is
raised and the current lattice hybridization is kept.

Contrary to an operator update (see :ref:`updating operators`), existing model instances remain valid after the
update: the cluster solutions do not depend on the lattice hybridization and are kept, whereas the quantities that
depend on it (lattice averages, CDMFT host function, etc.) are forgotten and recomputed with the new lattice
hybridization when needed. Note that the frequency grid used in a CDMFT computation must be that of the current
lattice hybridization file.

This feature has not been fully tested yet, do beware.