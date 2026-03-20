######################
Lattice hybridizations
######################

When using **pyqcm** in the context of *ab initio* computations, the set of uncorrelated orbitals affect the correlated orbitals through a lattice hybridization function :math:`\Gamma(\mathbf k,\omega)`  that depends both on wavevector and frequency. This lattice hybridization will appear in the correlated subspace Green function as added to the self-energy and to the bath hybridization used in DMFT. The information contained in :math:`\Gamma(\mathbf k,\omega)` is numerical, and must be stored on a grid of frequencies and wavevectors, in HDF5 format. 

When defining a ``lattice_model``  object, pyqcm can be told (via an optional argument) to read such a file. 
When that happens, it signals *pyqcm* that Green functions can only computed on this grid, not at an arbitrary frequency of wavevector. It can then perform CDMFT or computed lattice averages, as usual. The lattice hybridization function can only be in a lower mixing state than the problem defined in *pyqcm*. In particular, it can be defined in the normal state, whereas the impurity problem can be defined in the Nambu or the spin-flip mixing or both. On the other hand, if the lattice  hybridization function is already in the spin-flip mixing (because of spin orbit couplings in the lattice environment), then the impurity problem must have at least the correspond mixing state, either spin-flip of full Nambu mixing.

The frequency grid is independent of the momentum grid; in other words, the full grid is a Cartesian product of a frequency grid and of a wavevector grid. The structure of the HDF file is the following:

* A real array ``w`` of frequencies (defined along the imaginary axis)
* A real array ``weight`` of weights :math:`W_n` associated with the frequencies
* An array ``k`` of wavevectors, *defined in the unit cube* (hence in multiples of :math:`2\pi`). This is a :math:`M\times3` array of reals, :math:`M` being the number of wavevectors.
* An array ``hybrid_real`` (:math:`M\times d\times d`) containing the real part of :math:`\Gamma(\mathbf k,\omega)` 
* An array ``hybrid_imag`` (:math:`M\times d\times d`) containing the imaginary part of :math:`\Gamma(\mathbf k,\omega)` 

This feature has not been fully tested yet, do beware.