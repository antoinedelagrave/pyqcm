import pyqcm
from pyqcm.cdmft import CDMFT
import numpy as np

import model_1D_2_4b_2b as M

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
M.model.set_target_sectors(['N6:S0', 'N6:S0', 'N2:S0'])

# Simulation parameters
M.model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    ty = 0.5
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
    tb1_2=0.5
    tb2_2=0.4
    eb1_2=1.0
    eb2_2=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1', 'eb1_2', 'eb2_2', 'tb1_2', 'tb2_2']

X = CDMFT(M.model, varia=varia, wc=[0.5,5,5], grid_type='legendre', accur=1e-3, convergence='self-energy', miniter=1, maxiter=64, depth=1, iteration='fixed_point')


X.I.write_all_hdf5('test.h5')
print('GF written:\n', X.I.cluster_Green_function(0.1j, 1))
I = pyqcm.model_instance(M.model)
I.read_all_hdf5('test.h5')
print('\nGF read:\n', I.cluster_Green_function(0.1j, 1))

