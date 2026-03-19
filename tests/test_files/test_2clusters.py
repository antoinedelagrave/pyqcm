import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT

from model_1D_4 import model

pyqcm.set_global_parameter('GF_method', 'M')

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors('N4:S0')

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
""")

I = pyqcm.model_instance(model)  
print('\ncluster Green function at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=0, spin_down=False, blocks=False))
