import numpy as np
import pyqcm
import matplotlib.pyplot as plt
import model_2x2_C2 as M

I = pyqcm.model_instance(M.model)

ax = None
F = None
# F = None
#-----------------------------------------------------------------
# import record_2x2_anom
# import record_spin

# I.plot_dispersion(); exit()
##################################################################
# TEST UNITAIRE

K = np.array([(1,1,0.)])
print('testing dispersion()')
print(I.dispersion(K))

print('testing epsilon()')
print(I.epsilon(K))
