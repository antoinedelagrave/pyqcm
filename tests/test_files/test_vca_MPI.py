import pyqcm
import pyqcm.vca as V
import matplotlib.pyplot as plt
import numpy as np

# ax = plt.gca()
ax = None
F = None

# print(dir(pyqcm.vca)); exit()
#-----------------------------------------------------------------

# plt.show()
##################################################################
# TEST UNITAIRE

def test_vca():
    x = None
    import model_2x2_C2_vca as M

    F = 'test_vca.pdf'

    pyqcm.banner('testing vca()', c='#', skip=1)
    vca = V.VCA(M.model, varia=['M_1', 't_1'], start=(0.3, 1.3), accur=1e-3, steps=[5e-5, 5e-5], max=[10,10], method='SYMR1')


test_vca()

# transition
# transition_line
# vca_min