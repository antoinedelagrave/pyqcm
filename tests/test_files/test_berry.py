from pyqcm import *
from pyqcm.berry import *
import matplotlib.pyplot as plt

# ax = plt.gca()
ax = None
F = None

#-----------------------------------------------------------------
# import record_WSM
# k = np.array([0.695,0,-1e-8])
# Berry_curvature(k_perp=0.01, range=[0.5, 0, 0.5], plane='xz', file=F, plt_ax=ax)
# Berry_field_map(k_perp=0.01, plane='xy', file=F, plt_ax=ax)
# Berry_flux(k, 0.1)
# print('Berry flux : ', Berry_flux(k, 0.1))
# Berry_flux_map(nk=100, k_perp=-0.01, file=F, plt_ax=ax)
# print('Chern number : ', Chern_number())
# print('monopole at ', k, ' : ', monopole(k, a=0.05, nk=40))
# monopole_map(plane='xy', k_perp=0.0, file=F, plt_ax=ax)

# plt.show()
##################################################################
# TEST UNITAIRE

def test_berry():
    x = None
    import record_WSM
    lab = 9
    k = np.array([0.695,0,-1e-8])

    F = 'test_Berry_curvature.pdf'
    banner('testing Berry_curvature()', c='#', skip=1); 
    Berry_curvature(nk=20, k_perp=0.01, range=[0.5, 0, 0.5], plane='xz', label=lab, file=F, plt_ax=ax)

    F = 'test_Berry_field_map.pdf'
    banner('testing Berry_field_map()', c='#', skip=1); 
    Berry_field_map(nk=20, k_perp=0.01, label=lab, plane='xy', file=F, plt_ax=ax)

    banner('testing Berry_flux()', c='#', skip=1); 
    print('Berry flux : ', Berry_flux(k, 0.1, label=lab))
    
    F = 'test_Berry_flux_map.pdf'
    banner('testing Berry_flux_map()', c='#', skip=1); 
    Berry_flux_map(nk=20, k_perp=-0.01, label=lab, file=F, plt_ax=ax)
    
    banner('testing Chern_number()', c='#', skip=1); 
    print('Chern number : ', Chern_number(label=lab))
    
    banner('testing monopole()', c='#', skip=1); 
    print('monopole at ', k, ' : ', monopole(k, a=0.05, nk=20, label=lab))
    
    F = 'test_monopole_map.pdf'
    banner('testing monopole_map()', c='#', skip=1); 
    monopole_map(nk=20, plane='xy', k_perp=0.0, label=lab, file=F, plt_ax=ax)


test_berry()