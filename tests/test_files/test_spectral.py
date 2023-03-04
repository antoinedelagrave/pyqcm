from pyqcm import *
from pyqcm.spectral import *
import matplotlib.pyplot as plt

ax = None
F = None
# F = None
#-----------------------------------------------------------------
# import record_2x2_anom
# import record_spin

# DoS(w=4, eta=0.2, file = F, plt_ax=ax)
# Fermi_surface(file = F, plt_ax=ax)
# G_dispersion(max=20, contour=True, file = F, plt_ax=ax)
# Luttinger_surface(file = F, plt_ax=ax)
# cluster_spectral_function(file = F, plt_ax=ax)
# dispersion(file = F, contour=False, plt_ax=ax, labels=True)
# mdc(file = F, plt_ax=ax)
# momentum_profile('t', file = F, plt_ax=ax)
# segment_dispersion(file = F, plt_ax=ax)
# spectral_function(file = F, plt_ax=ax)
# spectral_function_Lehmann(lims=(-5,5), file = F, plt_ax=ax)
# spin_mdc(nk=20, orb=1, file = F, plt_ax=ax, opt='spins')
# spin_mdc(nk=200, orb=1, file = F, plt_ax=ax, opt='spinp')
# spin_mdc(nk=200, orb=1, file = F, plt_ax=ax, opt='sz')
# spin_mdc(nk=25, orb=1, file = F, plt_ax=ax, opt='spins')
# mdc_anomalous(orbs=(1,1), self=True, quadrant=True, file = F, plt_ax=ax)

#-----------------------------------------------------------------
# import record_2x2_8b

# hybridization_function(file = F, plt_ax=ax)

#-----------------------------------------------------------------
# import record_2x2_anom

# plt.show()


##################################################################
# TEST UNITAIRE
Lab = 9
def test_spectral():
    x = None
    import record_2x2_anom
    read_cluster_model_instance(record_2x2_anom.solution[0], 9)

    F = 'test_DoS.pdf'
    banner('testing DoS()', c='#', skip=1); DoS(w=1, eta=0.2, label = Lab, file = F, plt_ax=ax)
    
    F = 'test_G_dispersion.pdf'
    banner('testing G_dispersion()', c='#', skip=1); G_dispersion(max=20, label = Lab, file = F, plt_ax=ax)
    
    F = 'test_Luttinger_surface.pdf'
    banner('testing Luttinger_surface()', c='#', skip=1); Luttinger_surface(label = Lab, file = F, plt_ax=ax)
    
    F = 'test_cluster_spectral_function.pdf'
    banner('testing cluster_spectral_function()', c='#', skip=1); cluster_spectral_function(label = Lab, file = F, plt_ax=ax)
    
    F = 'test_mdc.pdf'
    banner('testing mdc()', c='#', skip=1); mdc(nk = 50, label = Lab, file = F, plt_ax=ax)
    
    F = 'test_momentum_profile.pdf'
    banner('testing momentum_profile()', c='#', skip=1); momentum_profile('t', nk=10, label = Lab, file = F, plt_ax=ax)
    
    F = 'test_spectral_function.pdf'
    banner('testing spectral_function()', c='#', skip=1); spectral_function(label = Lab, nk=8, file = F, plt_ax=ax)
    
    F = 'test_spectral_function_Lehmann.pdf'
    banner('testing spectral_function_Lehmann()', c='#', skip=1); spectral_function_Lehmann(lims=(-5,5), nk=8, label = Lab, file = F, plt_ax=ax)
    
    F = 'test_spin_mdc.pdf'
    banner('testing spin_mdc()', c='#', skip=1); spin_mdc(nk=40, opt='spins', label = Lab, file = F, plt_ax=ax)

    F = 'test_hybridization_function.pdf'
    banner('testing hybridization_function()', c='#', skip=1); hybridization_function(label = Lab, file = F, plt_ax=ax)

    F = 'test_mdc_anomalous.pdf'
    banner('testing mdc_anomalous()', c='#', skip=1); mdc_anomalous(nk=40, label = Lab, file = F, plt_ax=ax)

    F = 'test_Fermi_surface.pdf'
    banner('testing Fermi_surface()', c='#', skip=1); Fermi_surface(label = Lab, file = F, plt_ax=ax)
    
    F = 'test_dispersion.pdf'
    banner('testing dispersion()', c='#', skip=1); plot_dispersion(label = Lab, file = F, plt_ax=ax)
    
    F = 'test_dispersionC.pdf'
    banner('testing dispersion() with contours', c='#', skip=1); plot_dispersion(label = Lab, contour=True, file = F, plt_ax=ax)

    F = 'test_segment_dispersion.pdf'
    banner('testing segment_dispersion()', c='#', skip=1); segment_dispersion(label = Lab, file = F, plt_ax=ax)
    

test_spectral()