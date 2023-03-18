import pyqcm
from pyqcm.cdmft import *

ns = 8
CM = pyqcm.cluster_model(ns)  # call to constructor of class "cluster_model"
clus = pyqcm.cluster(CM, [(i,0,0) for i in range(ns)]) # call to constructor of class "cluster"
model = pyqcm.lattice_model('1D_8', clus, ((ns,0,0),)) # call to constructor of class "lattice_model"

model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)

############# model definition end #################

model.set_target_sectors(['R0:N8:S0'])
model.set_parameters("""
U = 1e-9
mu = 0.5*U
t = 1
""")

I1 = pyqcm.model_instance(model) # call to constructor of class "model_instance"
for u in np.arange(1e-9, 16.1, 2):
    model.set_parameter('U', u)
    I2 = pyqcm.model_instance(model, label = 1) # call to constructor of class "model_instance"
    print('U = ', u, '\tfidelity = ', model.fidelity(I1,I2))
