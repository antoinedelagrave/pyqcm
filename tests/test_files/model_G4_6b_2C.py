import numpy as np
import pyqcm

ns = 4
nb = 6
no = ns+nb
CM = pyqcm.cluster_model(ns, nb)

CM.new_operator('eb1', 'one-body', [
    (1+ns, 1+ns, 1), (1+ns+no, 1+ns+no, 1),
    (3+ns, 3+ns, 1), (3+ns+no, 3+ns+no, 1),
    (5+ns, 5+ns, 1), (5+ns+no, 5+ns+no, 1)
])
CM.new_operator('eb2', 'one-body', [
    (2+ns, 2+ns, 1), (2+ns+no, 2+ns+no, 1),
    (4+ns, 4+ns, 1), (4+ns+no, 4+ns+no, 1),
    (6+ns, 6+ns, 1), (6+ns+no, 6+ns+no, 1)
])

CM.new_operator('tb1', 'one-body', [
    (1, 1+ns,  1), (1+no , 1+ns+no , 1),
    (2, 3+ns,  1), (2+no , 3+ns+no , 1),
    (3, 5+ns,  1), (3+no , 5+ns+no , 1)
])

CM.new_operator('tb2', 'one-body', [
    (1, 2+ns,  1), (1+no , 2+ns+no , 1),
    (2, 4+ns,  1), (2+no , 4+ns+no , 1),
    (3, 6+ns,  1), (3+no , 6+ns+no , 1)
])

clus1 = pyqcm.cluster(CM, ((-1,-1, 0), ( 0, 1, 0), ( 1, 0, 0), ( 0, 0, 0)), ( 1, 0, 0))
clus2 = pyqcm.cluster(CM, (( 1, 1, 0), ( 0,-1, 0), (-1, 0, 0), ( 0, 0, 0)), (-1, 0, 0))
model = pyqcm.lattice_model('graphene_4_2C', (clus1, clus2), ((4, 2, 0), (2, -2, 0)), ((1, -1, 0), (2, 1, 0)))
sq3 = np.sqrt(3.0)/2
model.set_basis(((1,0,0),(-0.5, sq3, 0)))
model.interaction_operator('U')
model.hopping_operator('t', (-1, 0, 0), 1, orbitals=(1,2))
model.hopping_operator('t', ( 0,-1, 0), 1, orbitals=(1,2))
model.hopping_operator('t', ( 1, 1, 0), 1, orbitals=(1,2))
