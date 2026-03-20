import pyqcm
import numpy as np

ns = 2
nb = 4
no = ns+nb
CM = pyqcm.cluster_model(ns, nb)
CM2 = pyqcm.cluster_model(2, 0, name='clus2')

CM.new_operator('tb1', 'one-body', [
    (1, 3, -1.0),
    (2, 4, -1.0),
    (7, 9, -1.0),
    (8, 10, -1.0)
])

CM.new_operator('tb2', 'one-body', [
    (1, 5, -1.0),
    (2, 6, -1.0),
    (7, 11, -1.0),
    (8, 12, -1.0)
])

CM.new_operator('eb1', 'one-body', [
    (3, 3, 1.0),
    (4, 4, 1.0),
    (9, 9, 1.0),
    (10, 10, 1.0)
])

CM.new_operator('eb2', 'one-body', [
    (5, 5, 1.0),
    (6, 6, 1.0),
    (11, 11, 1.0),
    (12, 12, 1.0)
])


#-------------------------------------------------------------------
# construction of the lattice model 

clus1 = pyqcm.cluster((CM, CM), (( 0, 0, 0), ( 1, 0, 0)))
clus2 = pyqcm.cluster(CM2, (( 0, 0, 0), ( 1, 0, 0)), (0,1,0))
model = pyqcm.lattice_model('1D_2_4b_2sb', (clus1, clus2), ((2, 0, 0),))

model.interaction_operator('U')
model.hopping_operator('t', ( 1, 0, 0), -1)
model.hopping_operator('ty', ( 1, 0, 0), -1)
