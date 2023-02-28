from pyqcm import *

ns = 2
nb = 4
no = ns+nb
new_cluster_model('L2_4b', ns, nb)

new_cluster_operator('L2_4b', 'tb1', 'one-body', [
    (1, 3, -1.0),
    (2, 4, -1.0),
    (7, 9, -1.0),
    (8, 10, -1.0)
])

new_cluster_operator('L2_4b', 'tb2', 'one-body', [
    (1, 5, -1.0),
    (2, 6, -1.0),
    (7, 11, -1.0),
    (8, 12, -1.0)
])

new_cluster_operator('L2_4b', 'eb1', 'one-body', [
    (3, 3, 1.0),
    (4, 4, 1.0),
    (9, 9, 1.0),
    (10, 10, 1.0)
])

new_cluster_operator('L2_4b', 'eb2', 'one-body', [
    (5, 5, 1.0),
    (6, 6, 1.0),
    (11, 11, 1.0),
    (12, 12, 1.0)
])

new_cluster_operator('L2_4b', 'sb1', 'anomalous', [
    (1, 3+no, -1.0),
    (2, 4+no, -1.0),
    (3, 1+no, 1.0),
    (4, 2+no, 1.0)
])

new_cluster_operator('L2_4b', 'sb2', 'anomalous', [
    (1, 5+no, -1.0),
    (2, 6+no, -1.0),
    (5, 1+no, 1.0),
    (6, 2+no, 1.0)
])

new_cluster_operator_complex('L2_4b', 'sbi1', 'anomalous', [
    (1, 3+no, -1.0j),
    (2, 4+no, -1.0j),
    (3, 1+no, 1.0j),
    (4, 2+no, 1.0j)
])

new_cluster_operator_complex('L2_4b', 'sbi2', 'anomalous', [
    (1, 5+no, -1.0j),
    (2, 6+no, -1.0j),
    (5, 1+no, 1.0j),
    (6, 2+no, 1.0j)
])

new_cluster_operator('L2_4b', 'pb1', 'anomalous', [
    (3, 4+no, 1.0),
    (4, 3+no, -1.0),
    (5, 6+no, 1.0),
    (6, 5+no, -1.0)
])

#-------------------------------------------------------------------
# construction of the lattice model 

add_cluster('L2_4b', ( 0, 0, 0), [( 0, 0, 0), ( 1, 0, 0)])
lattice_model('1D_2_4b', [( 2, 0, 0)])

interaction_operator('U')
hopping_operator('t', ( 1, 0, 0), -1) # NN hopping
hopping_operator('ti', ( 1, 0, 0), -1, tau=2) # NN hopping with imaginary amplitude
hopping_operator('tp', ( 2, 0, 0), -1) # NNN hopping
hopping_operator('sf', ( 0, 0, 0), -1, tau=0, sigma=1) # on-site spin flip
hopping_operator('h', ( 0, 0, 0), -1, tau=0, sigma=3) # on-site spin flip
anomalous_operator('D', ( 1, 0, 0), 1) # NN singlet
anomalous_operator('Di', ( 1, 0, 0), 1j) # NN singlet with imaginary amplitude
anomalous_operator('S', ( 0, 0, 0), 1) # on-site singlet 
anomalous_operator('Si', ( 0, 0, 0), 1j) # on-site singlet with imaginary amplitude
anomalous_operator('dz', ( 1, 0, 0), 1, type='dz') # NN triplet
anomalous_operator('dy', ( 1, 0, 0), 1, type='dy') # NN triplet
anomalous_operator('dx', ( 1, 0, 0), 1, type='dx') # NN triplet
density_wave('M', 'Z', ( 1, 0, 0))
density_wave('pT', 'dz', ( 1, 0, 0), amplitude=1, link=( 1, 0, 0))