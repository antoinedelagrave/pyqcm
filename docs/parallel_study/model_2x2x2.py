from pyqcm import *

new_cluster_model('2x2x2', 8, 0)
add_cluster('2x2x2', [0, 0, 0], [[0, 0, 0],( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0),[0, 0, 1],[1, 0, 1], [0, 1, 1], [1, 1, 1]])
lattice_model('2x2x2', [( 2, 0, 0), ( 0, 2, 0), [0, 0, 2]])
interaction_operator('U')
interaction_operator('J', link=( 1, 0, 0), type='Hund')

hopping_operator('t', ( 0, 1, 0), -1)
hopping_operator('t', ( 1, 0, 0), -1)
hopping_operator('tz', [0, 0, 1], -1)
hopping_operator('t2', ( 1, 1, 0), -1)
hopping_operator('t2', (-1, 1, 0), -1)
density_wave('M', 'Z', ( 1, 1, 0))
density_wave('H', 'Z', [0, 0, 0])
hopping_operator('Hx', [0, 0, 0], -1, tau=0, sigma=1)
hopping_operator('t3', ( 2, 0, 0), -1)
hopping_operator('t3', ( 0, 2, 0), -1)
anomalous_operator('D', ( 1, 0, 0), 1)
anomalous_operator('D', ( 0, 1, 0), -1)
anomalous_operator('p', ( 1, 0, 0), 1, type='dz')

