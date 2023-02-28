from pyqcm import *

new_cluster_model('L4', 4, 0, [[4,3,2,1]])
add_cluster('L4', [0, 0, 0], [[0, 0, 0], ( 1, 0, 0), ( 2, 0, 0), [3, 0, 0]])
lattice_model('1D_L4', [[4, 0, 0]])

interaction_operator('U')
interaction_operator('J', link=( 1, 0, 0), type='Hund')
hopping_operator('t', ( 1, 0, 0), -1)  # NN hopping
hopping_operator('tp', ( 2, 0, 0), -1)  # NNN hopping
hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)  # field in the x direction
hopping_operator('h', [0, 0, 0], 1, tau=0, sigma=3)  # field in the x direction
anomalous_operator('D', ( 1, 0, 0), 1)  # NN singlet
anomalous_operator('Di', ( 1, 0, 0), 1j)  # NN singlet with imaginary amplitude
anomalous_operator('S', [0, 0, 0], 1)  # on-site singlet
anomalous_operator('Si', [0, 0, 0], 1j)  # on-site singlet with imaginary amplitude
density_wave('H', 'Z', [0, 0, 0])
density_wave('Hx', 'X', [0, 0, 0])
