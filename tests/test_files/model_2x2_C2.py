from pyqcm import *

new_cluster_model('2x2_C2', 4, 0, [[4, 3, 2, 1]])
add_cluster('2x2_C2', [0, 0, 0], [[0, 0, 0],
                                      ( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0)])
lattice_model('2x2_C2', [( 2, 0, 0), ( 0, 2, 0)])
interaction_operator('U')
interaction_operator('V', link=( 1, 0, 0), amplitude=1)
interaction_operator('V', link=( 0, 1, 0), amplitude=1)
hopping_operator('Vm', [0, 0, 0], 1, tau=0)  # NN hopping
V_eig = 0.5
interaction_operator('J', link=( 1, 0, 0), type='Hund')

hopping_operator('t', ( 0, 1, 0), -1)
hopping_operator('t', ( 1, 0, 0), -1)
hopping_operator('t2', ( 1, 1, 0), -1)
hopping_operator('t2', (-1, 1, 0), -1)
hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
density_wave('M', 'Z', ( 1, 1, 0))
hopping_operator('t3', ( 2, 0, 0), -1)
hopping_operator('t3', ( 0, 2, 0), -1)
anomalous_operator('D', ( 1, 0, 0), 1)
anomalous_operator('D', ( 0, 1, 0), -1)


################################################
set_target_sectors(['R0:N4:S0'])
set_parameters("""
t = 1
t2 = 1e-9
U = 8
mu = 4
M = 0
M_1 = 0.15
t_1 = 1
""")
