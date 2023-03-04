import pyqcm
import model_1D_4_C2

sec = 'R0:N4:S0'
pyqcm.set_target_sectors([sec])
pyqcm.set_parameters("""
t=1
U = 4
mu = 0.5*U
""")

print("\ntesting lattice averages (all operators, also with Potthoff functional and potential energy):\n")
pyqcm.new_model_instance()
print('Potthoff functional = ', pyqcm.Potthoff_functional())
print('potential energy = ', pyqcm.potential_energy())
pyqcm.print_averages(pyqcm.averages())

print("\ntesting lattice averages (selected operators):\n")
pyqcm.new_model_instance()
pyqcm.print_averages(pyqcm.averages(['mu']))

print("\ntesting cluster averages from the wavefunction:\n")
pyqcm.print_cluster_averages(pyqcm.cluster_averages())

print("\ntesting cluster Green function averages:\n")
ave = pyqcm.Green_function_average()
print('\naverages of c^\dagger_i c_j :\n\n', ave)

if pyqcm.mixing() == 4:
    ave = pyqcm.Green_function_average(spin_down=True)
    print('\naverages of c^\dagger_i c_j (spin down):\n\n', ave)

print('\naverage of t from GF= ', -(ave[0,1]+ave[1,2]+ave[2,3]))
print('average of mu from GF = ', pyqcm.Green_function_density())



