from pyqcm import *
from pyqcm._spectral import *
import model_2x2x2

set_global_parameter('verb_integrals')
set_target_sectors(['R0:S0'])
set_parameters("""
t = 1
tz = 1
U = 1
mu = 0
D = 0.2
""")
new_model_instance()
print_averages(averages())

