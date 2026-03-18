import numpy as np
from tumbling_matrices import compute_tumbling_kinematics

timesteps = 1000
long_deg = 1.036
ratio_is_il = 1.639
spinrate = 1.0145280415550686e-05




timesteps = 1
long_deg = 0
lat_deg = 90
theta_deg = 30
P_spin = 1
P_prec = 100
n_cycles=1
    

aaa = compute_tumbling_kinematics(timesteps, long_deg, lat_deg, theta_deg, P_spin, P_prec, n_cycles)

