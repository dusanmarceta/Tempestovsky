import numpy as np
from astropy import constants as const
from utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix, sun_direction, compute_tumbling_dynamics
) 
import matplotlib.pyplot as plt


from types import SimpleNamespace

simulation = SimpleNamespace()

# =============================================================================
# # OLD
# config_data = {
#     "I1": 10.0,
#     "I2": 10.0,
#     "phi_0": 0.0,
#     "psi_0": 0.0,
#     "E_ratio": 1.0,         # theta_0 = 0 deg (fiksirana osa rotacije)
#     "P_phi_h": 240.0,       # P_psi = P_phi * (1/I2) = 240 * 0.1 = 24.0 h
#     "lambda_L": 0.0,        # Ekliptička dužina ose rotacije
#     "beta_L": 60.0,         # Nagib 30 deg od vertikale (90 - 30 = 60)
#     "a_au": 1.0,
#     "ecc": 0,
#     "version": 'old'
# }
# =============================================================================

# NEW
config_data = {
    "I1": 316319.46,        # Prilagođeno za P_phi = 1000 god i P_psi = 24h pri theta_0 = 30 deg
    "I2": 316319.46,        # Aksijalno simetrično telo
    "phi_0": 0.0,
    "psi_0": 90.0,
    "E_ratio": 0.7508,      # Generiše nagib theta_0 = 30 deg u odnosu na L
    "P_phi_h": 8766000.0,   # Precesija od 1000 godina (1000 x 365.25d x 24h)
    "lambda_L": 0.0,        # Centar precesije je vertikala
    "beta_L": 90.0,         # Vertikala (Ekliptički pol)
    "a_au": 1.0,
    "ecc": 0.0,
    'version': 'new'
}

timesteps_per_day = 240
delta_t = 360




# ============================================================
# UČITAVANJE PARAMETARA IZ CONFIG-A
# ============================================================

for key, value in config_data.items():
    if isinstance(value, list):
        value = np.array(value)
    globals()[key] = value




# Compute unit vector from RA and Dec
ra_radians = np.radians(lambda_L)
dec_radians = np.radians(beta_L)

rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                               np.sin(ra_radians) * np.cos(dec_radians), 
                               np.sin(dec_radians)])

# ============================================================
# MOMENTI INERCIJE
# ============================================================

I3 = 1
I = (I1, I2, I3)
simulation.I = I


# ============================================================
# UGLovi IZ STEPENI U RADIJANE
# ============================================================

phi_0 = np.deg2rad(phi_0)
psi_0 = np.deg2rad(psi_0)


# ============================================================
# PERIODI, ENERGIJA I EPOHA
# ============================================================

A = (
    np.sin(psi_0)**2 / I1
    + np.cos(psi_0)**2 / I2
)

sin2_theta0 = (E_ratio - 1.0) / (I3 * A - 1.0)

theta_0 = np.arcsin(np.sqrt(sin2_theta0))


# ============================================================
# POČETNE UGAONE BRZINE
# ============================================================

w_phi = (2 * np.pi) / (P_phi_h * 3600.0)

# Univerzalno: radi i za I1 != I2 i za I1 == I2
L_mag = w_phi / A

w1_0 = (L_mag * np.sin(theta_0) * np.sin(psi_0)) / I1
w2_0 = (L_mag * np.sin(theta_0) * np.cos(psi_0)) / I2
w3_0 = (L_mag * np.cos(theta_0)) / I3


# Početno stanje
initial_rotation_state = [
    w1_0,
    w2_0,
    w3_0,
    phi_0,
    theta_0,
    psi_0
]


# ============================================================
# REŽIM ROTACIJE I P_psi
# ============================================================

if np.isclose(E_ratio, 1.0) or theta_0 < 1e-7:

    # Egzaktno za fiksiranu rotaciju theta_0 = 0
    P_psi_h_calculated = P_phi_h * A * I3
    rotation_period_s = P_psi_h_calculated * 3600

    angular_velocity = (2 * np.pi) / rotation_period_s

    mode = "Fixed-Axis Rotation"

    print(
        f'fiksna rotacija sa periodom od '
        f'{P_psi_h_calculated} h'
    )

# else:

#     # U tumblingu se dobija integracijom
#     P_psi_h_calculated = None

#     mode = "General Tumbling"


# ============================================================
# ORBITALNI PARAMETRI
# ============================================================

mean_motion = np.sqrt(
    const.GM_sun.value /
    (a_au * const.au.value)**3
)

orbital_period = 2 * np.pi / mean_motion


simulation.ecc = ecc
simulation.a_au = a_au
simulation.mean_motion = mean_motion

# ============================================================
# TERMIČKI PARAMETRI
# ============================================================






# ============================================================
# VREMENSKI KORAK
# ============================================================





timesteps_per_orbit = int(
    np.ceil(orbital_period / delta_t)
)

simulation.delta_t = delta_t


# ============================================================
# ROTACIONA OSA IZ RA I DEC
# ============================================================

ra_radians = np.radians(lambda_L)
dec_radians = np.radians(beta_L)

rotation_axis = np.array([
    np.cos(ra_radians) * np.cos(dec_radians),
    np.sin(ra_radians) * np.cos(dec_radians),
    np.sin(dec_radians)
])


# ============================================================
# ORBITALNI PERIOD I BROJ KORAKA GODIŠNJE
# ============================================================

orbital_period = (
    2 * np.pi /
    np.sqrt(
        const.GM_sun.value /
        (a_au * const.au.value)**3
    )
)

timesteps_per_year = int(
    np.ceil(orbital_period / delta_t)
)


number_of_orbit_sections = 100

timesteps_per_orbit = int(np.ceil(orbital_period / delta_t))

timesteps_per_orbit_section = (np.ones(number_of_orbit_sections) * timesteps_per_orbit/number_of_orbit_sections).astype(int)
timesteps_per_orbit_section[-1] = timesteps_per_orbit - sum(timesteps_per_orbit_section[:-1])

r_sun = []
x_trans_old = []
y_trans_old = []
z_trans_old = []

x_trans_new = []
y_trans_new = []
z_trans_new = []


time_array = []
total_tumbling_time = 0
for orbit_section in range(number_of_orbit_sections):
    
    
    print(f'orbit section: {orbit_section}')
    total_time = np.sum(timesteps_per_orbit_section[:orbit_section]) * delta_t
    
    rotation_matrices = np.zeros((timesteps_per_orbit_section[orbit_section], 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)
    current_sunlight_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)
    rotated_transfersal_directions = np.zeros((timesteps_per_orbit_section[orbit_section], 3), dtype=np.float64)

    current_sun_distance = np.zeros(timesteps_per_orbit_section[orbit_section])
    true_anomaly = np.zeros(timesteps_per_orbit_section[orbit_section])
    
    for t in range(timesteps_per_orbit_section[orbit_section]):
        total_time += delta_t
           
        current_sunlight_directions[t], current_sun_distance[t], true_anomaly[t] = sun_direction(total_time, simulation) # this is OK
        
        
        r_sun.append(current_sun_distance[t])
        
        current_transfersal_direction = np.cross(current_sunlight_directions[t], np.array([0, 0, 1])) # this is OK
        
        rotation_matrix = calculate_rotation_matrix(rotation_axis, 
                                                 (2 * np.pi / timesteps_per_day) * (np.sum(timesteps_per_orbit_section[:orbit_section]) + t)) # THIS
 
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, current_sunlight_directions[t]) # THIS
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])
        
        rotated_transfersal_directions[t] = np.dot(rotation_matrix.T, current_transfersal_direction) # THIS


        time_array.append(total_time)
        x_trans_old.append(rotated_transfersal_directions[t][0])
        y_trans_old.append(rotated_transfersal_directions[t][1])
        z_trans_old.append(rotated_transfersal_directions[t][2])
        
    rotation = compute_tumbling_dynamics(dt = simulation.delta_t, timesteps = timesteps_per_orbit_section[orbit_section], 
                                         y0 = initial_rotation_state,
                                         I = simulation.I, 
                                         r_inertial = current_sunlight_directions,
                                         lambda_L_deg = lambda_L,
                                         beta_L_deg = beta_L, 
                                         initialisation = 0
                                         )
    
    
    
    
    
    
    total_tumbling_time -= timesteps_per_orbit_section[orbit_section] * simulation.delta_t
    
    # print('---------------------------- TUMBLING TIME ---------------------------')
    # print(total_tumbling_time)
    
    initial_rotation_state = rotation['last_state']
    rotated_sunlight_directions = rotation['r_body']
    rotated_transfersal_directions = rotation['t_body']
    rotation_matrices = rotation['rotations']
    
    x_trans_new.extend(np.transpose(rotated_transfersal_directions)[0])
    y_trans_new.extend(np.transpose(rotated_transfersal_directions)[1])
    z_trans_new.extend(np.transpose(rotated_transfersal_directions)[2])
    # 
    
if version == 'old':
    filename = 'old.txt'
    x_write = x_trans_old
    y_write = y_trans_old
    z_write = z_trans_old
elif version == 'new':
    filename = 'new.txt'
    x_write = x_trans_new
    y_write = y_trans_new
    z_write = z_trans_new
    
# np.savetxt('old.txt', np.column_stack((x_trans_old, y_trans_old, y_trans_old)))
np.savetxt(filename, np.column_stack((x_write, y_write, z_write)))