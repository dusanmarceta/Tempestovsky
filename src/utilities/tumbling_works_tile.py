#!/src/utilities/tumbling_matrices.py
"""
Compute free-precession rotation matrices for an arbitrary ASCII STL shape,
ensuring a rational number of spin and precession cycles.

This is a rough work in progress and TEMPEST does not currently support tumbling bodies.

Dependencies:
    pip install numpy-stl numpy
"""


import numpy as np


import matplotlib.pyplot as plt



from scipy.integrate import solve_ivp


from utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix, sun_direction
)  


import yaml

config_file = '../../data/config/yarko_config.yaml'
# Putanja do tvog modela
# shape_file = "../../data/shape_models/Apophis.stl"
shape_file = "../../data/shape_models/Rubber_Duck_1500_facets.stl"
# shape_file = "../../data/shape_models/500m_ico_sphere_80_facets.stl"

# 1. UČITAVANJE CONFIG FAJLA
with open(config_file, "r") as file:
    config = yaml.safe_load(file)


time_step = 20
N_steps = 360
N_rotation = 360

sun_old = np.zeros([N_steps, 3])
sun_tumb = np.zeros([N_steps, 3])
sun_inverse = np.zeros([N_steps, 3])
current_sunlight_directions = np.array([1, 0, 0])

sun_A_B = np.zeros([N_steps, 3])


def compute_tumbling_dynamics(dt, timesteps, y0, I, r_inertial, BODY_AXIS_TO_TRACK):
    """
    Rešava Eulerove jednačine i vraća rezultate uključujući i fiksni vektor ugaonog momenta L.
    """
    
    I1, I2, I3 = I
    
    # --- 1. Definisanje dinamike ---
    def dynamics(t, y):
        w1, w2, w3, phi, theta, psi = y
        # Eulerove jednačine (Dinamika ugaonih brzina)
        dw1 = ((I2 - I3) * w2 * w3) / I1
        dw2 = ((I3 - I1) * w3 * w1) / I2
        dw3 = ((I1 - I2) * w1 * w2) / I3
        
        # Kinematika Eulerovih uglova (3-1-3 konvencija)
        st = np.sin(theta) if abs(np.sin(theta)) > 1e-9 else 1e-9
        dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
        dtheta = w1 * np.cos(psi) - w2 * np.sin(psi)
        dpsi = w3 - dphi * np.cos(theta)
        
        return [dw1, dw2, dw3, dphi, dtheta, dpsi]

    # --- 2. Integracija ---
    t_eval = np.arange(timesteps) * dt
    sol = solve_ivp(dynamics, (0, t_eval[-1]), y0, t_eval=t_eval, 
                    method='DOP853', rtol=1e-11, atol=1e-13)
    
    # --- 3. Proračun ugaonog momenta L (Fiksan u inercijalnom prostoru) ---
    # Uzimamo početne uslove (t=0) da odredimo L_hat
    w0 = np.array([y0[0], y0[1], y0[2]])
    phi0, theta0, psi0 = y0[3], y0[4], y0[5]
    
    c1, s1 = np.cos(phi0), np.sin(phi0)
    c2, s2 = np.cos(theta0), np.sin(theta0)
    c3, s3 = np.cos(psi0), np.sin(psi0)
    
    # Početna matrica R (Inercijalni -> Telo)
    R0 = np.array([
        [ c3*c1 - s3*c2*s1,  c3*s1 + s3*c2*c1, s3*s2],
        [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
        [ s2*s1,            -s2*c1,            c2]
    ])
    
    # L u sistemu tela: [I1*w1, I2*w2, I3*w3]
    L_body = np.array([I1 * w0[0], I2 * w0[1], I3 * w0[2]])
    # L u inercijalnom sistemu: R.T @ L_body
    L_inertial = R0.T @ L_body
    L_hat = L_inertial / np.linalg.norm(L_inertial)

    # --- 4. Transformacije kroz vreme ---
    rotations = []
    r_body_coords = []
    spin_axis_iner = []
    body_axis_iner = []

    for i in range(len(sol.t)):
        phi_i, theta_i, psi_i = sol.y[3, i], sol.y[4, i], sol.y[5, i]
        w_body = sol.y[0:3, i]
        
        ci, si = np.cos(phi_i), np.sin(phi_i)
        cj, sj = np.cos(theta_i), np.sin(theta_i)
        ck, sk = np.cos(psi_i), np.sin(psi_i)
        
        Ri = np.array([
            [ ck*ci - sk*cj*si,  ck*si + sk*cj*ci, sk*sj],
            [-sk*ci - ck*cj*si, -sk*si + ck*cj*ci, ck*sj],
            [ sj*si,            -sj*ci,            cj]
        ])
        
        rotations.append(Ri)
        r_body_coords.append(Ri @ r_inertial[i])
        # PREDLOG POPRAVKE:
        spin_axis_iner.append(Ri.T @ w_body)       # Dodato .T
        body_axis_iner.append(Ri.T @ BODY_AXIS_TO_TRACK) # Dodato .T

    # --- NOVI DEO: Čuvanje poslednjeg stanja ---
    # sol.y[:,-1] uzima sve promenljive (w1, w2, w3, phi, theta, psi) iz poslednjeg vremenskog koraka
    last_state = sol.y[:, -1].tolist() 

    # ... (tvoj postojeći kod za transformacije kroz vreme) ...

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,
        'r_sun': np.stack(r_body_coords),
        'G_axes': np.stack(spin_axis_iner),
        'omega_axes': np.stack(body_axis_iner),
        'time': sol.t,
        'omega_body': sol.y[0:3, :].T,
        'body_axis': np.stack(body_axis_iner),
        'last_state': last_state  # DODATO: spremno za sledeći y0
    }



r_inertial = np.tile(np.array([1, 0, 0]), (N_steps, 1))

if __name__ == "__main__":
    

        
    # 2. IZDVAJANJE PARAMETARA I KONVERZIJA U SI JEDINICE
    I1 = float(config["I1"])
    I2 = float(config["I2"])
    I3 = float(config["I3"])
    I = (I1, I2, I3)

    # Uglovi iz stepeni u radijane
    lambda_L = np.deg2rad(config["lambda_L"])
    beta_L = np.deg2rad(config["beta_L"])
    
    
    
    Rz = np.array([
    [np.cos(lambda_L), -np.sin(lambda_L), 0],
    [np.sin(lambda_L),  np.cos(lambda_L), 0],
    [0,            0,           1]
    ])

    Ry = np.array([
        [np.cos(np.pi/2 - beta_L), 0, np.sin(np.pi/2 - beta_L)],
        [0,                       1, 0],
        [-np.sin(np.pi/2 - beta_L), 0, np.cos(np.pi/2 - beta_L)]
    ])
    
    R = Rz @ Ry
    
    '''
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    '''
    r_new = (R.T @ r_inertial.T).T
    

    phi_0 = np.deg2rad(config["phi_0"])
    psi_0 = np.deg2rad(config["psi_0"])

    # Periodi, energija i epoha
    P_phi_h = float(config["P_phi_h"])
    P_psi_h = float(config["P_psi_h"])
    E_ratio = float(config["E_ratio"])

    u_y = np.array([-np.sin(lambda_L), np.cos(lambda_L), 0.0])
    u_x = -np.array([-np.sin(beta_L) * np.cos(lambda_L), -np.sin(beta_L) * np.sin(lambda_L), np.cos(beta_L)])
    u_z = np.array([np.cos(beta_L) * np.cos(lambda_L), np.cos(beta_L) * np.sin(lambda_L), np.sin(beta_L)])
    

    # Matrica rotacije: R_L2Ecl * v_local = v_ecliptic
    R_L2Ecl = np.column_stack((u_x, u_y, u_z))
    
    
    A = (np.sin(psi_0) ** 2 / I1) + (np.cos(psi_0) ** 2 / I2)
    sin2_theta0 = (E_ratio - 1.0) / (I3 * A - 1.0)
    theta_0 = np.arcsin(np.sqrt(sin2_theta0))

    # 4. PRORAČUN UGAONIH BRZINA I VEKTORA POČETNOG STANJA y0
    w_phi = (2 * np.pi) / (P_phi_h * 3600.0)

    # UNIVERZALNO: Deljenje sa A radi i za I1 != I2 i za I1 == I2
    L_mag = w_phi / A

    w1_0 = (L_mag * np.sin(theta_0) * np.sin(psi_0)) / I1
    w2_0 = (L_mag * np.sin(theta_0) * np.cos(psi_0)) / I2
    w3_0 = (L_mag * np.cos(theta_0)) / I3

    y0 = [w1_0, w2_0, w3_0, phi_0, theta_0, psi_0]
    
    sim = compute_tumbling_dynamics(dt = time_step, timesteps = N_steps, y0 = y0,
                                        I = np.array([I1, I2, I3]), 
                                        r_inertial = r_new, 
                                        BODY_AXIS_TO_TRACK = np.array([1.0, 0.0, 0.0]))
    
    
    print("Computing kinematics (rotations, axes, and hodograph)...")
    output_file = "tumbling_animation_2.gif"
    fps = 30

    # Izdvajamo ono što nam treba za animaciju
    rotations = sim['rotations']
    L_hat = sim['L_axis']
    omega_vecs = sim['G_axes']
    r_sun = sim['r_sun']
    
    time = sim['time']
    body_axis = sim['body_axis']  # Putanja z-ose tela u inercijalnom prostoru
    
    x_body = body_axis[:, 0]
    y_body = body_axis[:, 1]
    z_body = body_axis[:, 2]
    
    axis_inertial = -sim['r_sun']  # Putanja z-ose tela u inercijalnom prostoru
    
    x_inertial = axis_inertial[:, 0]
    y_inertial = axis_inertial[:, 1]
    z_inertial = axis_inertial[:, 2]

    
    # Longituda i latituda u inercijalnom prostoru
    long = np.rad2deg(np.arctan2(y_body, x_body))
    lat = np.rad2deg(np.arcsin(np.clip(z_body, -1.0, 1.0)))  # clip radi numeričke stabilnosti
    
    # Longituda i latituda u inercijalnom prostoru
    long_inertial = np.rad2deg(np.arctan2(y_inertial, x_inertial))
    lat_inertial = np.rad2deg(np.arcsin(np.clip(z_inertial, -1.0, 1.0)))  # clip radi numeričke stabilnosti
   


t = 0
rotation_old = np.zeros((N_steps, 3, 3), dtype=np.float64)
rotation_axis = np.array([np.cos(lambda_L) * np.cos(beta_L), np.sin(lambda_L) * np.cos(beta_L), np.sin(beta_L)])
for i in range(N_steps):
    
    rotation_matrix = calculate_rotation_matrix(rotation_axis, 
                                             (2 * np.pi / N_rotation) * i)
    

    rotation_old[i] = rotation_matrix
    
    sun_old[i] = np.dot(rotation_matrix.T, current_sunlight_directions) # THIS
    

    


'''
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
'''
sun_A_B = axis_inertial @ R.T
    
sun_long_old = np.rad2deg(np.arctan2(sun_old.T[1], sun_old.T[0]))
sun_lat_old = np.rad2deg(np.arcsin(sun_old.T[2]))


sun_long_tumb = np.rad2deg(np.arctan2(sun_tumb.T[1], sun_tumb.T[0]))


plt.figure()
plt.title('x', fontsize = 24)
plt.plot(time, sun_old.T[0])
plt.plot(time, sun_A_B.T[0])

plt.figure()
plt.title('y', fontsize = 24)
plt.plot(time, sun_old.T[1])
plt.plot(time, sun_A_B.T[1])

plt.figure()
plt.title('z', fontsize = 24)
plt.plot(time, sun_old.T[2])
plt.plot(time, sun_A_B.T[2])


plt.figure()
plt.title('long', fontsize = 24)
plt.plot(time, sun_long_old)

plt.figure()
plt.title('lat', fontsize = 24)
plt.plot(time, sun_lat_old)









