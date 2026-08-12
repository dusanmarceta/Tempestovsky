#!/src/utilities/tumbling_matrices.py
"""
Compute free-precession rotation matrices for an arbitrary ASCII STL shape,
ensuring a rational number of spin and precession cycles.

This is a rough work in progress and TEMPEST does not currently support tumbling bodies.

Dependencies:
    pip install numpy-stl numpy
"""

import argparse
import numpy as np
from stl import mesh
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp


from utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix, sun_direction
)  


import yaml

config_file = '../../data/config/yarko_config_no_precession.yaml'
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

# lambda_L_deg = 0
# beta_L_deg = 90




# lam = np.deg2rad(lambda_L_deg)
# bet = np.deg2rad(beta_L_deg)



sun_old = np.zeros([N_steps, 3])
sun_tumb = np.zeros([N_steps, 3])
sun_inverse = np.zeros([N_steps, 3])
current_sunlight_directions = np.array([1, 0, 0])

sun_A_B = np.zeros([N_steps, 3])



    
def animate_rotation_inertial(shape_file, rotation_matrices, omega_vectors, 
                     lambda_L_deg, beta_L_deg,
                     output_file=None, fps=20, skip_frames=1, hodograf_axis='z'):
    """
    Animacija rotacije tela u inercijalnom ekliptičkom koordinatnom sistemu.
    
    Parametri:
    - lambda_L_deg: Ekliptička dužina vektora L u stepenima
    - beta_L_deg: Ekliptička širina vektora L u stepenima
    - hodograf_axis: 'x', 'y', 'z', 'omega' ili 'all' (None za onemogućavanje)
    """
    # --- 1. MATRICA TRANSFORMACIJE: LOKALNI SISTEM L -> EKLIPTIČKI SISTEM ---
    lam = np.deg2rad(lambda_L_deg)
    bet = np.deg2rad(beta_L_deg)

    # Jedinični vektori baza L-sistema izraženi u ekliptičkim koordinatama
    u_x = np.array([-np.sin(lam), np.cos(lam), 0.0])
    u_y = np.array([-np.sin(bet) * np.cos(lam), -np.sin(bet) * np.sin(lam), np.cos(bet)])
    u_z = np.array([np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)])

    # Matrica rotacije: R_L2Ecl * v_local = v_ecliptic
    R_L2Ecl = np.column_stack((u_x, u_y, u_z))
    
    # Vektor ugaonog momenta L u ekliptičkom sistemu
    L_hat_ecl = u_z

    # --- 2. UČITAVANJE STL MREŽE I PRIPREMA SCENE ---
    shape_mesh = mesh.Mesh.from_file(shape_file)
    vertices = shape_mesh.vectors
    
    center = np.mean(vertices.reshape(-1, 3), axis=0)
    max_range = np.max(np.ptp(vertices.reshape(-1, 3), axis=0))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    collection = Poly3DCollection(vertices, alpha=0.7, edgecolor='k', linewidth=0.3)
    collection.set_facecolor('lightgray')
    ax.add_collection3d(collection)
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    ax.view_init(elev=30, azim=260)
    axis_length = max_range * 1.2
    
    # Fiksni vektor L u ekliptičkom sistemu (Crni)
    ax.quiver(0, 0, 0, L_hat_ecl[0]*axis_length, L_hat_ecl[1]*axis_length, L_hat_ecl[2]*axis_length, 
              color='k', label=f'L (λ={lambda_L_deg}°, β={beta_L_deg}°)', linewidth=4)
    
    # Trenutna osa rotacije (Cijan)
    omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Omega (Trenutna osa)', linewidth=2)
    
    # Inicijalizacija glavnih osa
    osa_1 = ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', label='Glavna osa 1 (X)', linewidth=2)
    osa_2 = ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', label='Glavna osa 2 (Y)', linewidth=2)
    osa_3 = ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', label='Glavna osa 3 (Z)', linewidth=2)
    
    # --- 3. PRIPREMA HODOGRAFA ---
    hodograf_lines = {}
    hodograf_pts = {'x': [], 'y': [], 'z': [], 'omega': []}
    
    target_axes = []
    if hodograf_axis in ['x', 1, 'all']: target_axes.append('x')
    if hodograf_axis in ['y', 2, 'all']: target_axes.append('y')
    if hodograf_axis in ['z', 3, 'all']: target_axes.append('z')
    if hodograf_axis in ['omega', 'w', 'all']: target_axes.append('omega')
    
    colors = {'x': 'red', 'y': 'green', 'z': 'blue', 'omega': 'cyan'}
    labels = {'x': 'Hodograf X-ose', 'y': 'Hodograf Y-ose', 'z': 'Hodograf Z-ose', 'omega': 'Herpolhodogram (Omega)'}
    
    for ax_name in target_axes:
        line, = ax.plot([], [], [], color=colors[ax_name], linestyle='--', label=labels[ax_name], lw=1.5)
        hodograf_lines[ax_name] = line

    vertices_original = vertices.copy()
    
    


    # --- 4. ANIMATION UPDATE FUNKCIJA ---
    def update(frame):
        nonlocal omega_quiver, osa_1, osa_2, osa_3
        
        i = frame * skip_frames
        if i >= len(rotation_matrices):
            i = len(rotation_matrices) - 1
            
        R_local = rotation_matrices[i]
        
        print(i)
        
        # Rotacija temena STL tela u ekliptički sistem
        # V_ecl = (V_body @ R_local) @ R_L2Ecl.T
        rotated_vertices = (vertices_original - center) @ R_local @ R_L2Ecl.T + center
        collection.set_verts(rotated_vertices)
        
        # Transformacija Omega vektora u ekliptički sistem
        w_vec_local = omega_vectors[i]
        w_vec_ecl = R_L2Ecl @ w_vec_local
        w_dir = (w_vec_ecl / np.linalg.norm(w_vec_ecl)) * axis_length
        
        omega_quiver.remove()
        omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
        
        # Transformacija osa tela u ekliptički sistem
        # Za kolona-vektore osa: v_ecl = R_L2Ecl @ (R_local.T @ e_i)
        v1_rotated = R_L2Ecl @ (R_local.T @ np.array([1, 0, 0])) * axis_length
        v2_rotated = R_L2Ecl @ (R_local.T @ np.array([0, 1, 0])) * axis_length
        v3_rotated = R_L2Ecl @ (R_local.T @ np.array([0, 0, 1])) * axis_length
        
        osa_1.remove()
        osa_2.remove()
        osa_3.remove()
        osa_1 = ax.quiver(0, 0, 0, v1_rotated[0], v1_rotated[1], v1_rotated[2], color='red', linewidth=2)
        osa_2 = ax.quiver(0, 0, 0, v2_rotated[0], v2_rotated[1], v2_rotated[2], color='green', linewidth=2)
        osa_3 = ax.quiver(0, 0, 0, v3_rotated[0], v3_rotated[1], v3_rotated[2], color='blue', linewidth=2)

        # Update hodografa u ekliptičkom prostoru
        current_vectors = {
            'x': v1_rotated, 
            'y': v2_rotated, 
            'z': v3_rotated, 
            'omega': w_dir
        }
        
        for ax_name in target_axes:
            hodograf_pts[ax_name].append(current_vectors[ax_name])
            pts = np.array(hodograf_pts[ax_name])
            hodograf_lines[ax_name].set_data(pts[:, 0], pts[:, 1])
            hodograf_lines[ax_name].set_3d_properties(pts[:, 2])
        
        ax.set_title(f'Inertial Ecliptic Frame | Frame {i+1}/{len(rotation_matrices)}')
        return collection, omega_quiver, osa_1, osa_2, osa_3, *hodograf_lines.values()

    ax.legend(loc='upper right')
    n_frames = len(rotation_matrices) // skip_frames
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    if output_file:
        anim.save(output_file, writer='pillow', fps=fps)
    else:
        plt.show()
        
    # --- 5. CHUAVANJE POSLEDNJEG FREJMA U PDF ---
    last_i = len(rotation_matrices) - 1
    R_local = rotation_matrices[last_i]

    rotated_vertices = (vertices_original - center) @ R_local @ R_L2Ecl.T + center
    collection.set_verts(rotated_vertices)
    
    

    osa_1.remove()
    osa_2.remove()
    osa_3.remove()
    v1_final = R_L2Ecl @ (R_local.T @ np.array([1, 0, 0])) * axis_length
    v2_final = R_L2Ecl @ (R_local.T @ np.array([0, 1, 0])) * axis_length
    v3_final = R_L2Ecl @ (R_local.T @ np.array([0, 0, 1])) * axis_length
    
    osa_1 = ax.quiver(0, 0, 0, v1_final[0], v1_final[1], v1_final[2], color='red', linewidth=2)
    osa_2 = ax.quiver(0, 0, 0, v2_final[0], v2_final[1], v2_final[2], color='green', linewidth=2)
    osa_3 = ax.quiver(0, 0, 0, v3_final[0], v3_final[1], v3_final[2], color='blue', linewidth=2)

    w_vec_local = omega_vectors[last_i]
    w_vec_ecl = R_L2Ecl @ w_vec_local
    w_dir = (w_vec_ecl / np.linalg.norm(w_vec_ecl)) * axis_length
    omega_quiver.remove()
    omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)

    for ax_name in target_axes:
        pts = np.array(hodograf_pts[ax_name])
        if len(pts) > 0:
            hodograf_lines[ax_name].set_data(pts[:, 0], pts[:, 1])
            hodograf_lines[ax_name].set_3d_properties(pts[:, 2])

    ax.set_title(f"Final Frame (Ecliptic: λ={lambda_L_deg}°, β={beta_L_deg}°)")
    fig.savefig("last_frame.pdf", bbox_inches='tight')
        
    return anim



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
        r_body_coords.append(Ri @ r_inertial)
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
    






































r_inertial = np.array([1, 0, 0])


if __name__ == "__main__":
    
    
    
    
    
    rotation_tumbling = np.zeros((N_steps, 3, 3), dtype=np.float64)
    rotation_inverse = np.zeros((N_steps, 3, 3), dtype=np.float64)
        
    
    
    
    
    
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
    

    
    r_new = R.T @ r_inertial
    
    

    
    # lambda_L = lam
    # beta_L = bet
    
    
    phi_0 = np.deg2rad(config["phi_0"])
    psi_0 = np.deg2rad(config["psi_0"])

    # Periodi, energija i epoha
    P_phi_h = float(config["P_phi_h"])
    P_psi_h = float(config["P_psi_h"])
    E_ratio = float(config["E_ratio"])
    epoch_jd = float(config["epoch_jd"])
    
    
    


    # # # Jedinični vektori baza L-sistema izraženi u ekliptičkim koordinatama
    # u_x = np.array([-np.sin(lambda_L), np.cos(lambda_L), 0.0])
    # u_y = np.array([-np.sin(beta_L) * np.cos(lambda_L), -np.sin(beta_L) * np.sin(lambda_L), np.cos(beta_L)])
    # u_z = np.array([np.cos(beta_L) * np.cos(lambda_L), np.cos(beta_L) * np.sin(lambda_L), np.sin(beta_L)])
    
    
    # # Jedinični vektori baza L-sistema izraženi u ekliptičkim koordinatama
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
    
    
    

    # Primer pozivanja za prikaz hodografa Z-ose i Omega ose istovremeno:
    animate_rotation_inertial(shape_file = shape_file, 
                              rotation_matrices = rotations, 
                              omega_vectors = omega_vecs, 
                              lambda_L_deg = np.rad2deg(lambda_L), beta_L_deg = np.rad2deg(beta_L),
                              output_file = output_file, 
                              fps = fps, 
                              skip_frames = 1, 
                              hodograf_axis = 'z'
                              )
    
    # # Vreme u sekundama ili satima radi grafika
    time = sim['time']
    body_axis = sim['body_axis']  # Putanja z-ose tela u inercijalnom prostoru
    
    x_body = body_axis[:, 0]
    y_body = body_axis[:, 1]
    z_body = body_axis[:, 2]
    
    axis_inertial = sim['r_sun']  # Putanja z-ose tela u inercijalnom prostoru
    
    x_inertial = axis_inertial[:, 0]
    y_inertial = axis_inertial[:, 1]
    z_inertial = axis_inertial[:, 2]
    
    
    
    
    
    
    
    
    
    # Longituda i latituda u inercijalnom prostoru
    long = np.rad2deg(np.arctan2(y_body, x_body))
    lat = np.rad2deg(np.arcsin(np.clip(z_body, -1.0, 1.0)))  # clip radi numeričke stabilnosti
    
    # Longituda i latituda u inercijalnom prostoru
    long_inertial = np.rad2deg(np.arctan2(y_inertial, x_inertial))
    lat_inertial = np.rad2deg(np.arcsin(np.clip(z_inertial, -1.0, 1.0)))  # clip radi numeričke stabilnosti
    
    for i in range(len(rotations)):
        rotation_tumbling[i] = rotations[i] @ R_L2Ecl.T
        rotation_inverse[i] = rotation_tumbling[i].T
        
        sun_tumb[i] = np.dot((rotation_tumbling[i]).T, current_sunlight_directions) # THIS
        sun_inverse[i] = np.dot(rotation_inverse[i], current_sunlight_directions) # THIS
        


t = 0
rotation_old = np.zeros((N_steps, 3, 3), dtype=np.float64)
rotation_axis = np.array([np.cos(lambda_L) * np.cos(beta_L), np.sin(lambda_L) * np.cos(beta_L), np.sin(beta_L)])
for i in range(N_steps):
    
    rotation_matrix = calculate_rotation_matrix(rotation_axis, 
                                             (2 * np.pi / N_rotation) * i)
    

    rotation_old[i] = rotation_matrix
    
    sun_old[i] = np.dot(rotation_matrix.T, current_sunlight_directions) # THIS
    
    
    # sun_A_B[i] = rotation_matrix.T @ R @ np.array([x_inertial[i], y_inertial[i], z_inertial[i]])
    sun_A_B[i] = R @ np.array([x_inertial[i], y_inertial[i], z_inertial[i]])
    
    
sun_long_old = np.rad2deg(np.arctan2(sun_old.T[1], sun_old.T[2]))
sun_lat_old = np.rad2deg(np.arcsin(sun_old.T[0]))


sun_long_tumb = np.rad2deg(np.arctan2(sun_tumb.T[1], sun_tumb.T[0]))
sun_lat_tumb = np.rad2deg(np.arcsin(sun_tumb.T[2]))

sun_long_inverse = np.rad2deg(np.arctan2(sun_inverse.T[1], sun_inverse.T[0]))
sun_lat_inverse = np.rad2deg(np.arcsin(sun_inverse.T[2]))

plt.figure()
plt.plot(time, sun_old.T[0])
plt.plot(time, sun_A_B.T[0])

plt.figure()
plt.plot(time, sun_old.T[1])
plt.plot(time, sun_A_B.T[1])

plt.figure()
plt.plot(time, sun_old.T[2])
plt.plot(time, sun_A_B.T[2])




