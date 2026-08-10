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

# =============================================================================
# def animate_rotation(shape_file, rotation_matrices, L_hat, omega_vectors, 
#                      output_file=None, fps=20, skip_frames=1, hodograf_axis='z'):
#     """
#     Parametar hodograf_axis određuje za koju osu inercije se crta hodograf:
#     - 'x' ili 1: Crvena osa
#     - 'y' ili 2: Zelena osa
#     - 'z' ili 3: Plava osa
#     - 'all': Sve tri ose
#     - None: Bez hodografa
#     """
#     shape_mesh = mesh.Mesh.from_file(shape_file)
#     vertices = shape_mesh.vectors
#     
#     center = np.mean(vertices.reshape(-1, 3), axis=0)
#     max_range = np.max(np.ptp(vertices.reshape(-1, 3), axis=0))
#     
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     
#     collection = Poly3DCollection(vertices, alpha=0.7, edgecolor='k', linewidth=0.3)
#     collection.set_facecolor('lightgray')
#     ax.add_collection3d(collection)
#     
#     ax.set_xlim(center[0] - max_range, center[0] + max_range)
#     ax.set_ylim(center[1] - max_range, center[1] + max_range)
#     ax.set_zlim(center[2] - max_range, center[2] + max_range)
#     
#     ax.view_init(elev=30, azim=260)
#     
#     axis_length = max_range * 1.2
#     
#     # Fiksni vektor L (Crni)
#     ax.quiver(0, 0, 0, L_hat[0]*axis_length, L_hat[1]*axis_length, L_hat[2]*axis_length, 
#               color='k', label='L (Ugaoni moment)', linewidth=5)
#     
#     # Trenutna osa rotacije (Cijan)
#     omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Omega (Trenutna osa)', linewidth=2)
#     
#     # Inicijalizacija glavnih osa
#     osa_1 = ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', label='Glavna osa 1 (X)', linewidth=2)
#     osa_2 = ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', label='Glavna osa 2 (Y)', linewidth=2)
#     osa_3 = ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', label='Glavna osa 3 (Z)', linewidth=2)
#     
#     # Inicijalizacija linija i nizova za hodograme na osnovu izabranog parametra
#     hodograf_lines = {}
#     hodograf_pts = {'x': [], 'y': [], 'z': []}
#     
#     target_axes = []
#     if hodograf_axis in ['x', 1, 'all']: target_axes.append('x')
#     if hodograf_axis in ['y', 2, 'all']: target_axes.append('y')
#     if hodograf_axis in ['z', 3, 'all']: target_axes.append('z')
#     
#     colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
#     labels = {'x': 'Hodograf X-ose', 'y': 'Hodograf Y-ose', 'z': 'Hodograf Z-ose'}
#     
#     for ax_name in target_axes:
#         line, = ax.plot([], [], [], color=colors[ax_name], linestyle='--', label=labels[ax_name], lw=1.5)
#         hodograf_lines[ax_name] = line
# 
#     vertices_original = vertices.copy()
# 
#     def update(frame):
#         nonlocal omega_quiver, osa_1, osa_2, osa_3
#         
#         i = frame * skip_frames
#         if i >= len(rotation_matrices):
#             i = len(rotation_matrices) - 1
#             
#         print(i)
#             
#         R = rotation_matrices[i]
#         
#         # Rotacija tela
#         rotated_vertices = (vertices_original - center) @ R + center
#         collection.set_verts(rotated_vertices)
#         
#         # Update Omega vektora
#         omega_quiver.remove()
#         w_vec = omega_vectors[i]
#         w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
#         omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
#         
#         # Update glavnih osa inercije
#         osa_1.remove()
#         osa_2.remove()
#         osa_3.remove()
#         
#         v1_rotated = R.T @ np.array([1, 0, 0]) * axis_length
#         v2_rotated = R.T @ np.array([0, 1, 0]) * axis_length
#         v3_rotated = R.T @ np.array([0, 0, 1]) * axis_length
#         
#         osa_1 = ax.quiver(0, 0, 0, v1_rotated[0], v1_rotated[1], v1_rotated[2], color='red', linewidth=2)
#         osa_2 = ax.quiver(0, 0, 0, v2_rotated[0], v2_rotated[1], v2_rotated[2], color='green', linewidth=2)
#         osa_3 = ax.quiver(0, 0, 0, v3_rotated[0], v3_rotated[1], v3_rotated[2], color='blue', linewidth=2)
# 
#         # Update hodografa za izabrane ose
#         current_vectors = {'x': v1_rotated, 'y': v2_rotated, 'z': v3_rotated}
#         
#         for ax_name in target_axes:
#             hodograf_pts[ax_name].append(current_vectors[ax_name])
#             pts = np.array(hodograf_pts[ax_name])
#             hodograf_lines[ax_name].set_data(pts[:, 0], pts[:, 1])
#             hodograf_lines[ax_name].set_3d_properties(pts[:, 2])
#         
#         ax.set_title(f'Tumbling: Frame {i+1}/{len(rotation_matrices)}')
#         return collection, omega_quiver, osa_1, osa_2, osa_3, *hodograf_lines.values()
# 
#     ax.legend()
#     n_frames = len(rotation_matrices) // skip_frames
#     anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
#     
#     if output_file:
#         anim.save(output_file, writer='pillow', fps=fps)
#     else:
#         plt.show()
#         
#     # --- SAVE LAST FRAME AS PDF ---
#     last_i = len(rotation_matrices) - 1
#     R = rotation_matrices[last_i]
# 
#     rotated_vertices = (vertices_original - center) @ R.T + center
#     collection.set_verts(rotated_vertices)
# 
#     osa_1.remove()
#     osa_2.remove()
#     osa_3.remove()
#     v1_final = R @ np.array([1, 0, 0]) * axis_length
#     v2_final = R @ np.array([0, 1, 0]) * axis_length
#     v3_final = R @ np.array([0, 0, 1]) * axis_length
#     
#     osa_1 = ax.quiver(0, 0, 0, v1_final[0], v1_final[1], v1_final[2], color='red', linewidth=2)
#     osa_2 = ax.quiver(0, 0, 0, v2_final[0], v2_final[1], v2_final[2], color='green', linewidth=2)
#     osa_3 = ax.quiver(0, 0, 0, v3_final[0], v3_final[1], v3_final[2], color='blue', linewidth=2)
# 
#     w_vec = omega_vectors[last_i]
#     w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
#     omega_quiver.remove()
#     omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
# 
#     for ax_name in target_axes:
#         pts = np.array(hodograf_pts[ax_name])
#         if len(pts) > 0:
#             hodograf_lines[ax_name].set_data(pts[:, 0], pts[:, 1])
#             hodograf_lines[ax_name].set_3d_properties(pts[:, 2])
# 
#     ax.set_title("Final frame")
#     fig.savefig("last_frame.pdf", bbox_inches='tight')
#         
#     return anim
# =============================================================================
    
def animate_rotation(shape_file, rotation_matrices, L_hat, omega_vectors, 
                     output_file=None, fps=20, skip_frames=1, hodograf_axis='z'):
    """
    Parametar hodograf_axis određuje za šta se crta hodograf:
    - 'x' ili 1: Crvena osa tela (X)
    - 'y' ili 2: Zelena osa tela (Y)
    - 'z' ili 3: Plava osa tela (Z)
    - 'omega' ili 'w': Cijan osa trenutne ugaone brzine (Herpolhodogram)
    - 'all': Sve tri ose tela + Omega
    - None: Bez hodografa
    """
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
    
    # Fiksni vektor L (Crni)
    ax.quiver(0, 0, 0, L_hat[0]*axis_length, L_hat[1]*axis_length, L_hat[2]*axis_length, 
              color='k', label='L (Ugaoni moment)', linewidth=5)
    
    # Trenutna osa rotacije (Cijan)
    omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Omega (Trenutna osa)', linewidth=2)
    
    # Inicijalizacija glavnih osa
    osa_1 = ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', label='Glavna osa 1 (X)', linewidth=2)
    osa_2 = ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', label='Glavna osa 2 (Y)', linewidth=2)
    osa_3 = ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', label='Glavna osa 3 (Z)', linewidth=2)
    
    # --- PROŠIRENJE ZA HODOGRAF OMEGE ---
    hodograf_lines = {}
    hodograf_pts = {'x': [], 'y': [], 'z': [], 'omega': []}
    
    target_axes = []
    if hodograf_axis in ['x', 1, 'all']: target_axes.append('x')
    if hodograf_axis in ['y', 2, 'all']: target_axes.append('y')
    if hodograf_axis in ['z', 3, 'all']: target_axes.append('z')
    if hodograf_axis in ['omega', 'w', 'all']: target_axes.append('omega')
    
    colors = {'x': 'red', 'y': 'green', 'z': 'blue', 'omega': 'cyan'}
    labels = {'x': 'Hodograf X-ose', 'y': 'Hodograf Y-ose', 'z': 'Hodograf Z-ose', 'omega': 'Hodograf Omega (Herpolhodogram)'}
    
    for ax_name in target_axes:
        line, = ax.plot([], [], [], color=colors[ax_name], linestyle='--', label=labels[ax_name], lw=1.5)
        hodograf_lines[ax_name] = line

    vertices_original = vertices.copy()

    def update(frame):
        nonlocal omega_quiver, osa_1, osa_2, osa_3
        
        i = frame * skip_frames
        if i >= len(rotation_matrices):
            i = len(rotation_matrices) - 1
            
        print(i)
            
        R = rotation_matrices[i]
        
        # Rotacija tela
        rotated_vertices = (vertices_original - center) @ R + center
        collection.set_verts(rotated_vertices)
        
        # Update Omega vektora
        omega_quiver.remove()
        w_vec = omega_vectors[i]
        w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
        omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
        
        # Update glavnih osa inercije
        osa_1.remove()
        osa_2.remove()
        osa_3.remove()
        
        v1_rotated = R.T @ np.array([1, 0, 0]) * axis_length
        v2_rotated = R.T @ np.array([0, 1, 0]) * axis_length
        v3_rotated = R.T @ np.array([0, 0, 1]) * axis_length
        
        osa_1 = ax.quiver(0, 0, 0, v1_rotated[0], v1_rotated[1], v1_rotated[2], color='red', linewidth=2)
        osa_2 = ax.quiver(0, 0, 0, v2_rotated[0], v2_rotated[1], v2_rotated[2], color='green', linewidth=2)
        osa_3 = ax.quiver(0, 0, 0, v3_rotated[0], v3_rotated[1], v3_rotated[2], color='blue', linewidth=2)

        # Update hodografa uključujući i omegu
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
        
        ax.set_title(f'Tumbling: Frame {i+1}/{len(rotation_matrices)}')
        return collection, omega_quiver, osa_1, osa_2, osa_3, *hodograf_lines.values()

    ax.legend()
    n_frames = len(rotation_matrices) // skip_frames
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    if output_file:
        anim.save(output_file, writer='pillow', fps=fps)
    else:
        plt.show()
        
    # --- SAVE LAST FRAME AS PDF ---
    last_i = len(rotation_matrices) - 1
    R = rotation_matrices[last_i]

    rotated_vertices = (vertices_original - center) @ R.T + center
    collection.set_verts(rotated_vertices)

    osa_1.remove()
    osa_2.remove()
    osa_3.remove()
    v1_final = R @ np.array([1, 0, 0]) * axis_length
    v2_final = R @ np.array([0, 1, 0]) * axis_length
    v3_final = R @ np.array([0, 0, 1]) * axis_length
    
    osa_1 = ax.quiver(0, 0, 0, v1_final[0], v1_final[1], v1_final[2], color='red', linewidth=2)
    osa_2 = ax.quiver(0, 0, 0, v2_final[0], v2_final[1], v2_final[2], color='green', linewidth=2)
    osa_3 = ax.quiver(0, 0, 0, v3_final[0], v3_final[1], v3_final[2], color='blue', linewidth=2)

    w_vec = omega_vectors[last_i]
    w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
    omega_quiver.remove()
    omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)

    for ax_name in target_axes:
        pts = np.array(hodograf_pts[ax_name])
        if len(pts) > 0:
            hodograf_lines[ax_name].set_data(pts[:, 0], pts[:, 1])
            hodograf_lines[ax_name].set_3d_properties(pts[:, 2])

    ax.set_title("Final frame")
    fig.savefig("last_frame.pdf", bbox_inches='tight')
        
    return anim

def rodrigues_gemini(axis, angle):
    """Pomocna funkcija za Rodriguesovu formulu rotacije"""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



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
    





















# =============================================================================
# # --- 1. DEFENISANJE ULAZNIH PARAMETARA (Apofis - Tabela 2) ---
# I1, I2, I3 = 0.61, 0.965, 1.0
# I = (I1, I2, I3)
# 
# # Periodi i početne vrednosti
# P_phi_h = 27.38
# w_phi = (2 * np.pi) / (P_phi_h * 3600)
# 
# phi_start = np.deg2rad(152.0)
# psi_start = np.deg2rad(14.0)
# theta_start = np.deg2rad(37)  # Srednji ugao nutacije (theta_aver)
# 
# # Moment impulsa L u SAM režimu: L = I3 * w_phi
# L_fixed = I3 * w_phi
# 
# # Početne ugaone brzine (u sistemu tela)
# w1_0 = (L_fixed * np.sin(theta_start) * np.sin(psi_start)) / I1
# w2_0 = (L_fixed * np.sin(theta_start) * np.cos(psi_start)) / I2
# w3_0 = (L_fixed * np.cos(theta_start)) / I3
# y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start]
# =============================================================================


# --- 1. PARAMETRI IZ TABELE 2 ---
I1, I2, I3 = 0.61, 0.965, 1.0

phi_0 = np.deg2rad(152.0)
psi_0 = np.deg2rad(14.0)
E_ratio_target = 1.024
P_phi_h = 27.38

lambda_L = np.deg2rad(250.0)
beta_L = np.deg2rad(-75.0)


I = (I1, I2, I3)
# --- 2. EGZAKATAN PRORAČUN theta_0 IZ E/E0 I psi_0 ---
# A = (np.sin(psi_0)**2 / I1) + (np.cos(psi_0)**2 / I2)
# # E/E0 = I3 * (sin^2(theta)*A + cos^2(theta)/I3)
# # E/E0 - 1 = sin^2(theta) * (I3*A - 1)
# sin2_theta0 = (E_ratio_target - 1.0) / (I3 * A - 1.0)
# theta_0 = np.arcsin(np.sqrt(sin2_theta0))  # radijani (~35.38 deg)

# # --- 3. INTENZITET I SMER L ---
# w_phi = (2 * np.pi) / (P_phi_h * 3600)
# L_mag = I3 * w_phi

# # Vektor L u ekliptičkim inercijalnim koordinatama
# L_ecliptic_hat = np.array([
#     np.cos(beta_L) * np.cos(lambda_L),
#     np.cos(beta_L) * np.sin(lambda_L),
#     np.sin(beta_L)
# ])
# L_ecliptic = L_mag * L_ecliptic_hat

# # --- 4. POČETNE UGAONE BRZINE U SISTEMU TELA ---
# w1_0 = (L_mag * np.sin(theta_0) * np.sin(psi_0)) / I1
# w2_0 = (L_mag * np.sin(theta_0) * np.cos(psi_0)) / I2
# w3_0 = (L_mag * np.cos(theta_0)) / I3

# y0 = [w1_0, w2_0, w3_0, phi_0, theta_0, psi_0]

# 3. PRORAČUN POČETNOG NUTACIONOG UGLA theta_0 IZ E_ratio I psi_0
A = (np.sin(psi_0) ** 2 / I1) + (np.cos(psi_0) ** 2 / I2)
sin2_theta0 = (E_ratio_target - 1.0) / (I3 * A - 1.0)
theta_0 = np.arcsin(np.sqrt(sin2_theta0))

# 4. PRORAČUN UGAONIH BRZINA I VEKTORA POČETNOG STANJA y0
w_phi = (2 * np.pi) / (P_phi_h * 3600.0)

# UNIVERZALNO: Deljenje sa A radi i za I1 != I2 i za I1 == I2
L_mag = w_phi / A

w1_0 = (L_mag * np.sin(theta_0) * np.sin(psi_0)) / I1
w2_0 = (L_mag * np.sin(theta_0) * np.cos(psi_0)) / I2
w3_0 = (L_mag * np.cos(theta_0)) / I3

y0 = [w1_0, w2_0, w3_0, phi_0, theta_0, psi_0]



# Vremenska mreža (npr. simulacija u trajanju od 10 dana)
dt = 100  # Vremenski korak u sekundama (1 minut)
total_time_days = 100
timesteps = int((total_time_days * 86400) / dt)

# Pomoćni vektori za funkciju
r_inertial = np.array([1.0, 0.0, 0.0])
BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0])

# --- 2. POKRETANJE SIMULACIJE ---
res = compute_tumbling_dynamics(dt, timesteps, y0, I, r_inertial, BODY_AXIS_TO_TRACK)

# --- 3. IZDVAJANJE REZULTATA I PROVERA ---
time_hours = res['time'] / 3600.0
w1, w2, w3 = res['omega_body'][:, 0], res['omega_body'][:, 1], res['omega_body'][:, 2]

# Rotacioni uglovi kroz vreme (potrebno izvući iz rešenja ako nisu direktno u rečniku)
# U tvojoj funkciji sol.y[4, :] predstavlja theta(t)
# Za dijagnostiku računamo direktno nutacioni ugao iz ugaonog momenta i ose I3:
# cos(theta) = L_3 / |L| = (I3 * w3) / sqrt((I1*w1)^2 + (I2*w2)^2 + (I3*w3)^2)
L1 = I1 * w1
L2 = I2 * w2
L3 = I3 * w3
L_norm = np.sqrt(L1**2 + L2**2 + L3**2)

theta_deg = np.rad2deg(np.arccos(np.clip(L3 / L_norm, -1.0, 1.0)))

# Provera održanja kinetičke energije E i ugaonog momenta |L|
E_kin = 0.5 * (I1 * w1**2 + I2 * w2**2 + I3 * w3**2)
rel_err_E = np.abs(E_kin - E_kin[0]) / E_kin[0]
rel_err_L = np.abs(L_norm - L_norm[0]) / L_norm[0]

print(f"Maksimalna relativna greška energije: {np.max(rel_err_E):.2e}")
print(f"Maksimalna relativna greška ugaonog momenta: {np.max(rel_err_L):.2e}")
print(f"Opseg nutacionog ugla theta: {np.min(theta_deg):.2f}° do {np.max(theta_deg):.2f}°")

# --- 4. GRAFIČKA DIJAGNOSTIKA ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Graph 1: Evolucija nutacionog ugla theta(t)
axes[0].plot(time_hours, theta_deg, color='b', label=r'$\theta(t)$')
axes[0].axhline(y=12.0, color='r', linestyle='--', label=r'$\theta_{\min} = 12^\circ$')
axes[0].axhline(y=55.0, color='r', linestyle='--', label=r'$\theta_{\max} = 55^\circ$')
axes[0].set_ylabel(r'$\theta \ [^\circ]$')
axes[0].set_title('Evolucija nutacionog ugla (Nutational Motion)')
axes[0].grid(True)
axes[0].legend(loc='upper right')

# Graph 2: Ugaone brzine w1, w2, w3
axes[1].plot(time_hours, w1, label=r'$\omega_1$')
axes[1].plot(time_hours, w2, label=r'$\omega_2$')
axes[1].plot(time_hours, w3, label=r'$\omega_3$')
axes[1].set_ylabel(r'$\omega \ [\mathrm{rad/s}]$')
axes[1].set_title('Komponente ugaone brzine u sistemu tela')
axes[1].grid(True)
axes[1].legend(loc='upper right')

# Graph 3: Održanje energije i ugaonog momenta (Numerička stabilnost)
axes[2].semilogy(time_hours, rel_err_E + 1e-16, label=r'$\Delta E / E_0$')
axes[2].semilogy(time_hours, rel_err_L + 1e-16, label=r'$\Delta |L| / |L_0|$')
axes[2].set_xlabel('Vreme [h]')
axes[2].set_ylabel('Relativna greška')
axes[2].set_title('Numerička konzervacija invarijanti (DOP853 Integrator)')
axes[2].grid(True)
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
