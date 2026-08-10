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
    
    
    Input:
        
    dt: vremenski korak
    
    timesteps: broj vremenskih koraka
    
    y0: pocetno stanje (vidi liniju 404)
    
    I: momenti inercije za 3 glavne ose
    
    r_inertial: vektor u inercijalnom prostoru za koji dobijamo koordinate u koordinatnom sistemu vezanom za asteroid 
    (ovo moze biti vektor ka Suncu ili transferzalni vektor koji nam je potreban za Jarkovski)
    
    BODY_AXIS_TO_TRACK: osa krutovezana za asteroid ciji polozaj mozemo da pratimo u inercijalnom prostoru
    ovo je za proveru da li tumbling radi dobro (mislim da se za sada ova osa ne prati dobro)
    
    
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
        
        rotations.append(Ri.T) # Telo -> Inercijalni
        r_body_coords.append(Ri @ r_inertial)  # inercijalni vektor (npr. ka Suncu u body-fixed sistemu)
        spin_axis_iner.append(Ri.T @ w_body) # trenutna osa rotacije u inercijalnom sistemu
        body_axis_iner.append(Ri.T @ BODY_AXIS_TO_TRACK) # osa tela koju pratimo u inercijalnom sistemu

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
        'last_state': last_state  # DODATO: spremno za sledeći y0
    }
    
    



























def tumbling_rotation_1(dt, timesteps, y0, I, r_inertial):
    """
    Ovo je skracena funkcija compute_tumbling_dynamics koja za izlaz daje samo ono sto nam treba za racunanje Jarkovskog
    
    treba proveriti da li su izlazni vektori dobri:
        r_body_coords (ka Suncu),
        t_body_coords (transferzalni, upravan na pravac ka suncu a lezi u orbitalnoj ravni)

    """
    I1, I2, I3 = I
    
    
    print(np.shape(r_inertial))
    t_inertial = np.cross(r_inertial, np.array([0, 0, 1]))  # THIS IS OK
    
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
    
    last_state = sol.y[:, -1].tolist()
    
    r_body_coords = []
    t_body_coords = []
    
    

    for i in range(len(sol.t)):
        phi_i, theta_i, psi_i = sol.y[3, i], sol.y[4, i], sol.y[5, i]

        ci, si = np.cos(phi_i), np.sin(phi_i)
        cj, sj = np.cos(theta_i), np.sin(theta_i)
        ck, sk = np.cos(psi_i), np.sin(psi_i)
        
        Ri = np.array([
            [ ck*ci - sk*cj*si,  ck*si + sk*cj*ci, sk*sj],
            [-sk*ci - ck*cj*si, -sk*si + ck*cj*ci, ck*sj],
            [ sj*si,            -sj*ci,            cj]
        ])
        
        r_body_coords.append(Ri @ r_inertial)
        t_body_coords.append(Ri @ t_inertial)

    return {
        'r_body_fixed': r_body_coords,
        't_body_fixed': t_body_coords,
        'last_state': last_state  # DODATO: spremno za sledeći y0
    }


def animate_rotation(shape_file, rotation_matrices, L_hat, omega_vectors, output_file=None, fps=20, skip_frames=1):
    '''
    Ovo ne treba dirati. Ova funkcija samo pravi animaciju.
    
    '''
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
    
    # --- NOVI ELEMENTI ---
    axis_length = max_range * 1.2
    # Fiksni vektor L (Plava)
    ax.quiver(0, 0, 0, L_hat[0]*axis_length, L_hat[1]*axis_length, L_hat[2]*axis_length, 
              color='k', label='L (Ugaoni moment)', linewidth=5)
    
    # Inicijalizacija strele za trenutnu osu rotacije (Cijan)
    omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Omega (Trenutna osa)', linewidth=2)
    
    # Inicijalizacija hodografa (Zuta linija)
    hodograf_line, = ax.plot([], [], [], color='red', label='Hodograf', lw=1.5)
    hodograf_pts = []

    # Inicijalizacija ose inercije 1 (Crvena)
    # Postavljamo je inicijalno duž x-ose
    osa_1_dir = np.array([1, 0, 0]) * axis_length
    osa_1 = ax.quiver(0, 0, 0, osa_1_dir[0], osa_1_dir[1], osa_1_dir[2], color='red', label='Glavna osa 1', linewidth=2)
    
    osa_2_dir = np.array([0, 1, 0]) * axis_length
    osa_2 = ax.quiver(0, 0, 0, osa_2_dir[0], osa_2_dir[1], osa_2_dir[2], color='green', label='Glavna osa 2', linewidth=2)
    
    osa_3_dir = np.array([0, 0, 1]) * axis_length
    osa_3 = ax.quiver(0, 0, 0, osa_3_dir[0], osa_3_dir[1], osa_3_dir[2], color='blue', label='Glavna osa 3', linewidth=2)
    
    # ---------------------

    vertices_original = vertices.copy()

    def update(frame):
        i = frame * skip_frames
        print(i)
        if i >= len(rotation_matrices):
            i = len(rotation_matrices) - 1
            
        R = rotation_matrices[i]
        
        # Rotacija asteroida
        rotated_vertices = (vertices_original - center) @ R.T + center
        collection.set_verts(rotated_vertices)
        
        # --- UPDATE OSE I HODOGRAFA ---
        # Update trenutne ose rotacije (Omega)
        nonlocal omega_quiver, osa_1, osa_2, osa_3
        omega_quiver.remove()
        w_vec = omega_vectors[i]
        w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
        omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
        
        # Update ose inercije 1 (Prati rotaciju R)
        osa_1.remove()
        osa_2.remove()
        osa_3.remove()
        # Inicijalni pravac (1,0,0) rotiramo matricom R
        v1_rotated = R @ np.array([1, 0, 0]) * axis_length
        v2_rotated = R @ np.array([0, 1, 0]) * axis_length
        v3_rotated = R @ np.array([0, 0, 1]) * axis_length
        osa_1 = ax.quiver(0, 0, 0, v1_rotated[0], v1_rotated[1], v1_rotated[2], color='red', linewidth=2)
        osa_2 = ax.quiver(0, 0, 0, v2_rotated[0], v2_rotated[1], v2_rotated[2], color='green', linewidth=2)
        osa_3 = ax.quiver(0, 0, 0, v3_rotated[0], v3_rotated[1], v3_rotated[2], color='blue', linewidth=2)

        # Update hodografa
        hodograf_pts.append(w_dir)
        pts = np.array(hodograf_pts)
        hodograf_line.set_data(pts[:, 0], pts[:, 1])
        hodograf_line.set_3d_properties(pts[:, 2])
        # ------------------------------
        
        ax.set_title(f'Tumbling: Frame {i+1}/{len(rotation_matrices)}')
        return collection, omega_quiver, osa_1, hodograf_line

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

    # Update osa_1 za finalni frame
    osa_1.remove()
    v1_final = R @ np.array([1, 0, 0]) * axis_length
    v2_final = R @ np.array([0, 1, 0]) * axis_length
    v3_final = R @ np.array([0, 0, 1]) * axis_length
    osa_1 = ax.quiver(0, 0, 0, v1_final[0], v1_final[1], v1_final[2], color='red', linewidth=2)
    osa_2 = ax.quiver(0, 0, 0, v2_final[0], v2_final[1], v2_final[2], color='green', linewidth=2)
    osa_3 = ax.quiver(0, 0, 0, v3_final[0], v3_final[1], v3_final[2], color='blue', linewidth=2)


    # Update omega vector
    w_vec = omega_vectors[last_i]
    w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
    omega_quiver.remove()
    omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)

    pts = np.array(hodograf_pts)
    if len(pts) > 0:
        hodograf_line.set_data(pts[:, 0], pts[:, 1])
        hodograf_line.set_3d_properties(pts[:, 2])
    
    return anim




if __name__ == "__main__":

    shape_file = "Apophis.stl"
    output_file = "tumbling_animation.gif"
    # Parametri simulacije
    fps = 30 # ovo ne treba dirati (sluzi samo za brzinu animacije)
    dt = 1000 # vremenski korak
    timesteps = 500 # broj koraka

        # --- 1. PARAMETRI IZ TABELE 2 (Apofis) ---
    I1, I2, I3 = 0.61, 0.965, 1.0
    P_phi_h = 27.38
    w_phi = (2 * np.pi) / (P_phi_h * 3600)

    # Početni uslovi (SAM režim - Short Axis Mode)
    theta_start = np.deg2rad(56)
    psi_start = np.deg2rad(14.0)
    phi_start = np.deg2rad(152.0)
    


    # Izračunavanje momenta L i početnih w komponenti (mislim da je kriticno proveriti da li je ovo dobro)
    L_fixed = w_phi * ((I1 + I2) / 2) / np.cos(theta_start)
    w1_0 = (L_fixed * np.sin(theta_start) * np.sin(psi_start)) / I1
    w2_0 = (L_fixed * np.sin(theta_start) * np.cos(psi_start)) / I2
    w3_0 = (L_fixed * np.cos(theta_start)) / I3
    y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start] # pocetni uslovvi 
    
    
    # Vektori za praćenje
    V_INERTIAL_FIXED = np.array([1.0, 0.0, 0.0]) # Npr. pravac ka Suncu
    BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0]) # Najkraća osa (I3)
    

    # --- PRVA SIMULACIJA (Prva 3 dana) ---
    sim_data_1 = compute_tumbling_dynamics(
        dt, timesteps, y0, np.array([I1, I2, I3]), V_INERTIAL_FIXED, BODY_AXIS_TO_TRACK
    )
    
    
    # FINKCIJA KOJA CE ICI ZA YARKOVSKI KOJA U SVAKOM TRENUTKU VRACA RADIJALNI I TRANSFERZALNI VEKTOR
    
    sim_data_11 = tumbling_rotation_1(dt, timesteps, y0, np.array([I1, I2, I3]), V_INERTIAL_FIXED)
    

    
    print("Computing kinematics (rotations, axes, and hodograph)...")
    


    # Izdvajamo ono što nam treba za proveru
    rotations = sim_data_1['rotations']
    L_hat = sim_data_1['L_axis']
    omega_vecs = sim_data_1['omega_axes']
    r_sun = sim_data_1['r_sun']
    
    
    # ovo je sve sto nam treba za Yarko
    r_1 = sim_data_11['r_body_fixed']
    t_1 = sim_data_11['t_body_fixed']
    
    long1 = np.zeros(len(r_sun))
    lat1 = np.zeros(len(r_sun))
    long11 = np.zeros(len(r_sun))
    lat11 = np.zeros(len(r_sun))
    
    long11_t = np.zeros(len(r_sun))
    lat11_t = np.zeros(len(r_sun))

    # provera da li obe fnkcije imaju isti izlaz
    for i in range(len(long1)):
        long1[i] = np.arctan2(r_sun[i][1], r_sun[i][0])
        lat1[i] = np.arcsin(r_sun[i][2])
        
        long11[i] = np.arctan2(r_1[i][1], r_1[i][0])
        lat11[i] = np.arcsin(r_1[i][2])
        
        long11_t[i] = np.arctan2(t_1[i][1], t_1[i][0])
        lat11_t[i] = np.arcsin(t_1[i][2])
        
        
    
    print(f"Computed {len(rotations)} frames.")
    
    
    
    # Pozivamo osveženu funkciju za animaciju
    # Prosleđujemo i L_hat (fiksni) i omega_vecs (promenljivi za hodograf)
    animate_rotation(
        shape_file=shape_file,
        rotation_matrices=rotations,
        L_hat=L_hat,
        omega_vectors=omega_vecs,
        output_file=output_file,
        fps=fps,
        skip_frames=1  # Možeš povećati na 2 ili 3 ako je fajl prevelik
    )
    
    
   
    print(f"Animation saved to {output_file}")
 
