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

def rodrigues(u, theta):
    """Return 3×3 rotation matrix rotating by theta around unit-vector u."""
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + ux*ux*C,    ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz*uz*C   ]
    ])
    
    
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


def compute_tumbling_matrices(timesteps, ratio_im_il, ratio_is_il,
                             spinrate, tilt, cone, max_precessions=3, max_spins=100):
    """Compute free-precession rotation matrices based on provided arguments."""

    # --- Moments of inertia ---
    Ilong   = 1.0
    Imiddle = ratio_im_il * Ilong
    Ishort  = ratio_is_il * Ilong
    Ibody   = np.diag([Ilong, Imiddle, Ishort])
    Iinv    = np.linalg.inv(Ibody) # Constant in body frame

    # --- Initial angular momentum vector in the body frame (before cone tilt) ---
    theta_tilt = np.radians(tilt)
    L_dir = np.array([np.sin(theta_tilt), 0.0, np.cos(theta_tilt)])
    Lmag  = Ishort * spinrate # Magnitude definition based on initial spin around short axis
    L_body0 = Lmag * L_dir

    # --- Precession rate and period ratio r = T_prec/T_spin ---
    # WARNING: This formula's applicability to asymmetric rotors might be limited.
    # It approximates the precession frequency when L is near the short axis.
    omega_prec = abs(Lmag) * abs((1.0/Imiddle) - (1.0/Ishort))
    
    
    
    print(f'omega prec = {omega_prec}')
    if omega_prec == 0:
        # Handle cases like sphere (I1=I2=I3) or symmetric rotor tilted along symmetry axis
        print("Warning: Calculated precession rate is zero. Using spin rate only.")
        # Avoid division by zero; assign a very small value, motion will be mostly spin.
        omega_prec = 1e-9
    r = spinrate / omega_prec # Ratio of initial spin rate to precession rate

    # --- Rational approximation search (p spins per q precessions) ---
    best = {'p': None, 'q': None, 'err': np.inf, 'r_approx': None} # Initialize r_approx
    found_valid_approximation = False
    for q in range(1, max_precessions + 1):
        p_float = r * q
        p = int(round(p_float))

        # Basic validity check
        if p < 1:
            print(f"  Skipping q={q}: resultant p={p} < 1")
            continue

        # Check against max_spins constraint
        if p > max_spins:
            print(f"  Skipping q={q}: p={p} exceeds max_spins={max_spins}")
            continue

        # This is a valid candidate within constraints
        found_valid_approximation = True
        r_approx = p / q
        err = abs(r_approx - r)

        # Check if this is the best one found so far
        if err < best['err']:
            best.update(p=p, q=q, err=err, r_approx=r_approx)

    # --- Handle case where no valid approximation was found ---
    if not found_valid_approximation:
        # Construct informative error message
        error_message = (
            f"Could not find a suitable rational approximation (p spins / q precessions) "
            f"within the specified limits (max_precessions={max_precessions}, max_spins={max_spins}).\n"
            f"Target ratio r = spinrate / omega_prec = {r:.4f}.\n"
            f"Consider increasing max_spins or max_precessions, or check input parameters "
            f"(tilt, moment ratios, spinrate)."
        )
        # Try to find the *closest* approximation ignoring max_spins, just for reporting
        closest_p_unconstrained = int(round(r * 1)) # Closest for q=1
        if closest_p_unconstrained >= 1:
             error_message += (f"\nFor q=1, the closest integer number of spins is p={closest_p_unconstrained}, "
                               f"which might exceed max_spins.")

        raise ValueError(error_message)


    # --- Report chosen approximation ---
    print(f"Target ratio r = T_prec/T_spin: {r:.6f}")
    print(f"Using omega_prec = |L| * |1/Im - 1/Is| = {omega_prec:.6f}")
    print(f"Best approx within limits: p={best['p']} spins per q={best['q']} precessions") # Clarified output
    print(f"Number of spins per precession cycle (p): {best['p']}")
    print(f" => approximate ratio: {best['r_approx']:.6f}, error = {best['err']:.6e}")

    # --- Time stepping ---
    # Calculate dt based on the desired number of steps per *approximate* precession cycle
    dt = (2 * np.pi / omega_prec) / timesteps
    total_steps = timesteps * best['q']
    print(f"Simulating {best['q']} approximate precession cycles.")
    print(f"Total timesteps = {total_steps}, dt = {dt:.6e}")

    # --- Initialize orientation and storage ---
    orientation = np.eye(3) # Represents transformation from body to space frame
    # Apply initial cone opening (rotation around space Y-axis before simulation starts)
    theta_cone = np.radians(cone)
    R_cone = rodrigues(np.array([0,1,0]), theta_cone)
    orientation = R_cone.dot(orientation) # Initial orientation matrix R(t=0)

    # Calculate constant angular momentum vector in space frame
    # L_space = R(t=0) * L_body(t=0)
    L_space = orientation @ L_body0

    rotations = [orientation.copy()] # Store initial orientation R(t=0)

    # --- Time integration loop ---
    for step in range(total_steps):
        # Calculate current angular momentum and velocity in the *body* frame
        L_body = orientation.T.dot(L_space) # L_body(t) = R(t)^T * L_space
        w_body = Iinv.dot(L_body)           # w_body(t) = I_body^-1 * L_body(t)
        w_norm = np.linalg.norm(w_body)

        # Calculate incremental rotation matrix (rotation in body frame over dt)
        # If w_norm is zero, no rotation occurs.
        deltaR = rodrigues(w_body / w_norm, w_norm * dt) if w_norm > 1e-15 else np.eye(3)

        # Update orientation: R(t+dt) = deltaR(t) * R(t)
        orientation = deltaR.dot(orientation)
        rotations.append(orientation.copy()) # Store R(t = (step+1)*dt)

    # --- Finalize ---
    # Return orientations from t=dt to t=total_steps*dt
    # Exclude the initial t=0 orientation stored before the loop
    rotations = np.stack(rotations[1:])  # shape (total_steps, 3, 3)
    print(f"Computed rotation matrix array with shape {rotations.shape}.")
    return rotations



# =============================================================================
# 
# # OVO RADI
# def compute_tumbling_kinematics(timesteps, long_deg, lat_deg, theta_deg, P_spin, P_prec, n_cycles=1):
#     """
#     Simulacija složene rotacije prema tvojim uputstvima:
#     - long, lat: Pravac vektora ugaonog momenta L u prostoru
#     - theta_deg: Ugao između glavne ose inercije i vektora L (mislim da nije nego je ugao izmedju 0,0,1 i glavne ose)
#     - P_spin: Period rotacije asteroida oko sopstvene ose
#     - P_prec: Period precesije glavne ose oko vektora L
#     
#     
#     Treba animirati i ose oko kojih se rotira, kao i osu precesije, i hodograf ose rotacije
#     """
#     
#     # 1. Definisanje vektora ugaonog momenta L u prostoru (fiksiran)
#     lon_rad = np.radians(long_deg)
#     lat_rad = np.radians(lat_deg)
#     L_hat = np.array([
#         np.cos(lat_rad) * np.cos(lon_rad),
#         np.cos(lat_rad) * np.sin(lon_rad),
#         np.sin(lat_rad)
#     ])
# 
#     # 2. Definisanje vremenskog koraka
#     # Koristimo duži period kao bazu za ukupno vreme simulacije
#     total_time = n_cycles * max(P_spin, P_prec)
#     dt = total_time / timesteps
#     
#     omega_spin = 2 * np.pi / P_spin
#     omega_prec = 2 * np.pi / P_prec
#     theta = np.radians(theta_deg)
# 
#     rotations = []
# 
#     for i in range(timesteps):
#         t = i * dt
#         
#         # --- KORAK A: Rotacija oko glavne ose inercije (Spin) ---
#         # Pretpostavljamo da je glavna osa inicijalno Z osa tela
#         psi = omega_spin * t
#         R_spin = rodrigues_gemini(np.array([0, 0, 1]), psi)
#         
#         # --- KORAK B: Nagib za ugao theta ---
#         # Ovo postavlja glavnu osu pod traženi ugao u odnosu na L
#         # Rotiramo oko Y ose da bismo otklonili Z osu od vertikale
#         R_tilt = rodrigues_gemini(np.array([0, 1, 0]), theta)
# 
#         # --- KORAK C: Precesija oko vektora L ---
#         # Glavna osa sada kruži oko vektora L u prostoru
#         phi = omega_prec * t
#         R_prec = rodrigues_gemini(L_hat, phi)
#         
#         # --- KOMPOZICIJA ---
#         # Redosled: prvo spin, pa nagib, pa precesija u prostoru
#         # R_total transformiše vektor iz tela u prostor
#         orientation = R_prec @ R_tilt @ R_spin
#         
#         rotations.append(orientation)
# 
#     return np.stack(rotations)
# =============================================================================


import numpy as np

def compute_tumbling_kinematics(timesteps, long_deg, lat_deg, theta_deg, P_spin, P_prec, n_cycles=1):
    # 1. Pravac ugaonog momenta L (fiksiran u prostoru)
    lon_rad = np.radians(long_deg)
    lat_rad = np.radians(lat_deg)
    L_hat = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # 2. Parametri
    total_time = n_cycles * max(P_spin, P_prec)
    dt = total_time / timesteps
    omega_spin_mag = 2 * np.pi / P_spin
    omega_prec_mag = 2 * np.pi / P_prec
    theta = np.radians(theta_deg)

    # Liste za čuvanje podataka
    rotations = []
    G_axes = []      # Glavna osa inercije u prostoru
    omega_vecs = []  # Trenutni vektor ugaone brzine (za hodograf)

    # Inicijalna glavna osa u lokalnom sistemu (Z-osa tela)
    G_body = np.array([0, 0, 1])

    for i in range(timesteps):
        t = i * dt
        
        # --- Kinematika ---
        # R_spin: Rotacija oko sopstvene ose
        R_spin = rodrigues(G_body, omega_spin_mag * t)
        
        # R_tilt: Nagib glavne ose (Z) u odnosu na L (ako je L inicijalno Z)
        # Ovde koristimo Y osu za tilt da bismo dobili otklon
        R_tilt = rodrigues(np.array([0, 1, 0]), theta)

        # R_prec: Precesija oko fiksne ose L
        R_prec = rodrigues(L_hat, omega_prec_mag * t)
        
        # --- Matrica orijentacije (Body to Space) ---
        orientation = R_prec @ R_tilt @ R_spin
        rotations.append(orientation)

        # --- Trenutni položaji osa u prostoru ---
        # 1. Trenutni pravac glavne ose inercije G (ne zavisi od spina, samo od precesije)
        G_space = R_prec @ R_tilt @ G_body
        G_axes.append(G_space)

        # 2. Trenutna ukupna ugaona brzina omega
        # omega = omega_prec (oko L) + omega_spin (oko trenutne ose G)
        w_vec = (omega_prec_mag * L_hat) + (omega_spin_mag * G_space)
        omega_vecs.append(w_vec)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,             # Fiksni vektor
        'G_axes': np.stack(G_axes),   # Niz vektora kroz vreme
        'omega_axes': np.stack(omega_vecs), # Za hodograf (omega kroz vreme)
        'time': np.linspace(0, total_time, timesteps)
    }
    




def compute_tumbling_kinematics_1(timesteps, long_deg, lat_deg, theta_deg, P_spin, P_prec, n_cycles=1):
    # 1. Pravac ugaonog momenta L (fiksiran u prostoru)
    lon_rad = np.radians(long_deg)
    lat_rad = np.radians(lat_deg)
    L_hat = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # 2. Parametri
    total_time = n_cycles * max(P_spin, P_prec)
    dt = total_time / timesteps
    omega_spin_mag = 2 * np.pi / P_spin
    omega_prec_mag = 2 * np.pi / P_prec
    theta = np.radians(theta_deg)

    rotations = []
    G_axes = []      
    omega_vecs = []  

    # Inicijalna glavna osa u lokalnom sistemu (Z-osa tela)
    G_body = np.array([0, 0, 1])

    # Da bismo nagnuli G za ugao theta u odnosu na L, 
    # moramo naći pomoćnu osu koja je normalna na L
    # (npr. kros proizvod L i neke proizvoljne ose)
    if abs(L_hat[2]) < 0.9:
        ortho_axis = np.cross(L_hat, np.array([0, 0, 1]))
    else:
        ortho_axis = np.cross(L_hat, np.array([1, 0, 0]))
    ortho_axis /= np.linalg.norm(ortho_axis)

    # Inicijalni nagib: postavljamo G tako da zaklapa theta sa L
    # R_initial_tilt postavlja G_space_0
    R_initial_tilt = rodrigues(ortho_axis, theta)
    G_space_0 = R_initial_tilt @ L_hat

    for i in range(timesteps):
        t = i * dt
        
        # --- Kinematika ---
        # 1. Precesija glavne ose G oko vektora L
        phi = omega_prec_mag * t
        R_prec = rodrigues(L_hat, phi)
        G_space = R_prec @ G_space_0
        G_axes.append(G_space)

        # 2. Rotacija tela oko te pokretne glavne ose G
        psi = omega_spin_mag * t
        R_spin_around_G = rodrigues(G_space, psi)
        
        # 3. Ukupna orijentacija (kumulativna)
        # Primenjujemo inicijalni nagib, pa precesiju, pa spin
        # Da bismo dobili matricu koja transformiše iz tela u prostor:
        # Prvo rotiramo telo oko Z_body (spin), pa ga nagnemo, pa vrtimo oko L
        # Ali pošto G_space već sadrži precesiju i nagib, koristimo to:
        
        # Inicijalna matrica koja poravnava G_body sa G_space_0
        # (Ovo je matematički čistije preko baze)
        orientation = R_spin_around_G @ R_prec @ R_initial_tilt
        # Napomena: Možda će biti potreban dodatni korak poravnanja osa 
        # zavisno od tvoje početne orijentacije modela, ali ovo je geometrijski core.
        
        rotations.append(orientation)

        # --- Trenutna ukupna ugaona brzina omega ---
        w_vec = (omega_prec_mag * L_hat) + (omega_spin_mag * G_space)
        omega_vecs.append(w_vec)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,
        'G_axes': np.stack(G_axes),
        'omega_axes': np.stack(omega_vecs),
        'time': np.linspace(0, total_time, timesteps)
    }
    





def compute_tumbling_kinematics_2(timesteps, long_deg, lat_deg, theta_deg, phi_0_deg, P_spin, P_prec, n_cycles=1):
    # 1. Pravac ugaonog momenta L (fiksiran u prostoru)
    lon_rad = np.radians(long_deg)
    lat_rad = np.radians(lat_deg)
    L_hat = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # 2. Parametri
    total_time = n_cycles * max(P_spin, P_prec)
    dt = total_time / timesteps
    omega_spin_mag = 2 * np.pi / P_spin
    omega_prec_mag = 2 * np.pi / P_prec
    theta = np.radians(theta_deg)
    phi_0 = np.radians(phi_0_deg)

    rotations = []
    G_axes = []      
    omega_vecs = []  

    # Inicijalna glavna osa u lokalnom sistemu (Z-osa tela)
    G_body = np.array([0, 0, 1])

    # Pronalaženje pomoćne ose normalne na L
    if abs(L_hat[2]) < 0.9:
        ortho_axis = np.cross(L_hat, np.array([0, 0, 1]))
    else:
        ortho_axis = np.cross(L_hat, np.array([1, 0, 0]))
    ortho_axis /= np.linalg.norm(ortho_axis)

    # NOVO: Rotiramo ortho_axis oko L za ugao phi_0 da bismo postavili početni azimut
    R_azimuth = rodrigues(L_hat, phi_0)
    ortho_axis_rotated = R_azimuth @ ortho_axis

    # Inicijalni nagib: postavljamo G_space_0 koristeći rotirani ortho_axis
    R_initial_tilt = rodrigues(ortho_axis_rotated, theta)
    G_space_0 = R_initial_tilt @ L_hat

    for i in range(timesteps):
        t = i * dt
        
        # --- Kinematika ---
        # 1. Precesija glavne ose G oko vektora L
        phi = omega_prec_mag * t
        R_prec = rodrigues(L_hat, phi)
        G_space = R_prec @ G_space_0
        G_axes.append(G_space)

        # 2. Rotacija tela oko te pokretne glavne ose G
        psi = omega_spin_mag * t
        R_spin_around_G = rodrigues(G_space, psi)
        
        # 3. Ukupna orijentacija (kompozicija preostaje ista kao u tvom originalu)
        orientation = R_spin_around_G @ R_prec @ R_initial_tilt
        
        rotations.append(orientation)

        # --- Trenutna ukupna ugaona brzina omega ---
        w_vec = (omega_prec_mag * L_hat) + (omega_spin_mag * G_space)
        omega_vecs.append(w_vec)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,
        'G_axes': np.stack(G_axes),
        'omega_axes': np.stack(omega_vecs),
        'time': np.linspace(0, total_time, timesteps)
    }
    
    
    
def compute_tumbling_kinematics_3(timesteps, long_deg, lat_deg, theta_deg, phi_0_deg, G_body, P_spin, P_prec, n_cycles=1):
    # 1. Pravac ugaonog momenta L (fiksiran u prostoru)
    lon_rad = np.radians(long_deg)
    lat_rad = np.radians(lat_deg)
    L_hat = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # 2. Parametri
    total_time = n_cycles * max(P_spin, P_prec)
    dt = total_time / timesteps
    omega_spin_mag = 2 * np.pi / P_spin
    omega_prec_mag = 2 * np.pi / P_prec
    theta = np.radians(theta_deg)
    phi_0 = np.radians(phi_0_deg)
    
    # Osiguravamo da je G_body jedinični vektor
    G_body = np.array(G_body) / np.linalg.norm(G_body)

    rotations = []
    G_axes = []      
    omega_vecs = []  

    # Pronalaženje pomoćne ose normalne na L za definisanje nagiba
    if abs(L_hat[2]) < 0.9:
        ortho_axis = np.cross(L_hat, np.array([0, 0, 1]))
    else:
        ortho_axis = np.cross(L_hat, np.array([1, 0, 0]))
    ortho_axis /= np.linalg.norm(ortho_axis)

    # Rotiramo ortho_axis oko L za početni azimut phi_0
    R_azimuth = rodrigues(L_hat, phi_0)
    ortho_axis_rotated = R_azimuth @ ortho_axis

    # Inicijalni nagib: Postavljamo ciljni pravac G_space_0 u prostoru
    R_tilt_L = rodrigues(ortho_axis_rotated, theta)
    G_space_0 = R_tilt_L @ L_hat

    # DODATAK: Matrica koja inicijalno poravnava G_body sa G_space_0
    # Koristimo pomoćnu funkciju za rotaciju između dva vektora
    R_align = rotation_matrix_from_vectors(G_body, G_space_0)

    for i in range(timesteps):
        t = i * dt
        
        # --- Kinematika ---
        # 1. Precesija pravca G oko L
        phi = omega_prec_mag * t
        R_prec = rodrigues(L_hat, phi)
        G_space = R_prec @ G_space_0
        G_axes.append(G_space)

        # 2. Rotacija tela oko te pokretne ose G_space
        psi = omega_spin_mag * t
        R_spin_around_G = rodrigues(G_space, psi)
        
        # 3. Ukupna orijentacija
        # Redosled: Inicijalno poravnanje -> Precesija -> Spin oko G
        orientation = R_spin_around_G @ R_prec @ R_align
        
        rotations.append(orientation)

        # --- Trenutna ukupna ugaona brzina ---
        w_vec = (omega_prec_mag * L_hat) + (omega_spin_mag * G_space)
        omega_vecs.append(w_vec)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,
        'G_axes': np.stack(G_axes),
        'omega_axes': np.stack(omega_vecs),
        'time': np.linspace(0, total_time, timesteps)
    }






def rotation_matrix_from_vectors(vec1, vec2):
    """ Pronalazi matricu rotacije koja prevodi vec1 u vec2 (jedinični vektori) """
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(v)
    if s < 1e-10: return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))














def animate_rotation(shape_file, rotation_matrices, L_hat, omega_vectors, output_file=None, fps=20, skip_frames=1):
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
    
    # --- NOVI ELEMENTI ---
    axis_length = max_range * 1.2
    # Fiksni vektor L (Plava)
    ax.quiver(0, 0, 0, L_hat[0]*axis_length, L_hat[1]*axis_length, L_hat[2]*axis_length, 
              color='blue', label='L (Ugaoni moment)', linewidth=2)
    
    # Inicijalizacija strele za trenutnu osu rotacije (Cijan)
    omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Omega (Trenutna osa)', linewidth=2)
    
    # Inicijalizacija hodografa (Zuta linija)
    hodograf_line, = ax.plot([], [], [], color='red', label='Hodograf', lw=1.5)
    hodograf_pts = []
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
        nonlocal omega_quiver
        omega_quiver.remove()
        w_vec = omega_vectors[i]
        # Normalizujemo i skaliramo za prikaz
        w_dir = (w_vec / np.linalg.norm(w_vec)) * axis_length
        omega_quiver = ax.quiver(0, 0, 0, w_dir[0], w_dir[1], w_dir[2], color='cyan', linewidth=2)
        
        # Update hodografa (dodajemo vrh omega vektora u niz)
        hodograf_pts.append(w_dir)
        pts = np.array(hodograf_pts)
        hodograf_line.set_data(pts[:, 0], pts[:, 1])
        hodograf_line.set_3d_properties(pts[:, 2])
        # ------------------------------
        
        ax.set_title(f'Tumbling: Frame {i+1}/{len(rotation_matrices)}')
        return collection, omega_quiver, hodograf_line

    ax.legend()
    n_frames = len(rotation_matrices) // skip_frames
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    if output_file:
        anim.save(output_file, writer='pillow', fps=fps)
    else:
        plt.show()
        
    return anim

#def compute_real_tumbling(duration, dt, ratio_im_il, ratio_is_il, spinrate, tilt, cone):
#    """
#    Simulira realno, neperiodično kretanje sa početnim nagibom sistema (cone).
#    
#    duration: ukupno vreme u sekundama
#    dt: vremenski korak integracije
#    ratio_im_il: Imiddle / Ilong
#    ratio_is_il: Ishort / Ilong
#    spinrate: početna brzina rotacije
#    tilt: ugao između L i ose asteroida (u stepenima)
#    cone: ugao nagiba celog sistema u prostoru (u stepenima)
#    """
#    # --- 1. Momenti inercije ---
#    Ilong   = 1.0
#    Imiddle = ratio_im_il * Ilong
#    Ishort  = ratio_is_il * Ilong
#    Iinv    = np.diag([1.0/Ilong, 1.0/Imiddle, 1.0/Ishort])
#
#    # --- 2. Inicijalni ugaoni moment u lokalnom sistemu (body frame) ---
#    theta_tilt = np.radians(tilt)
#    # L_dir postavlja pravac ugaonog momenta unutar asteroida
#    L_dir_body = np.array([np.sin(theta_tilt), 0.0, np.cos(theta_tilt)])
#    Lmag  = Ishort * spinrate 
#    L_body0 = Lmag * L_dir_body
#
#    # --- 3. Početna orijentacija (Space Frame) ---
#    # Inicijalno, body ose se poklapaju sa space osama
#    orientation = np.eye(3) 
#
#    # Primenjujemo CONE nagib oko Y-ose prostora
#    theta_cone = np.radians(cone)
#    if abs(theta_cone) > 1e-12:
#        # Rodriguesova matrica rotacije za fiksni nagib sistema
#        K_y = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
#        R_cone = np.eye(3) + np.sin(theta_cone)*K_y + (1 - np.cos(theta_cone))*(K_y @ K_y)
#        orientation = R_cone @ orientation
#
#    # FIKSIRAMO L u prostoru na osnovu početne orijentacije
#    # Od ovog trenutka, L_space se više ne menja (zakon održanja)
#    L_space = orientation @ L_body0
#    
#    total_steps = int(duration / dt)
#    rotations = []
#
#    # --- 4. Integraciona petlja ---
#    for _ in range(total_steps):
#        # Projektujemo konstantni L nazad u rotirajući asteroid (body frame)
#        L_body = orientation.T @ L_space
#        
#        # w = I^-1 * L
#        w_body = Iinv @ L_body
#        w_norm = np.linalg.norm(w_body)
#
#        # Inkrementalna rotacija za delić vremena dt
#        if w_norm > 1e-15:
#            axis = w_body / w_norm
#            angle = w_norm * dt
#            K = np.array([[0, -axis[2], axis[1]],
#                          [axis[2], 0, -axis[0]],
#                          [-axis[1], axis[0], 0]])
#            deltaR = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
#            
#            # Ažuriranje orijentacije: Nova = Inkrement * Stara
#            orientation = deltaR @ orientation
#
#        rotations.append(orientation.copy())
#
#    return np.stack(rotations)








#def compute_tumbling_matrices_any(duration, dt, T_spin, T_prec, tilt, cone):
#    """
#    Računa matrice rotacije za proizvoljne periode bez racionalne aproksimacije.
#    
#    duration: Ukupno vreme simulacije (npr. u satima)
#    dt: Vremenski korak (npr. 0.1 sat)
#    T_spin: Period rotacije oko sopstvene ose (sati)
#    T_prec: Period precesije ose (sati)
#    tilt: Ugao nagnutosti (stepeni)
#    cone: Početni položaj sistema u prostoru (stepeni)
#    """
#
#    # --- Prebacivanje perioda u ugaone brzine ---
#    spinrate = 2 * np.pi / T_spin
#    omega_prec = 2 * np.pi / T_prec
#
#    # --- Definisanje fiktivne fizike koja podržava ove periode ---
#    # Koristimo stabilan model simetričnog rotora (I1 = I2)
#    # Iz fizike precesije: omega_prec = spinrate * (Ilong/Ishort - 1) * cos(tilt)
#    Ilong = 1.0
#    Imiddle = 1.0
#    # Izvodimo Ishort tako da numerička integracija proizvede tvoj T_prec
#    cos_tilt = np.cos(np.radians(tilt))
#    if cos_tilt == 0: cos_tilt = 1e-9 # Izbegavanje deljenja nulom kod 90 stepeni
#    
#    Ishort = Ilong / (1.0 + (omega_prec / (spinrate * cos_tilt)))
#    
#    Ibody = np.diag([Ilong, Imiddle, Ishort])
#    Iinv = np.linalg.inv(Ibody)
#
#    # --- Početni uslovi ---
#    theta_tilt = np.radians(tilt)
#    L_dir = np.array([np.sin(theta_tilt), 0.0, np.cos(theta_tilt)])
#    Lmag = Ishort * spinrate 
#    L_body0 = Lmag * L_dir
#
#    # --- Početna orijentacija sa CONE nagibom ---
#    orientation = np.eye(3)
#    theta_cone = np.radians(cone)
#    # Rodrigues oko space Y ose
#    Ky = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
#    R_cone = np.eye(3) + np.sin(theta_cone)*Ky + (1 - np.cos(theta_cone))*(Ky @ Ky)
#    orientation = R_cone @ orientation
#
#    # L je konstantan u prostoru (Inertial Frame)
#    L_space = orientation @ L_body0
#    
#    total_steps = int(duration / dt)
#    rotations = []
#
#    # --- Integraciona petlja ---
#    for step in range(total_steps):
#        # 1. Projektuj L u body frame
#        L_body = orientation.T @ L_space
#        
#        # 2. Izračunaj trenutnu ugaonu brzinu w
#        w_body = Iinv @ L_body
#        w_norm = np.linalg.norm(w_body)
#
#        # 3. Rodriguesova inkrementalna rotacija
#        if w_norm > 1e-15:
#            axis = w_body / w_norm
#            angle = w_norm * dt
#            K = np.array([[0, -axis[2], axis[1]],
#                          [axis[2], 0, -axis[0]],
#                          [-axis[1], axis[0], 0]])
#            deltaR = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
#            
#            # 4. Ažuriraj orijentaciju
#            orientation = deltaR @ orientation
#
#        rotations.append(orientation.copy())
#
#    return np.stack(rotations)




# =============================================================================
# if __name__ == "__main__":
#     # Parameters
# #    shape_file = "../../data/shape_models/67P_not_to_scale_low_res.stl"
# #    shape_file = "../../data/shape_models/500m_ico_sphere_1280_facets.stl"
# #    shape_file = "../../data/shape_models/Apophis.stl"
#     shape_file = "../../data/shape_models/Rubber_Duck_1500_facets.stl"
#     
#     
#     
#     
#     fps = 30
#     timesteps = 500
#     lat = 90
#     long = 0
#     theta_deg = 30
#     P_spin = 0.025
#     P_prec = 1
#     n_cycles = 0.5
#     print("Computing rotation matrices...")
#         
#     
#     rotations = compute_tumbling_kinematics(
#             timesteps, 
#             long, 
#             lat, 
#             theta_deg, 
#             P_spin, 
#             P_prec, n_cycles = n_cycles)
#     
#     output_file = "tumbling_animation.gif"
#     animate_rotation(
#         shape_file=shape_file,
#         rotation_matrices=rotations,
#         output_file=output_file,
#         fps=fps,
#         skip_frames=1  # Skip frames to reduce file size
#     )
# 
#     # Print length of rotations
#     print(f"Length of rotations: {len(rotations)}")
#     
#     print(f"Animation saved to {output_file}")
# =============================================================================



if __name__ == "__main__":
    # Putanja do tvog modela
#    shape_file = "../../data/shape_models/Apophis.stl"
    shape_file = "../../data/shape_models/Rubber_Duck_1500_facets.stl"
    
    # Parametri simulacije
    fps = 30
    timesteps = 500
    lat = 60
    long = -90
    theta_deg = 32.8
    phi_0_deg = 90
    G_body = [1, 0, 0]
    P_spin = 0.05
    P_prec = 1
    n_cycles = 1
    
    print("Computing kinematics (rotations, axes, and hodograph)...")
    
    # Pozivamo funkciju koja sada vraća rečnik sa svim podacima
    sim_data = compute_tumbling_kinematics_3(
        timesteps=timesteps, 
        long_deg=long, 
        lat_deg=lat, 
        theta_deg=theta_deg,
        phi_0_deg = phi_0_deg,
        G_body = G_body,
        P_spin=P_spin, 
        P_prec=P_prec, 
        n_cycles=n_cycles
    )
    
    # Izdvajamo ono što nam treba za animaciju
    rotations = sim_data['rotations']
    L_hat = sim_data['L_axis']
    omega_vecs = sim_data['omega_axes']
    
    print(f"Computed {len(rotations)} frames.")
    
    output_file = "tumbling_animation.gif"
    
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
    