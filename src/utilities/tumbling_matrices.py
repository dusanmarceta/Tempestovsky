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



def compute_tumbling_kinematics(timesteps, long_deg, lat_deg, theta_deg, phi_0_deg, G_body, P_spin, P_prec, n_cycles=1):
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





def compute_tumbling_dynamics(t_limit, y0, I, V_INERTIAL_FIXED, BODY_AXIS_TO_TRACK, timesteps=1000):
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
    t_eval = np.linspace(0, t_limit, timesteps)
    sol = solve_ivp(dynamics, (0, t_limit), y0, t_eval=t_eval, 
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
    v_body_coords = []
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
        v_body_coords.append(Ri @ V_INERTIAL_FIXED)
        spin_axis_iner.append(Ri.T @ w_body)
        body_axis_iner.append(Ri.T @ BODY_AXIS_TO_TRACK)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,  # Vraća jedinični vektor ugaonog momenta
        'v_body_coords': np.stack(v_body_coords),
        'G_axes': np.stack(spin_axis_iner),
        'omega_axes': np.stack(body_axis_iner),
        'time': sol.t,
        'omega_body': sol.y[0:3, :].T
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



if __name__ == "__main__":
    # Putanja do tvog modela
#    shape_file = "../../data/shape_models/Apophis.stl"
    shape_file = "../../data/shape_models/Rubber_Duck_1500_facets.stl"
    
  
    
    
        # --- 1. PARAMETRI IZ TABELE 2 (Apofis) ---
    I1, I2, I3 = 0.61, 0.965, 1.0
    P_phi_h = 27.38
    w_phi = (2 * np.pi) / (P_phi_h * 3600)

    # Početni uslovi (SAM režim - Short Axis Mode)
    theta_start = np.deg2rad(37.0)
    psi_start = np.deg2rad(14.0)
    phi_start = np.deg2rad(152.0)
    
    # Vektori za praćenje
    V_INERTIAL_FIXED = np.array([1.0, 0.0, 0.0]) # Npr. pravac ka Suncu
    BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0]) # Najkraća osa (I3)

    # Izračunavanje momenta L i početnih w komponenti
    L_fixed = w_phi * ((I1 + I2) / 2) / np.cos(theta_start)
    w1_0 = (L_fixed * np.sin(theta_start) * np.sin(psi_start)) / I1
    w2_0 = (L_fixed * np.sin(theta_start) * np.cos(psi_start)) / I2
    w3_0 = (L_fixed * np.cos(theta_start)) / I3

    y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start]
    t_limit = 50 * 24 * 3600 # 3 dana

    
    # Parametri simulacije
    fps = 30
    timesteps = 10000
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
# =============================================================================
#     sim_data = compute_tumbling_kinematics(
#         timesteps=timesteps, 
#         long_deg=long, 
#         lat_deg=lat, 
#         theta_deg=theta_deg,
#         phi_0_deg = phi_0_deg,
#         G_body = G_body,
#         P_spin=P_spin, 
#         P_prec=P_prec, 
#         n_cycles=n_cycles
#     )
# =============================================================================
    
    
    sim_data = compute_tumbling_dynamics(t_limit, y0, np.array([I1, I2, I3]), V_INERTIAL_FIXED, BODY_AXIS_TO_TRACK, timesteps=timesteps)
    
    
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
    