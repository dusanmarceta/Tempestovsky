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
        r_body_coords.append(Ri.T @ r_inertial)
        spin_axis_iner.append(Ri @ w_body)
        body_axis_iner.append(Ri @ BODY_AXIS_TO_TRACK) # <-- BEZ .T !

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
    

# 1. GEOMETRIJA OBJEKTA
I1, I2, I3 = 1.0, 1.0, 2.0

# 2. VEKTOR UGAONOG MOMENTA L U INERCIJALNOM PROSTORU
L_inertial_target = np.array([1.0, 0.0, 2.0])
L_magnitude = np.linalg.norm(L_inertial_target)
L_hat_target = L_inertial_target / L_magnitude

# 3. NAGIB I SPIN
theta_deg = 0
theta_0 = np.deg2rad(theta_deg)

# 4. BRZINE
L1_0 = L_magnitude * np.sin(theta_0) * np.sin(0)
L2_0 = L_magnitude * np.sin(theta_0) * np.cos(0)
L3_0 = L_magnitude * np.cos(theta_0)

w1_0 = L1_0 / I1
w2_0 = L2_0 / I2
w3_0 = L3_0 / I3

# 5. EULEROVI UGLOVI ZA ČISTU ROTACIJU OKO Y-OSE
L_x, L_y, L_z = L_hat_target
phi_L = np.arctan2(L_y, L_x)
theta_L = np.arccos(np.clip(L_z, -1.0, 1.0))

phi_0 = phi_L + np.pi / 2.0   # Postavlja Z-X-Z ravan nagiba duž inercijalne Y-ose
theta_0_euler = theta_L       # Nagib od 45 stepeni
psi_0 = np.deg2rad(-90)                   # Zadržava x-osu tela u XZ ravni!

y0 = [w1_0, w2_0, w3_0, phi_0, theta_0_euler, psi_0]

# 4. SIMULACIJA
sim = compute_tumbling_dynamics(
    dt=0.1, 
    timesteps=800, 
    y0=y0, 
    I=np.array([I1, I2, I3]), 
    r_inertial=np.array([1.0, 0.0, 0.0]), 
    BODY_AXIS_TO_TRACK=np.array([1.0, 0.0, 0.0])  # Pratimo osu simetrije tela (z-osu tela)
)

# Vreme u sekundama ili satima radi grafika
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

# # 5. PRIKAZ REZULTATA
# plt.figure(figsize=(10, 4))
# plt.plot(time, long)
# plt.xlabel("Vreme [s]")
# plt.ylabel("Longituda [deg]")
# plt.title("Precesija ose simetrije oko inercijalne Z-ose")
# plt.grid(True)
# plt.show()




# plt.figure(figsize=(6, 6))
# plt.plot(x_body, y_body)
# plt.xlabel("X inercijalno")
# plt.ylabel("Y inercijalno")
# plt.title("Putanja z-ose tela u XY ravni (Kružnica)")
# plt.axis("equal")  # Ključno da kružnica ne izgleda kao elipsa
# plt.grid(True)
# plt.show()

plt.figure(figsize=(6, 6))
plt.plot(long_inertial, lat_inertial)
plt.plot(long_inertial[0], lat_inertial[0], 'o')
plt.plot(long_inertial[100], lat_inertial[100], 's')
plt.title('Inertial')
plt.grid()

plt.figure(figsize=(6, 6))
plt.plot(long, lat, 'r')
plt.plot(long[0], lat[0], 'o')
plt.plot(long[100], lat[100], 's')
plt.title('Body')
plt.grid()


