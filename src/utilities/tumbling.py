import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. PARAMETRI IZ TABELE 2 (Apofis) ---
I1, I2, I3 = 0.61, 0.965, 1.0
P_phi_h = 27.38
w_phi = (2 * np.pi) / (P_phi_h * 3600)

# Početni uslovi (SAM režim - Short Axis Mode)
theta_start = np.deg2rad(37.0)
psi_start = np.deg2rad(14.0)
phi_start = np.deg2rad(152.0)





# =============================================================================
# # --- 1. PARAMETRI  ---
# I1, I2, I3 = 0.8, 0.8, 1.0
# P_phi_h = 24
# w_phi = (2 * np.pi) / (P_phi_h * 3600)
# 
# # Početni uslovi (SAM režim - Short Axis Mode)
# theta_start = np.deg2rad(90)
# psi_start = np.deg2rad(60)
# phi_start = np.deg2rad(80)
# =============================================================================

# Vektori za praćenje
V_INERTIAL_FIXED = np.array([1.0, 0.0, 0.0]) # Npr. pravac ka Suncu
BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0]) # Najkraća osa (I3)

# Izračunavanje momenta L i početnih w komponenti
L_fixed = w_phi * ((I1 + I2) / 2) / np.cos(theta_start)
w1_0 = (L_fixed * np.sin(theta_start) * np.sin(psi_start)) / I1
w2_0 = (L_fixed * np.sin(theta_start) * np.cos(psi_start)) / I2
w3_0 = (L_fixed * np.cos(theta_start)) / I3

y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start]
t_limit = 10 * 24 * 3600 

# --- 2. DINAMIKA ---
def dynamics(t, y):
    w1, w2, w3, phi, theta, psi = y
    st = np.sin(theta) if abs(np.sin(theta)) > 1e-9 else 1e-9
    dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
    dtheta = w1 * np.cos(psi) - w2 * np.sin(psi)
    dpsi = w3 - dphi * np.cos(theta)
    return [((I2-I3)*w2*w3)/I1, ((I3-I1)*w3*w1)/I2, ((I1-I2)*w1*w2)/I3, 
            dphi, dtheta, dpsi]

dt = 60 
t_eval = np.arange(0, t_limit + dt, dt)

# 2. Poziv integratora sa t_eval parametrom
sol = solve_ivp(
    dynamics, 
    (0, t_limit), 
    y0, 
    method='DOP853', 
    t_eval=t_eval,    # OVO JE KLJUČNA IZMENA
    rtol=1e-11, 
    atol=1e-13
)

# 3. Rezultati su sada garantovano u t_eval tačkama
vreme = sol.t          # Biće [0, 60, 120, 180...]
w1, w2, w3 = sol.y[0], sol.y[1], sol.y[2]
phi, theta, psi = sol.y[3], sol.y[4], sol.y[5]

v_body_coords = []     # Fiksni inercijalni vektor u sistemu asteroida
spin_axis_iner = []    # Osa rotacije (omega) na nebu
body_axis_iner = []    # Fizička osa asteroida (I3) na nebu

for i in range(len(phi)):
    c1, s1 = np.cos(phi[i]), np.sin(phi[i])
    c2, s2 = np.cos(theta[i]), np.sin(theta[i])
    c3, s3 = np.cos(psi[i]), np.sin(psi[i])
    
    # Matrica rotacije R (Inercijalni -> Telo) 3-1-3
    R = np.array([
        [ c3*c1 - s3*c2*s1,  c3*s1 + s3*c2*c1, s3*s2],
        [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
        [ s2*s1,            -s2*c1,            c2]
    ])
    
    # 1. Inercijalni vektor ka zvezdi/Suncu u telu asteroida
    v_body_coords.append(R @ V_INERTIAL_FIXED)
    
    # 2. Osa rotacije (omega) na nebu
    w_body = np.array([w1[i], w2[i], w3[i]])
    spin_axis_iner.append(R.T @ w_body)
    
    # 3. Fizička osa asteroida (I3) na nebu
    body_axis_iner.append(R.T @ BODY_AXIS_TO_TRACK)

v_body_coords = np.array(v_body_coords)
spin_axis_iner = np.array(spin_axis_iner)
body_axis_iner = np.array(body_axis_iner)

# Pomoćna funkcija za sferne koordinate
def to_spherical(vecs):
    mag = np.linalg.norm(vecs, axis=1)
    unit = vecs / mag[:, np.newaxis]
    delta = np.arcsin(unit[:, 2])
    alpha = np.arctan2(unit[:, 1], unit[:, 0])
    return alpha, delta

alpha_spin, delta_spin = to_spherical(spin_axis_iner)
alpha_axis, delta_axis = to_spherical(body_axis_iner)

# --- 4. VIZUELIZACIJA ---
fig = plt.figure(figsize=(16, 12))

# 1. Nutacija (Theta)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(sol.t/3600, np.rad2deg(theta), color='blue')
ax1.axhline(12, color='r', ls='--', alpha=0.5, label='Min (Table 2)')
ax1.axhline(55, color='g', ls='--', alpha=0.5, label='Max (Table 2)')
ax1.set_title('Ugao nutacije (Theta)')
ax1.set_ylabel('Ugao [°]')
ax1.legend()
ax1.grid(True)

# 2. Inercijalni vektor [1,0,0] u sistemu tela (šta vidi "čovečuljak" na asteroidu)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(sol.t/3600, v_body_coords[:, 0], label='Body X')
ax2.plot(sol.t/3600, v_body_coords[:, 1], label='Body Y')
ax2.plot(sol.t/3600, v_body_coords[:, 2], label='Body Z')
ax2.set_title('Inercijalni vektor [1,0,0] u koordinatama asteroida')
ax2.set_ylabel('Projekcija')
ax2.legend()
ax2.grid(True)

# 3. Održanje E i L
E = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
L_mag = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
ax3 = plt.subplot(2, 2, 3)
ax3.plot(sol.t/3600, (E - E[0])/E[0], label='Delta E / E0')
ax3.plot(sol.t/3600, (L_mag - L_mag[0])/L_mag[0], label='Delta L / L0')
ax3.set_title('Numerička stabilnost (Relativna greška)')
ax3.set_ylabel('Greška')
ax3.legend()
ax3.grid(True)

# 4. Nebeska sfera: Putanja Ose Rotacije vs Putanja Ose Tela (I3)
ax4 = plt.subplot(2, 2, 4)
ax4.plot(np.rad2deg(alpha_spin), np.rad2deg(delta_spin), 'r-', label='Osa rotacije (omega)', alpha=0.6)
ax4.plot(np.rad2deg(alpha_axis), np.rad2deg(delta_axis), 'b-', label='Osa tela (I3)', lw=0.8)
ax4.set_title('Kretanje na nebeskoj sferi (Inercijalno)')
ax4.set_xlabel('Rektascenzija [°]')
ax4.set_ylabel('Deklinacija [°]')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()