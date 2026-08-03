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
        
        rotations.append(Ri.T) # Telo -> Inercijalni
        r_body_coords.append(Ri @ r_inertial)
        spin_axis_iner.append(Ri.T @ w_body)
        body_axis_iner.append(Ri.T @ BODY_AXIS_TO_TRACK)

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
# # --- PARAMETRI IZ TABELE 2 (Apofis) ---
# I1 = 0.61
# I2 = 0.965
# I3 = 1.0
# 
# 
# P_phi_h = 27.38   # Period precesije u satima
# P_psi_h = 263.0   # Period spina u satima
# 
# # Pretvaranje u rad/s
# w_phi = (2 * np.pi) / (P_phi_h * 3600)
# w_psi = -(2 * np.pi) / (P_psi_h * 3600)  # Retrogradno
# 
# # Početni uglovi
# phi_start = np.deg2rad(152.0)
# psi_start = np.deg2rad(14.0)
# theta_start = np.deg2rad(54.0)
# 
# # Kinematika 3-1-3 za komponente w
# dphi_0 = w_phi
# dpsi_0 = w_psi
# dtheta_0 = 0.0
# 
# w1_0 = dphi_0 * np.sin(theta_start) * np.sin(psi_start) + dtheta_0 * np.cos(psi_start)
# w2_0 = dphi_0 * np.sin(theta_start) * np.cos(psi_start) - dtheta_0 * np.sin(psi_start)
# w3_0 = dpsi_0 + dphi_0 * np.cos(theta_start)
# 
# y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start]
# 
# # Vektori
# V_INERTIAL_FIXED = np.array([1.0, 0.0, 0.0])
# BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0])
# 
# # --- DISKRETIZACIJA ZA SIMULACIJU ---
# # Integrišemo tokom npr. 120 sati (oko 5 dana) da se uoče ciklusi precesije
# t_total_seconds = 1200 * 3600  
# dt = 600  # Korak od 10 minuta (600 sekundi) za finu rezoluciju
# timesteps = int(t_total_seconds / dt)
# 
# # --- POZIV TVOJE FUNKCIJE ---
# sim_data_1 = compute_tumbling_dynamics(
#     dt=dt, 
#     timesteps=timesteps, 
#     y0=y0, 
#     I=np.array([I1, I2, I3]), 
#     r_inertial=V_INERTIAL_FIXED, 
#     BODY_AXIS_TO_TRACK=BODY_AXIS_TO_TRACK
# )
# =============================================================================

# ------------------------------------------------------------------------------
# 1. GEOMETRIJA OBJEKTA: Aksijalno simetrično telo (I1 = I2)
# ------------------------------------------------------------------------------
# 1. GEOMETRIJA OBJEKTA (Aksijalna simetrija: I1 = I2)
I1 = 1.0
I2 = 1.0
I3 = 2.0

# 2. ŽELJENI NAGIB (ugao nutacije theta) i INTENZITET L
theta_deg = 30.0  # nagib u stepenima
theta_0 = np.deg2rad(theta_deg)

L_magnitude = 2.0  # proizvoljni intenzitet ugaonog momenta
psi_0 = 0.0        # početna faza spina
phi_0 = 0.0        # precesijski ugao

# 3. PRORAČUN POČETNIH UGAONIH BRZINA
# Iz L_body = R * L_iner dobijamo komponente u sistemu tela:
L1_0 = L_magnitude * np.sin(theta_0) * np.sin(psi_0)
L2_0 = L_magnitude * np.sin(theta_0) * np.cos(psi_0)
L3_0 = L_magnitude * np.cos(theta_0)

w1_0 = L1_0 / I1
w2_0 = L2_0 / I2
w3_0 = L3_0 / I3

# Sklapanje početnog vektora y0
y0 = [w1_0, w2_0, w3_0, phi_0, theta_0, psi_0]

# 4. SIMULACIJA
sim = compute_tumbling_dynamics(
    dt=0.01, 
    timesteps=2000, 
    y0=y0, 
    I=np.array([I1, I2, I3]), 
    r_inertial=np.array([1.0, 0.0, 0.0]), 
    BODY_AXIS_TO_TRACK=np.array([0.0, 1.0, 0.0])
)

# Vreme pretvoreno u sate radi preglednosti na grafiku
time_hours = sim['time'] / 3600.0
body_axis = sim['body_axis']

x_body = np.transpose(body_axis)[0]
y_body = np.transpose(body_axis)[1]
z_body = np.transpose(body_axis)[2]



long = np.arctan2(y_body, x_body)
lat = np.arcsin(z_body)



plt.plot(time_hours*3600, np.rad2deg(long))
plt.grid()

# # Podaci za plot
# spin_x_iner = sim_data_1['G_axes'][:, 0]  # X komponenta spina u prostoru
# w1 = sim_data_1['omega_body'][:, 0]       # w1 u sistemu tela
# w1_sq = w1**2                            # w1^2 u sistemu tela

# # --- ISCRTAVANJE GRAFIKA ---
# fig, axs = plt.subplots(2, 1, figsize=(10, 7))

# # 1. Gornji plot: Inercijalna precesija
# axs[0].plot(time_hours, spin_x_iner, color='tab:blue', lw=1.5, label=r'Spin $G_x$ (Inercijalno)')
# # axs[0].set_title(f'Apofis (99942): Precesija u inercijalnom prostoru ($P_\phi \approx {P_phi_h}\ h$)')
# axs[0].set_xlabel('Vreme [h]')
# axs[0].set_ylabel('Ugaona brzina [rad/s]')
# axs[0].grid(True, linestyle='--', alpha=0.7)
# axs[0].legend(loc='upper right')

# # 2. Donji plot: w1 vs w1^2 u sistemu tela (normirano na [0, 1])
# w1_norm = w1 / np.max(np.abs(w1))
# w1_sq_norm = w1_sq / np.max(np.abs(w1_sq))

# axs[1].plot(time_hours, w1_norm, color='tab:orange', lw=1.5, label=r'$\omega_1(t) / \max(\omega_1)$')
# axs[1].plot(time_hours, w1_sq_norm, color='tab:green', lw=1.5, linestyle='--', label=r'$\omega_1^2(t) / \max(\omega_1^2)$')
# # axs[1].set_title(r'Apofis (99942): Dinamika $\omega_1$ i $\omega_1^2$ u sistemu tela')
# axs[1].set_xlabel('Vreme [h]')
# axs[1].set_ylabel('Normirana vrednost [-]')
# axs[1].set_ylim(-0.05, 1.05)  # fiksiran opseg od 0 do 1 sa malom marginom
# axs[1].grid(True, linestyle='--', alpha=0.7)
# axs[1].legend(loc='upper right')

# plt.tight_layout()
# plt.show()

# # --- PARAMETRI DIREKTNO IZ TVOJE TABELE 2 ---
# I1 = 0.61
# I2 = 0.965
# I3 = 1.0

# P_phi_h = 27.38   # Izmereni period precesije
# P_psi_h = 263.0   # Izmereni period spina

# # Pretvaranje u ugaonu brzinu (rad/s)
# w_phi = (2 * np.pi) / (P_phi_h * 3600)
# w_psi = -(2 * np.pi) / (P_psi_h * 3600)  # minus označava retrogradno kretanje spina za Apofis

# # Početni uglovi za epohu iz rada (Tabela 2)
# phi_start = np.deg2rad(152.0)
# psi_start = np.deg2rad(14.0)
# # Početni nagib (možeš staviti bilo koji između 12 i 55, rad koristi specifičnu vrednost u epohi)
# theta_start = np.deg2rad(54.0) 

# # --- INVERZNA KINEMATIKA KOJA SPRAJA OBA PERIODA ---
# dphi_0 = w_phi
# dpsi_0 = w_psi
# dtheta_0 = 0.0  # Pretpostavka za stacionarnu tačku u epohi

# # Izračunavanje pravih komponenti w na osnovu kinematike 3-1-3
# w1_0 = dphi_0 * np.sin(theta_start) * np.sin(psi_start) + dtheta_0 * np.cos(psi_start)
# w2_0 = dphi_0 * np.sin(theta_start) * np.cos(psi_start) - dtheta_0 * np.sin(psi_start)
# w3_0 = dpsi_0 + dphi_0 * np.cos(theta_start)



# # # Vektori za praćenje
# V_INERTIAL_FIXED = np.array([1.0, 0.0, 0.0]) # Npr. pravac ka Suncu
# BODY_AXIS_TO_TRACK = np.array([0.0, 0.0, 1.0]) # Najkraća osa (I3)


# # Ovo ide u tvoj solver (ovo treba proveriti!!!)
# y0 = [w1_0, w2_0, w3_0, phi_start, theta_start, psi_start]


# t_limit = 1* P_phi_h * 3600 


# dt = 10000
# timesteps = 20
# t_eval1 = np.arange(timesteps) * dt

# # sim_data = compute_tumbling_dynamics(t_limit, y0, np.array([I1, I2, I3]), V_INERTIAL_FIXED, BODY_AXIS_TO_TRACK, timesteps=timesteps)


# # --- PRVA SIMULACIJA (Prva 3 dana) ---
# sim_data_1 = compute_tumbling_dynamics(
#     dt, timesteps, y0, np.array([I1, I2, I3]), V_INERTIAL_FIXED, BODY_AXIS_TO_TRACK
# )


# # --- ANALIZA REZULTATA IZ SIMULACIJE (sim_data_1) ---

# # Pokupimo vreme i ugaone brzine iz simulacije
# vreme_sve = sim_data_1['time'] # u sekundama
# omega_body = sim_data_1['omega_body'] # w1, w2, w3 kroz vreme

# # 1. PERIOD PRECESIJE (P_phi) - Direktno iz kinematike na početku
# st = np.sin(theta_start) if abs(np.sin(theta_start)) > 1e-9 else 1e-9
# dphi_0 = (w1_0 * np.sin(psi_start) + w2_0 * np.cos(psi_start)) / st
# T_precession_h = (2 * np.pi) / (dphi_0 * 3600)

# # 2. PERIOD DINAMIKE UNUTAR TELA (T_omega / Nutacija)
# # Računamo frekvenciju rotacije vektora w unutar tela (preko momenata inercije)
# I_srednje = (I1 + I2) / 2
# omega_body_precession = abs(w3_0 * (I3 - I_srednje) / I_srednje)
# T_body_interaction_h = (2 * np.pi) / (omega_body_precession * 3600)

# # 3. PRAVI INERCIJALNI PERIOD SPINA (P_psi)
# # U fizici asteroida, inercijalni spin (onaj od 263h) je kombinacija relativnog spina i precesije.
# # Za SAM režim simetričnog/skoro-simetričnog topa važi fundamentalna relacija:
# dpsi_0 = w3_0 - dphi_0 * np.cos(theta_start) # Relativna brzina ugla psi
# omega_spin_inertial = abs(dpsi_0) # Brzina spina

# T_spin_inertial_h = (2 * np.pi) / (omega_spin_inertial * 3600)

# # --- PRINTOVANJE REZULTATA ---
# print("=" * 60)
# print("       STVARNI ROTACIONI PERIODI IZ DINAMIKE TELA")
# print("=" * 60)
# print(f"Momenti inercije koji se koriste:  I1={I1}, I2={I2}, I3={I3}")
# print("-" * 60)
# print(f"1. Period PRECESIJE (P_phi):        {T_precession_h:.2f} sati")
# print(f"2. Period INERCIJALNOG SPINA:       {T_spin_inertial_h:.2f} sati")
# print(f"3. Period NUTACIJE (u telu):        {T_body_interaction_h:.2f} sati")
# print("-" * 60)
# print(f"Odnos glavnih perioda (Precesija/Spin): {T_precession_h / T_spin_inertial_h:.4f}")
# print("=" * 60)
    




# '''

# NOVA PROVERA

# '''


# from scipy.signal import find_peaks

# # Pomoćna funkcija za detekciju perioda iz signala
# def estimate_period(time, signal):
#     sig_detrend = signal - np.mean(signal)
#     dt = time[1] - time[0]
#     peaks, _ = find_peaks(sig_detrend, distance=int(0.05 / dt))
#     if len(peaks) < 2:
#         return None
#     return np.mean(np.diff(time[peaks]))


# print("=== PROVERA ODNOSA PERIODA ===")

# # ------------------------------------------------------------------------------
# # TEST 1: Fajnmanov tanjir (I1 = I2 = 1, I3 = 2)
# # U prostoru: Period precesije ose rotacije iznosi tačno 0.5 * T_spin
# # ------------------------------------------------------------------------------
# dt = 0.001
# timesteps = 20000

# I_plate = [1.0, 1.0, 3.0]
# w3_0 = 2 * np.pi  # 1 Hz spina
# y0_plate = [0.1, 0.0, w3_0, 0.0, 0.1, 0.0] # blag nagib (theta0 = 0.1 rad)

# # Poziv tvoje funkcije
# res1 = compute_tumbling_dynamics(
#     dt=dt, 
#     timesteps=timesteps, 
#     y0=y0_plate, 
#     I=I_plate, 
#     r_inertial=np.array([1.0, 0.0, 0.0]), 
#     BODY_AXIS_TO_TRACK=np.array([0.0, 0.0, 1.0])
# )

# T_spin_theory = 2 * np.pi / w3_0
# # G_axes je vektor ugaone brzine preslikan u inercijalni prostor
# T_prec_sim = estimate_period(res1['time'], res1['G_axes'][:, 0])

# ratio1 = T_prec_sim / T_spin_theory
# print(f"\n[Test 1: Fajnmanov tanjir]")
# print(f"-> Teorijski T_spin:       {T_spin_theory:.5f} s")
# print(f"-> Simulirani T_precesija: {T_prec_sim:.5f} s")
# print(f"-> Odnos T_prec / T_spin:  {ratio1:.5f} (Očekivano: 0.50000)")

# if np.isclose(ratio1, 0.5, atol=1e-2):
#     print("STATUS: ✅ Prošlo!")
# else:
#     print("STATUS: ❌ Nije prošlo.")


# # ------------------------------------------------------------------------------
# # TEST 2: Nutacioni ciklus u sistemu tela (I1 = 1.0, I2 = 1.2, I3 = 2.0)
# # U telu: Kvadrat ugaone brzine w1^2 osciluje duplo brže od w1 (T_w1_sq = 0.5 * T_w1)
# # ------------------------------------------------------------------------------
# I_tri = [1.0, 1.2, 2.0]
# y0_tri = [0.5, 0.2, 3.0, 0.0, 0.3, 0.0]

# # Poziv tvoje funkcije
# res2 = compute_tumbling_dynamics(
#     dt=dt, 
#     timesteps=timesteps, 
#     y0=y0_tri, 
#     I=I_tri, 
#     r_inertial=np.array([1.0, 0.0, 0.0]), 
#     BODY_AXIS_TO_TRACK=np.array([0.0, 0.0, 1.0])
# )

# w1 = res2['omega_body'][:, 0]
# w1_sq = w1**2

# T_w1 = estimate_period(res2['time'], w1)
# T_w1_sq = estimate_period(res2['time'], w1_sq)

# ratio2 = T_w1_sq / T_w1
# print(f"\n[Test 2: Polhode u sistemu tela]")
# print(f"-> Period za w1:          {T_w1:.5f} s")
# print(f"-> Period za w1^2 (nut):  {T_w1_sq:.5f} s")
# print(f"-> Odnos T(w1^2) / T(w1): {ratio2:.5f} (Očekivano: 0.50000)")

# if np.isclose(ratio2, 0.5, atol=1e-2):
#     print("STATUS: ✅ Prošlo!")
# else:
#     print("STATUS: ❌ Nije prošlo.")


# # ------------------------------------------------------------------------------
# # GRAFIČKI PRIKAZ
# # ------------------------------------------------------------------------------
# fig, axs = plt.subplots(2, 1, figsize=(9, 5))

# # Test 1: Inercijalna precesija
# axs[0].plot(res1['time'][:3000], res1['G_axes'][:3000, 0], color='tab:blue')
# axs[0].set_title('Test 1: Inercijalna precesija ose spina (X komponenta)')
# axs[0].grid(True)

# # Test 2: w1 vs w1^2
# axs[1].plot(res2['time'][:3000], w1[:3000], label=r'$\omega_1$', color='tab:orange')
# axs[1].plot(res2['time'][:3000], w1_sq[:3000], label=r'$\omega_1^2$', color='tab:green', linestyle='--')
# axs[1].set_title(r'Test 2: Poređenje perioda $\omega_1$ i $\omega_1^2$ u sistemu tela')
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.show()
    
   