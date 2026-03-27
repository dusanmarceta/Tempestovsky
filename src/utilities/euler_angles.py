import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def apophis_rotation_dynamics(t_span, y0, I1, I2, I3):
    """
    Numericka integracija rotacije krutog tela (Apofis).
    
    Parametri:
    t_span: (t_start, t_end) - vremenski interval u sekundama
    y0: lista [w1, w2, w3, phi, theta, psi] - pocetni uslovi
    I1, I2, I3: Glavni momenti inercije (kg m^2)
    """
    
    def derivatives(t, y):
        w1, w2, w3, phi, theta, psi = y
        
        # 1. Ojlerove dinamicke jednacine (Torque-free)
        dw1 = ((I2 - I3) * w2 * w3) / I1
        dw2 = ((I3 - I1) * w3 * w1) / I2
        dw3 = ((I1 - I2) * w1 * w2) / I3
        
        # 2. Kinematicke jednacine (3-1-3 sekvenca: Z-X-Z)
        # Dodajemo mali epsilon da izbegnemo deljenje nulom (Gimbal Lock)
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-9:
            sin_theta = 1e-9
            
        dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / sin_theta
        dtheta = w1 * np.cos(psi) - w2 * np.sin(psi)
        dpsi = w3 - dphi * np.cos(theta)
        
        return [dw1, dw2, dw3, dphi, dtheta, dpsi]

    # Resavanje sistema diferencijalnih jednacina
    # Koristimo 'RK45' (Runge-Kutta 4/5 reda)
    solution = solve_ivp(derivatives, t_span, y0, method='RK45', rtol=1e-9, atol=1e-12)
    
    return solution

# --- Primer koriscenja ---
# Momenti inercije (proporcionalni, npr. za izduzeno telo)
    
I1 = 0.61
I2 = 0.965

I = [I1, I2, 1] # I1, I2, I3

P1 = 946800
P2 = 98568
P3 = P1/2

phi_0 = np.deg2rad(152)
psi_0 = np.deg2rad(14)
theta_0 = np.deg2rad(15)

# Pocetne ugaone brzine (rad/s) i uglovi (rad)
# Pretpostavimo lagano tumbanje
w0 = [2*np.pi/P1, 2*np.pi/P2, 2*np.pi/P3] 
angles0 = [phi_0, psi_0, theta_0] # phi, theta, psi
initial_state = w0 + angles0

t_limit = (0, 3600 * 24) # Simulacija za 24 sata

sol = apophis_rotation_dynamics(t_limit, initial_state, *I)

# Rezultati:
# sol.t -> vremenske tacke
# sol.y[0:3] -> komponente ugaone brzine (w)
# sol.y[3:6] -> Ojlerovi uglovi (phi, theta, psi)

# --- Vizuelizacija ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. Grafik ugaonih brzina (Body frame)
ax1.plot(sol.t / 3600, sol.y[0], label=r'$\omega_1$ (Short axis)')
ax1.plot(sol.t / 3600, sol.y[1], label=r'$\omega_2$ (Intermediate axis)')
ax1.plot(sol.t / 3600, sol.y[2], label=r'$\omega_3$ (Long axis)')
ax1.set_ylabel('Ugaona brzina [rad/s]')
ax1.set_title('Dinamika rotacije Apofisa (Tumbling)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Grafik Ojlerovih uglova (Inertial frame)
# Konvertujemo radijane u stepene za lakšu interpretaciju
ax2.plot(sol.t / 3600, np.degrees(sol.y[3]) % 360, label=r'$\phi$ (Precesija)')
ax2.plot(sol.t / 3600, np.degrees(sol.y[4]) % 360, label=r'$\theta$ (Nutacija)')
ax2.plot(sol.t / 3600, np.degrees(sol.y[5]) % 360, label=r'$\psi$ (Sopstvena rotacija)')
ax2.set_xlabel('Vreme [sati]')
ax2.set_ylabel('Ojlerovi uglovi [stepeni]')
ax2.set_title('Evolucija orijentacije u prostoru')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()