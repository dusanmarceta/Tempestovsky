import numpy as np
from linearni_model import yarko_diurnal_circular, yarko_seasonal_circular
from linearni_model_eccentric import yarko_eccentric
from constants import au, y2s
from astropy import constants


GM = constants.GM_sun.value
# Constants

#rho = 5000 # gustina (kg/m^3)
#k = 40 # koeficijent toplotne provodljivosti (W/(m*K))
#epsi = 1. # emissivity of the surface element
#cp = 500. # Toplotni kapacitet pri konstantnom pritisku (J/kg K)
#albedo = 0.
#semi_major_axis = 1 # au
#R = 100 # asteroid radius (m)
#rotation_period = 300 # seconds
#gam = 90. # spin axis obliquity (deg)





rho = 1000 # gustina (kg/m^3)

TI = 700
#k = 1e-3 # koeficijent toplotne provodljivosti (W/(m*K))
epsi = 0.95 # emissivity of the surface element
cp = 1000. # Toplotni kapacitet pri konstantnom pritisku (J/kg K)
albedo = 0.
semi_major_axis = 0.8 # au
eccentricity = 0.0
R = 238.97541362458387 # asteroid radius (m)
#R = 249.2796269674462 # asteroid radius (m)
rotation_period = 60. # hours
gamma = 92 # nagib ose rotacije od vertikale

mean_motion = np.sqrt(GM / (semi_major_axis * au)**3)
T = 2*np.pi / mean_motion


k = TI**2/(cp*rho)

l_diurnal = np.sqrt(k*rotation_period * 3600/rho/cp/(2*np.pi))
l_seasonal = np.sqrt(k*T/rho/cp/(2*np.pi))




drift_diurnal = yarko_diurnal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period, 1-albedo, epsi)

drift_seasonal = yarko_seasonal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period, 1-albedo, epsi)


drift_eccentric = yarko_eccentric(semi_major_axis, eccentricity, rho, k, cp, R, gamma, rotation_period, 1 - albedo, epsi, 0)

                                        

print('\n======= Wave depths =======')
print(f'diurnal wave depth: {np.round(l_diurnal, 3)}')
print(f'seasonal wave depth: {np.round(l_seasonal, 3)}')

print('\n======= Yarkovsky diurnal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_diurnal, 10), np.round(drift_diurnal * y2s /1000, 6), np.round(drift_diurnal * y2s / au * 1e6, 6)))
print('======= Yarkovsky seasonal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_seasonal, 10), np.round(drift_seasonal * y2s /1000, 6), np.round(drift_seasonal * y2s / au * 1e6, 6)))


print('======= Yarkovsky eccentric =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_eccentric, 10), np.round(drift_eccentric * y2s /1000, 6), np.round(drift_eccentric * y2s / au * 1e6, 6)))




print(f'diurnal/seasonal ratio: {np.round(drift_diurnal / drift_seasonal, 3)}')