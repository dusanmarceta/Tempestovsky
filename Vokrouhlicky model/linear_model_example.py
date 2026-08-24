import numpy as np
from linearni_model import yarko_diurnal_circular, yarko_seasonal_circular
from linearni_model_eccentric import yarko_eccentric
from constants import au, y2s

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

TI = 500
#k = 1e-3 # koeficijent toplotne provodljivosti (W/(m*K))
epsi = 0.95 # emissivity of the surface element
cp = 1000. # Toplotni kapacitet pri konstantnom pritisku (J/kg K)
albedo = 0.
semi_major_axis = 0.8 # au
eccentricity = 0.1
R = 238.97541362458387 # asteroid radius (m)
#R = 250 # asteroid radius (m)
rotation_period = 4. # hours
gamma = 73


k = TI**2/(cp*rho)


drift_diurnal = yarko_diurnal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period, 1-albedo, epsi)

drift_seasonal = yarko_seasonal_circular(rho, k, cp, R, semi_major_axis, gamma, rotation_period, 1-albedo, epsi)


drift_eccentric = yarko_eccentric(semi_major_axis, eccentricity, rho, k, cp, R, gamma, rotation_period, 1 - albedo, epsi, 0)

                                        


print('\n======= Yarkovsky diurnal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_diurnal, 6), np.round(drift_diurnal * y2s /1000, 6), np.round(drift_diurnal * y2s / au * 1e6, 6)))
print('======= Yarkovsky seasonal =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_seasonal, 6), np.round(drift_seasonal * y2s /1000, 6), np.round(drift_seasonal * y2s / au * 1e6, 6)))


print('======= Yarkovsky eccentric =======')
print('\n{} m/s, \n\n{} km/god, \n\n{} au/my\n'.format(np.round(drift_eccentric, 10), np.round(drift_eccentric * y2s /1000, 6), np.round(drift_eccentric * y2s / au * 1e6, 6)))




print(drift_diurnal / drift_seasonal)