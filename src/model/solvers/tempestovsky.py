import matplotlib.pyplot as plt

'''
proveriti insolaciju i uporediti sa njihovom insolacijom, nesto tu nije u redu, verovatno se dobija mnogo vise.

'''
import numpy as np
from numba import jit
from .base_solver import TemperatureSolver
from src.utilities.utils import conditional_print
from src.model.insolation import calculate_insolation, calculate_insolation_whole_orbit
from src.model.simulation import ThermalData_propagation
import matplotlib.pyplot as plt
import astropy.constants as const
import time

# Standalone numba functions
@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    if len(visible_facets) == 0 or len(view_factors) == 0:
        return 0.0
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)

@jit(nopython=True)
def calculate_temperatures(temperatures, layer_temperatures, insolation, visible_facets_list, 
                        view_factors_list, const1, const2, const3, self_heating_const,
                        timesteps_per_day, n_layers, include_self_heating):

    n_facets = temperatures.shape[0]
    current_column = 0  # Use column 0 for current timestep
    prev_column = 1    # Use column 1 for previous timestep
    
    for time_step in range(timesteps_per_day):
        # Swap columns for next iteration
        current_column, prev_column = prev_column, current_column
        
        for i in range(n_facets):
            # Surface temperature calculation
            prev_temp = layer_temperatures[i, prev_column, 0]
            prev_temp_layer1 = layer_temperatures[i, prev_column, 1]
            
            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                    insolation_term + 
                    re_emitted_radiation_term + 
                    conducted_heat_term + 
                    secondary_radiation_term)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # Update subsurface temperatures
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                            2 * prev_layer + 
                            prev_layer_minus)
                )
                    
    return temperatures



def update_thermal_state(thermal_data, current_insolation, simulation):
    '''
    thermal_data.layer_temperatures: (n_facets, n_layers) - trenutne temperature
    current_insolation: (n_facets) - fluks za ovaj specifični trenutak na orbiti
    '''
    
    T = thermal_data.layer_temperatures

    T_new = np.copy(T)

    coeff = simulation.thermal_diffusivity * simulation.delta_t / (simulation.layer_thickness**2)
    
    if coeff > 0.5:
        print("Upozorenje: Model je numerički nestabilan.")

    T_new[:, 1:-1] = T[:, 1:-1] + coeff * (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2])
    T_new[:, -1] = T_new[:, -2]
    
    # OVDE JE BILA GREŠKA: 
    # Dodajemo [:, 0] i [:, 1] da bi rezultat bio (1266,) a ne (1266, 45)
    conduction_to_subsurface = simulation.thermal_conductivity * (T[:, 0] - T[:, 1]) / simulation.layer_thickness

    
    # Sada će ova linija raditi jer su svi nizovi (1266,)
    dT_surf = (simulation.delta_t / (simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness)) * (current_insolation - simulation.emissivity * const.sigma_sb.value * T[:, 0]**4 - conduction_to_subsurface)
    
    
    
    T_new[:, 0] = T[:, 0] + dT_surf
    
#    print('----------------------------------------------------------------------')
#    
#    print(np.max(current_insolation))
#    print(simulation.delta_t)
#    print('Prethodna srednja temperatura = ',np.mean(T[:, 0]))
#    print('Trenutna srednja temperatura = ',np.mean(T_new[:, 0]))

    return T_new



def calculate_temperatures_whole_orbit(temperatures, layer_temperatures, insolation, visible_facets_list, 
                        view_factors_list, const1, const2, const3, self_heating_const,
                        timesteps_per_orbit, n_layers, include_self_heating):

    '''
    treba da upisa u  prvu vrednost temperaturu svih celija i da izracuna novu vrednos. Onda tu vrednost da upise kao prethodnu temeperaturu za
    sledeci vremenski korak
    
    Ovo je suprotno od prethodne funkcije koje sluzi za konvergenciju koja za sve trenutke ima prethodnu temperaturu pa ih racuna ponovo za sledecu rotaciju
    
    potrebno je i promeniti velicinu niza temeratures
    
    '''
    
    n_facets = temperatures.shape[0]
    current_column = 0  # Use column 0 for current timestep
    prev_column = 1    # Use column 1 for previous timestep
    
    for time_step in range(timesteps_per_orbit):
        # Swap columns for next iteration
        current_column, prev_column = prev_column, current_column # NAJVEROVATNIJE OVDE TREBA DA SE INTERVENISE
        
        for i in range(n_facets):
            # Surface temperature calculation
            prev_temp = layer_temperatures[i, prev_column, 0]
            prev_temp_layer1 = layer_temperatures[i, prev_column, 1]
            
            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                    insolation_term + 
                    re_emitted_radiation_term + 
                    conducted_heat_term + 
                    secondary_radiation_term)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # Update subsurface temperatures
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                            2 * prev_layer + 
                            prev_layer_minus)
                )
                    
    return temperatures


    
'''
ADDED 

'''

def calculate_yarkovsky(simulation, mean_motion, asteroid_mass, normals, areas, temperatures):
    
    F=2/3*simulation.emissivity*const.sigma_sb.value / const.c.value * np.sum(temperatures[:, None]**4 * normals * areas[:, None], axis=0)
 
    '''
    OVO SREDITI!
    '''
    
    true_anomaly = 0
    sun_distance = 1
    
    r_trans = np.array([0, 1, 0])
    r_rad = np.array([1, 0, 0])
    
    '''
    dalje je sve OK
    '''
    
    
    B = np.dot(F, r_trans)
        
    # Radial thermal force
    R = np.dot(F, r_rad)

    
    dadt = 2 * mean_motion * (simulation.a_au * const.au.value)**2 / const.GM_sun.value * (
                    R * simulation.a_au * const.au.value * simulation.ecc * np.sin(true_anomaly) / np.sqrt(1-simulation.ecc**2) + 
                    B * simulation.a_au**2 * const.au.value * np.sqrt(1-simulation.ecc**2) / sun_distance)/asteroid_mass
            
  
    return dadt






class YarkovskySolver(TemperatureSolver):
    def __init__(self):
        super().__init__("tempest_standard_yarko")
        self.required_parameters = [
            "emissivity",
            "density",
            "specific_heat_capacity",
            "thermal_conductivity",
            "n_layers",
            "convergence_target",
            "beaming_factor"
        ]

    def solve(self, thermal_data, shape_model, asteroid_mass, normals, areas, simulation, mean_motion, config):
        ''' 
        This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
        insolation and temperature arrays, and iterate until the model converges.
        '''

        # Initialize constants
        const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
        self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity**2 / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

        convergence_error = simulation.convergence_target + 1
        day = 0
        error_history = []
        comparison_temps = thermal_data.temperatures[:, 0].copy()
        
        # Initialization

        while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
            current_day_temperature = calculate_temperatures(
                thermal_data.temperatures,
                thermal_data.layer_temperatures,
                thermal_data.insolation,
                thermal_data.visible_facets,
                thermal_data.thermal_view_factors,
                const1, const2, const3, self_heating_const,
                simulation.timesteps_per_day, simulation.n_layers,
                config.include_self_heating
            )

            # Check for invalid temperatures
            for i in range(len(shape_model)):
                for time_step in range(simulation.timesteps_per_day):
                    if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                        conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                        return {
                            "final_day_temperatures": None,
                            "final_day_temperatures_all_layers": None,
                            "final_timestep_temperatures": None,
                            "days_to_convergence": day,
                            "mean_temperature_error": None,
                            "max_temperature_error": None
                        }

            if config.calculate_energy_terms:
                energy_terms = self.calculate_energy_terms(
                    current_day_temperature, 
                    thermal_data.insolation,
                    simulation.delta_t,
                    simulation.emissivity,
                    simulation.beaming_factor,
                    simulation.density,
                    simulation.specific_heat_capacity,
                    simulation.layer_thickness,
                    simulation.thermal_conductivity,
                    simulation.timesteps_per_day,
                    simulation.n_layers
                )
                
                thermal_data.insolation_energy = energy_terms[:, :, 0]
                thermal_data.re_emitted_energy = energy_terms[:, :, 1]
                thermal_data.surface_energy_change = energy_terms[:, :, 2]
                thermal_data.conducted_energy = energy_terms[:, :, 3]
                thermal_data.unphysical_energy_loss = energy_terms[:, :, 4]

            # Calculate convergence
            temperature_errors = np.abs(current_day_temperature[:, 0] - comparison_temps)
            
            if config.convergence_method == 'mean':
                convergence_error = np.mean(temperature_errors)
            else:
                convergence_error = np.max(temperature_errors)

            max_temperature_error = np.max(temperature_errors)
            mean_temperature_error = np.mean(temperature_errors)

            conditional_print(config.silent_mode, f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Max Temp Error: {max_temperature_error:.6f} K")
            
            comparison_temps = current_day_temperature[:, 0].copy()
            error_history.append(convergence_error)
            day += 1
            
            
            
        

        # Check for invalid temperatures
        for i in range(len(shape_model)):
            for time_step in range(simulation.timesteps_per_day):
                if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                    conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                    return {
                        "final_day_temperatures": None,
                        "final_day_temperatures_all_layers": None,
                        "final_timestep_temperatures": None,
                        "days_to_convergence": day,
                        "mean_temperature_error": None,
                        "max_temperature_error": None
                    }

        if config.calculate_energy_terms:
            energy_terms = self.calculate_energy_terms(
                current_day_temperature, 
                thermal_data.insolation,
                simulation.delta_t,
                simulation.emissivity,
                simulation.beaming_factor,
                simulation.density,
                simulation.specific_heat_capacity,
                simulation.layer_thickness,
                simulation.thermal_conductivity,
                simulation.timesteps_per_day,
                simulation.n_layers
            )
            
            thermal_data.insolation_energy = energy_terms[:, :, 0]
            thermal_data.re_emitted_energy = energy_terms[:, :, 1]
            thermal_data.surface_energy_change = energy_terms[:, :, 2]
            thermal_data.conducted_energy = energy_terms[:, :, 3]
            thermal_data.unphysical_energy_loss = energy_terms[:, :, 4]
            
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        print('Initialization finished')
                
        precomputed_insolation, r_sunca, ugao_sunca = calculate_insolation_whole_orbit(thermal_data, shape_model, simulation, config)
        
        print('**********************************************')
        print('**********************************************')
        print('**********************************************')
        
        surface_history = np.zeros_like(precomputed_insolation)
        
        mean_T = np.zeros(simulation.timesteps_per_orbit)
        mean_insolation = np.zeros(simulation.timesteps_per_orbit)
        
        thermal_data.layer_temperatures = thermal_data.layer_temperatures[:, 1, :]
        
        for t in range(simulation.timesteps_per_orbit):
            # KLJUČNI MOMENAT: 
            # Pozivamo funkciju i REZULTAT upisujemo nazad u thermal_data.
            # Tako u sledećoj iteraciji (t+1) funkcija uzima temperaturu od (t).
            thermal_data.layer_temperatures = update_thermal_state(thermal_data, precomputed_insolation[:, t], simulation)

            # Ovde možeš sačuvati površinsku temperaturu za ovaj trenutak ako ti treba za grafikon
            surface_history[:, t] = thermal_data.layer_temperatures[:, 0]
            mean_T[t] = np.mean(surface_history[:, t])
            mean_insolation[t] = np.mean(precomputed_insolation[:, t])
            
            if np.mod(t, 50) == 0:
                print(f'step {t} out of {simulation.timesteps_per_orbit}, mean(T) = {np.round(mean_T[t], 2)}')

        plt.figure()
        plt.plot(np.arange(simulation.timesteps_per_orbit) * simulation.delta_t, r_sunca)
        plt.title('r_sunca', fontsize = 16)
        plt.grid()
        plt.show()
        
        
        plt.figure()
        plt.plot(np.arange(simulation.timesteps_per_orbit) * simulation.delta_t, mean_T)
        plt.title('Mean surface temperature', fontsize = 16)
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.plot(np.arange(simulation.timesteps_per_orbit) * simulation.delta_t, mean_insolation)
        plt.title('Mean insolation', fontsize = 16)
        plt.grid()
        plt.show()
        
        
        
        
        return {
            "final_day_temperatures": current_day_temperature,
            "final_day_temperatures_all_layers": thermal_data.layer_temperatures,
            "final_timestep_temperatures": current_day_temperature[:, -1],
            "days_to_convergence": day,
            "mean_temperature_error": mean_temperature_error,
            "max_temperature_error": max_temperature_error
        } 
            
