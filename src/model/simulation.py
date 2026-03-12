# src/model/simulation.py

import numpy as np
from src.utilities.locations import Locations
import astropy.constants as const
from scipy.optimize import fsolve


class Simulation:
    def __init__(self, config):
        """
        Initialize the simulation using the provided Config object.
        """
        self.config = config
        self.load_configuration()

    def load_configuration(self):
        """
        Load configuration directly from the Config object.
        """
        # Assign configuration to attributes, converting lists to numpy arrays as needed
        for key, value in self.config.config_data.items():
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, key, value)
        
        # ADDED
        self.mean_motion = np.sqrt(const.GM_sun.value / (self.a_au * const.au.value)**3)
        self.orbital_period = 2*np.pi / self.mean_motion
        # -----------------------------------------------------------------
        # Initialization calculations based on the loaded parameters
#        self.solar_distance_m = self.solar_distance_au * 1.496e11  # Convert AU to meters
        self.rotation_period_s = self.rotation_period_hours * 3600  # Convert hours to seconds
        self.angular_velocity = (2 * np.pi) / self.rotation_period_s
        self.thermal_conductivity = (self.thermal_inertia**2 / (self.density * self.specific_heat_capacity))
        skin_depth_diurnal = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity)) ** 0.5
        skin_depth_seasonal = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.mean_motion)) ** 0.5
        
        if self.yarkovsky_component == 'seasonal':
            
            self.skin_depth = skin_depth_seasonal
            
            self.total_thickness = self.n_skin_depths * self.skin_depth
            self.first_layer_thickness = self.thicknes_0 * self.skin_depth
          
        elif self.yarkovsky_component == 'diurnal':
            
            self.skin_depth = skin_depth_diurnal
            
            self.total_thickness = self.n_skin_depths * self.skin_depth
            self.first_layer_thickness = self.thicknes_0 * self.skin_depth
            
        elif self.yarkovsky_component == 'general':
            
            self.skin_depth = np.min([skin_depth_diurnal, skin_depth_seasonal])
            
            self.total_thickness = self.n_skin_depths * np.max([skin_depth_seasonal, skin_depth_diurnal])
            self.first_layer_thickness = self.thicknes_0 * self.skin_depth
            

        self.dz, self.dz_center_up, self.dz_center_down = self.calculate_layer_thicknesses()
        

        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat_capacity)
        self.timesteps_per_day = self.calculate_adaptive_timesteps() # Adaptive timestep for low thermal inertia stability
        self.delta_t = self.rotation_period_s / self.timesteps_per_day
 
        
        self.timesteps_per_orbit = int(np.ceil(self.orbital_period / self.delta_t))
        
#        self.timesteps_per_orbit = int(np.ceil(self.orbital_period / self.delta_t))
        
        # Compute unit vector from RA and Dec
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                                       np.sin(ra_radians) * np.cos(dec_radians), 
                                       np.sin(dec_radians)])
    
        self.orbital_period = 2*np.pi / np.sqrt(const.GM_sun.value /(self.a_au * const.au.value)**3)
        self.timesteps_per_year = int(np.ceil(self.orbital_period / self.delta_t))
        

    def calculate_adaptive_timesteps(self):
        """
        Calculate timesteps with stability constraints for low thermal inertia.
        np.sqrt(const.GM_sun.value /(simulation.a_au * const.au.value)**3)
        This method adds an adaptive constraint that limits const1 to reasonable
        values, ensuring stability for low thermal inertia materials.
        """
        # Original CFL calculation
        cfl_denominator = self.first_layer_thickness**2 / (2 * self.thermal_diffusivity)
        timesteps_cfl = int(round(self.rotation_period_s / cfl_denominator))
        delta_t_cfl = self.rotation_period_s / timesteps_cfl
        
        # Calculate insolation coefficient with CFL timestep
        const1_cfl = delta_t_cfl / (self.first_layer_thickness * self.density * self.specific_heat_capacity)
        
        # Adaptive constraint: limit const1 for stability
        max_const1 = 0.1  # Maximum allowed insolation coefficient
        
        if const1_cfl > max_const1:
            # Calculate timestep that keeps const1 reasonable
            required_delta_t = max_const1 * self.first_layer_thickness * self.density * self.specific_heat_capacity
            adaptive_timesteps = int(np.ceil(self.rotation_period_s / required_delta_t))
            
            # Apply additional safety factor for very low thermal inertia
            if self.thermal_inertia < 100:
                safety_factor = 0.5
                adaptive_timesteps = int(adaptive_timesteps / safety_factor)
            
            return adaptive_timesteps
        else:
            return timesteps_cfl
        
        
    def calculate_layer_thicknesses(self):
        
        D = self.total_thickness
        x1 = self.first_layer_thickness
        N = self.n_layers
        
        print(D, x1, N)
    
        if D/x1 < N: # if the first section is to large so that other must be decreased to reach N sections
            # we set equidistant division so that the section size is smaller than the required first section x1
            
            N = int(np.ceil(D/x1)) + 1 
            
            layer_depths = np.linspace(0, D, N)
        
        else:
            # Funkcija koja vraća grešku za dati r
            def error(r):
                return x1 * (r**N - 1) / (r - 1) - D
        
            # Početna procena za r
            initial_guess = 1.1
            
            # Izračunaj r pomoću fsolve (Newton-Raphson metoda)
            r_solution = fsolve(error, initial_guess)[0]
            
            
            
            # Generiši korake koristeći izračunat r
            steps = x1 * r_solution ** np.arange(N)
            
            # Podesi korake tako da njihov zbir bude tačno D
            steps *= D / np.sum(steps)
        
            # Kreiraj tačke podele
            layer_depths = np.concatenate(([0], np.cumsum(steps)))
            
        dz = np.diff(layer_depths)
 
        dz_center_up = np.empty(len(dz))
        dz_center_down = np.empty(len(dz))
        
        # Površinski sloj
        dz_center_up[0] = np.inf
        dz_center_down[0] = 0.5 * dz[0] + 0.5 * dz[1]
        
        # Srednji slojevi (samo unutrašnji)
        for layer in range(1, len(dz) - 1):
            dz_center_up[layer] = 0.5 * dz[layer] + 0.5 * dz[layer - 1]    # do centra sloja iznad
            dz_center_down[layer] = 0.5 * dz[layer] + 0.5 * dz[layer + 1]  # do centra sloja ispod
        
        # Donji sloj
        dz_center_up[-1] = 0.5 * dz[-1] + 0.5 * dz[-2]
        dz_center_down[-1] = np.inf
        
        return dz, dz_center_up, dz_center_down
        
    
    
def calculate_layer_thicknesses_proba(D, x1, N):
    

    if D/x1 < N: # if the first section is to large so that other must be decreased to reach N sections
        # we set equidistant division so that the section size is smaller than the required first section x1
        
        N = int(np.ceil(D/x1)) + 1 
        
        layer_depths = np.linspace(0, D, N)
    
    else:
        # Funkcija koja vraća grešku za dati r
        def error(r):
            return x1 * (r**N - 1) / (r - 1) - D
    
        # Početna procena za r
        initial_guess = 1.1
        
        # Izračunaj r pomoću fsolve (Newton-Raphson metoda)
        r_solution = fsolve(error, initial_guess)[0]
        
        
        
        # Generiši korake koristeći izračunat r
        steps = x1 * r_solution ** np.arange(N)
        
        # Podesi korake tako da njihov zbir bude tačno D
        steps *= D / np.sum(steps)
    
        # Kreiraj tačke podele
        layer_depths = np.concatenate(([0], np.cumsum(steps)))
        
    dz = np.diff(layer_depths)
    
    dz_center_up = np.empty(len(dz))
    dz_center_down = np.empty(len(dz))
    
    # Površinski sloj
    dz_center_up[0] = np.inf
    dz_center_down[0] = 0.5 * dz[0] + 0.5 * dz[1]
    
    # Srednji slojevi (samo unutrašnji)
    for layer in range(1, len(dz) - 1):
        dz_center_up[layer] = 0.5 * dz[layer] + 0.5 * dz[layer - 1]    # do centra sloja iznad
        dz_center_down[layer] = 0.5 * dz[layer] + 0.5 * dz[layer + 1]  # do centra sloja ispod
    
    # Donji sloj
    dz_center_up[-1] = 0.5 * dz[-1] + 0.5 * dz[-2]
    dz_center_down[-1] = np.inf
    
    return dz, dz_center_up, dz_center_down
    

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        # Surface temperatures for one day only
        self.temperatures = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        # Two columns for current and previous timestep subsurface temperatures
        self.layer_temperatures = np.zeros((n_facets, 2, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]
        self.thermal_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        if calculate_energy_terms:
            # Energy terms for one day only
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day))

    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
    def set_thermal_view_factors(self, view_factors):
        """Set thermal view factors for all facets."""
        self.thermal_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
             
class ThermalData_propagation:
    def __init__(self, n_facets, n_layers, calculate_energy_terms):
        # Surface temperatures for one day only
        self.temperatures = np.zeros((n_facets, 1), dtype=np.float64)
        # Two columns for current and previous timestep subsurface temperatures
        self.layer_temperatures = np.zeros((n_facets, 2, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, 1), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]
        self.thermal_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        if calculate_energy_terms:
            # Energy terms for one day only
            self.insolation_energy = np.zeros((n_facets, 1))
            self.re_emitted_energy = np.zeros((n_facets, 1))
            self.surface_energy_change = np.zeros((n_facets, 1))
            self.conducted_energy = np.zeros((n_facets, 1))
            self.unphysical_energy_loss = np.zeros((n_facets, 1))

    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
    def set_thermal_view_factors(self, view_factors):
        """Set thermal view factors for all facets."""
        self.thermal_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]