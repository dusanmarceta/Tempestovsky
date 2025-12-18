from .factory import TemperatureSolverFactory
from .tempest_standard import TempestStandardSolver
from .tempestovsky import YarkovskySolver

# Register available solvers
TemperatureSolverFactory.register("tempest_standard", TempestStandardSolver) 
TemperatureSolverFactory.register("tempestovsky", YarkovskySolver)