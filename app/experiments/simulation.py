"""
Simulation module to provide placeholders for experiments when the real modules are not available.
This allows the application to run in simulation mode without requiring all dependencies.
"""

import time
import sys
import os
from typing import Dict, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


class SimulationExperiment:
    """Base class for simulation experiments"""
    
    def __init__(self, instrument, name, session=None):
        """
        Initialize the simulation experiment
        
        Args:
            instrument: The instrument to use
            name: Name of the experiment
            session: Optional session object
        """
        self.instrument = instrument
        self.name = name
        self.session = session
        self.settings = SimulationSettings()
    
    def run(self):
        """Run the experiment using the instrument's simulate_experiment method"""
        # Extract settings as a dictionary
        settings_dict = self.settings.as_dict()
        
        # Run the experiment simulation
        experiment_type = self.__class__.__name__
        self.instrument.simulate_experiment(experiment_type, settings_dict)


class SimulationSettings:
    """Settings class for simulation experiments"""
    
    def __init__(self):
        """Initialize with default settings"""
        # Common settings
        self.channels = [0, 1, 2, 3]
        
        # Specific settings for different experiment types
        # IV Measurement
        self.start_voltage = -1.0
        self.end_voltage = 1.0
        self.voltage_steps = 100
        
        # Pulse Measurement
        self.pulse_amplitude = 1.0
        self.pulse_width = 10  # ms
        self.num_pulses = 5
        
        # Noise Measurement
        self.duration = 10  # seconds
        self.sample_rate = 100  # Hz
        self.bias_voltage = 0.1  # V
        
        # Memory Capacity
        self.input_length = 100
        self.warmup_time = 20
        
        # Other experiment types
        self.steps = 100
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert settings to a dictionary"""
        return {name: value for name, value in vars(self).items() if not name.startswith('_')}


# Create placeholder classes for all standard experiment types
class IVMeasurement(SimulationExperiment):
    """Perform current-voltage measurements"""
    pass


class PulseMeasurement(SimulationExperiment):
    """Perform pulse response measurements"""
    pass


class NoiseMeasurement(SimulationExperiment):
    """Measure noise characteristics"""
    pass


class MemoryCapacity(SimulationExperiment):
    """Measure memory capacity"""
    pass


class ActivationPattern(SimulationExperiment):
    """Test activation patterns"""
    pass


class ConductivityMatrix(SimulationExperiment):
    """Measure conductivity matrix"""
    pass


class ReservoirComputing(SimulationExperiment):
    """Test reservoir computing"""
    pass


class Tomography(SimulationExperiment):
    """Perform tomography measurements"""
    pass


class TurnOn(SimulationExperiment):
    """Turn on device test"""
    pass 