import os
import sys
import multiprocessing
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import InstDebug
from app.instruments.instdebug import InstDebug

class DebugInstrument:
    """
    Wrapper for InstDebug that adapts it to the instrument interface
    expected by the InstrumentFactory.
    """
    
    def __init__(self, serial_number: str = "DEBUG001"):
        """
        Initialize the debug instrument
        
        Args:
            serial_number: Optional serial number for the debug instrument
        """
        self.inst = InstDebug(serial_number=serial_number, simulation_mode="normal")
        self.mapping = None
        
    def set_result_queue(self, queue: multiprocessing.Queue) -> None:
        """
        Set the result queue for sending experiment updates
        
        Args:
            queue: The multiprocessing Queue to send results to
        """
        self.inst.set_result_queue(queue)
    
    def set_mapping(self, mapping_name: str) -> None:
        """
        Set the mapping for this instrument
        
        Args:
            mapping_name: Name of the mapping to use
        """
        self.mapping = mapping_name
        # Create a simple channel mapper object that matches the interface
        # expected by InstDebug.set_channel_mapper
        class SimpleChannelMapper:
            def __init__(self, name, channels):
                self.name = name
                self.nwords = channels
                self.nbits = channels
                
        # Create a simple mapper with 16 channels
        mapper = SimpleChannelMapper(mapping_name, 16)
        self.inst.set_channel_mapper(mapper)
    
    def connect(self) -> bool:
        """Connect to the instrument"""
        return self.inst.connect()
    
    def disconnect(self) -> bool:
        """Disconnect from the instrument"""
        return self.inst.disconnect()
    
    def is_connected(self) -> bool:
        """Check if the instrument is connected"""
        return self.inst.is_connected()
    
    def get_id(self) -> str:
        """Get the instrument ID"""
        return self.inst.get_id()
    
    def simulate_experiment(self, experiment_type: str, params: dict) -> None:
        """
        Simulate an experiment
        
        Args:
            experiment_type: Type of experiment to simulate
            params: Dictionary of experiment parameters
        """
        self.inst.simulate_experiment(experiment_type, params)
    
    def measure_voltage(self, channel: int) -> float:
        """
        Measure voltage on a channel
        
        Args:
            channel: Channel to measure
            
        Returns:
            Measured voltage in volts
        """
        return self.inst.measure_voltage(channel)
    
    def measure_current(self, channel: int) -> float:
        """
        Measure current on a channel
        
        Args:
            channel: Channel to measure
            
        Returns:
            Measured current in amps
        """
        return self.inst.measure_current(channel)
    
    def set_voltage(self, channel: int, voltage: float) -> None:
        """
        Set voltage on a channel
        
        Args:
            channel: Channel to set
            voltage: Voltage to set in volts
        """
        self.inst.set_voltage(channel, voltage)
    
    def set_current(self, channel: int, current: float) -> None:
        """
        Set current on a channel
        
        Args:
            channel: Channel to set
            current: Current to set in amps
        """
        self.inst.set_current(channel, current)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the instrument
        
        Returns:
            Dictionary of instrument information
        """
        return self.inst.get_info()
    
    def __getattr__(self, name):
        """
        Forward any other attribute requests to the underlying InstDebug instance
        
        Args:
            name: Attribute name
            
        Returns:
            The requested attribute from the underlying InstDebug instance
        """
        return getattr(self.inst, name) 