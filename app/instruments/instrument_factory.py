# instrument_factory.py
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from app.instruments.debug_instrument import DebugInstrument

class InstrumentFactory:
    """
    Factory class for creating instrument instances
    """
    
    def __init__(self):
        """Initialize the instrument factory"""
        # Keep track of connected instruments
        self.connected_instruments = {}
        
        # Register supported instrument types
        self.instrument_types = {
            "InstDebug (Simulation)": DebugInstrument
        }
        
        # Try to import PyArC2 if available
        try:
            import pyarc2
            self.has_pyarc2 = True
            self.instrument_types["ARC2"] = pyarc2.Instrument
        except ImportError:
            self.has_pyarc2 = False
            print("PyArC2 not available. Only simulation instruments will be supported.")
    
    def get_instrument(self, instrument_id: str, mapping_name: str = None) -> Any:
        """
        Get an instrument instance
        
        Args:
            instrument_id: ID of the instrument to get
            mapping_name: Name of the mapping to use
            
        Returns:
            The instrument instance
        """
        # Check if the instrument is already connected
        if instrument_id in self.connected_instruments:
            return self.connected_instruments[instrument_id]
        
        # Extract the instrument type from the ID
        if "(" in instrument_id and ")" in instrument_id:
            # Format like "InstDebug (Simulation)"
            instrument_type = instrument_id
        else:
            # Default to ARC2
            instrument_type = "ARC2"
        
        # Check if the instrument type is supported
        if instrument_type not in self.instrument_types:
            raise ValueError(f"Unsupported instrument: {instrument_id}")
        
        # Create the instrument
        if instrument_type == "InstDebug (Simulation)":
            # Simulation instrument
            instrument = self.instrument_types[instrument_type]()
        elif instrument_type == "ARC2":
            # Real ARC2 instrument
            if not self.has_pyarc2:
                raise ValueError("PyArC2 not available. Cannot create ARC2 instrument.")
            instrument = self.instrument_types[instrument_type]()
            # Connect to the ARC2 instrument
            instrument.connect()
        else:
            raise ValueError(f"Unsupported instrument type: {instrument_type}")
        
        # Apply mapping if specified
        if mapping_name:
            self.apply_mapping(instrument, mapping_name)
        
        # Store the connected instrument
        self.connected_instruments[instrument_id] = instrument
        
        return instrument
    
    def apply_mapping(self, instrument: Any, mapping_name: str) -> None:
        """
        Apply a mapping to an instrument
        
        Args:
            instrument: The instrument to apply the mapping to
            mapping_name: Name of the mapping to apply
        """
        # For simulation instruments, just set the mapping name
        if isinstance(instrument, DebugInstrument):
            instrument.set_mapping(mapping_name)
            return
        
        # For real instruments, load and apply the mapping
        try:
            from app.instruments.arc2custom import mapper
            mapping = mapper.load_mapping(mapping_name)
            if hasattr(instrument, "apply_mapping"):
                instrument.apply_mapping(mapping)
        except Exception as e:
            print(f"Error applying mapping {mapping_name}: {e}")
    
    def disconnect_instrument(self, instrument_id: str) -> bool:
        """
        Disconnect an instrument
        
        Args:
            instrument_id: ID of the instrument to disconnect
            
        Returns:
            True if the instrument was disconnected, False otherwise
        """
        if instrument_id in self.connected_instruments:
            instrument = self.connected_instruments[instrument_id]
            try:
                if hasattr(instrument, "disconnect"):
                    instrument.disconnect()
                del self.connected_instruments[instrument_id]
                return True
            except Exception as e:
                print(f"Error disconnecting instrument {instrument_id}: {e}")
        return False
    
    def get_available_instruments(self) -> Dict[str, str]:
        """
        Get the available instruments
        
        Returns:
            Dictionary mapping instrument IDs to descriptions
        """
        instruments = {
            "InstDebug (Simulation)": "Simulation instrument for testing"
        }
        
        # Add real instruments if PyArC2 is available
        if self.has_pyarc2:
            try:
                import pyarc2
                found_instruments = pyarc2.find_instruments()
                for inst in found_instruments:
                    instruments[f"ARC2 {inst.serial}"] = f"ARC2 Serial: {inst.serial}"
            except Exception as e:
                print(f"Error finding ARC2 instruments: {e}")
        
        return instruments
