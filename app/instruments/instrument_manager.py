import sys
import os
import multiprocessing
from multiprocessing import Queue
from typing import List, Dict, Any, Optional
import time

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

class InstrumentManager:
    """Class to manage instrument connections"""
    
    def __init__(self):
        self.connected_instruments = {}
        self.debug_mode = False
        
    def scan_for_instruments(self) -> List[str]:
        """
        Scan for connected instruments
        
        Returns:
            List of instrument IDs that were found
        """
        # In a real implementation, this would scan for actual hardware
        # For now, just return a dummy instrument list
        
        instruments = []
        
        # Always add debug instrument
        instruments.append("InstDebug (Simulation)")
        
        
        # Simulate finding other instruments
        # In real implementation, this would use hardware-specific APIs
        try:
            # Simulate a scan delay
            time.sleep(1)
            
            # Add some dummy instruments for demonstration
            instruments.append("DummyInstrument1 (Not Connected)")
            instruments.append("DummyInstrument2 (Not Connected)")
        except Exception as e:
            print(f"Error scanning for instruments: {e}")
        
        return instruments
    
    def connect_to_instrument(self, instrument_id: str) -> bool:
        """
        Connect to the specified instrument
        
        Args:
            instrument_id: The ID of the instrument to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        # Check if we're already connected
        if instrument_id in self.connected_instruments:
            return True
        
        # Handle debug instrument
        if "InstDebug" in instrument_id:
            self.debug_mode = True
            self.connected_instruments[instrument_id] = {
                "type": "debug",
                "connected": True,
                "instance": None  # No real connection needed for debug
            }
            return True
            
        
        
        # In a real implementation, this would connect to actual hardware
        # For now, just simulate a connection
        try:
            # Simulate connection delay
            time.sleep(0.5)
            
            # Simulate successful connection
            self.connected_instruments[instrument_id] = {
                "type": "dummy",
                "connected": True,
                "instance": None  # Would be the actual connection object
            }
            return True
        except Exception as e:
            print(f"Error connecting to instrument {instrument_id}: {e}")
            return False
    
    def disconnect_from_instrument(self, instrument_id: str) -> bool:
        """
        Disconnect from the specified instrument
        
        Args:
            instrument_id: The ID of the instrument to disconnect from
            
        Returns:
            True if disconnection was successful, False otherwise
        """
        if instrument_id not in self.connected_instruments:
            return True  # Already disconnected
        
        try:
            # In a real implementation, this would close the connection
            # For now, just remove from the dictionary
            del self.connected_instruments[instrument_id]
            
            # If we disconnected from debug, reset debug mode
            if "InstDebug" in instrument_id:
                self.debug_mode = False
                
            return True
        except Exception as e:
            print(f"Error disconnecting from instrument {instrument_id}: {e}")
            return False
    
    def get_connected_instruments(self) -> Dict[str, Any]:
        """
        Get a dictionary of connected instruments
        
        Returns:
            Dictionary mapping instrument IDs to connection details
        """
        return self.connected_instruments
    
    def is_connected(self, instrument_id: str) -> bool:
        """
        Check if an instrument is connected
        
        Args:
            instrument_id: The ID of the instrument to check
            
        Returns:
            True if the instrument is connected, False otherwise
        """
        return instrument_id in self.connected_instruments
    
    def get_instrument_instance(self, instrument_id: str) -> Optional[Any]:
        """
        Get the instance of a connected instrument
        
        Args:
            instrument_id: The ID of the instrument to get
            
        Returns:
            The instrument instance, or None if not connected
        """
        if not self.is_connected(instrument_id):
            return None
        
        return self.connected_instruments[instrument_id].get("instance")


class InstrumentManagerProcess:
    """Runs an instrument manager in a separate process"""
    
    def __init__(self):
        self.command_queue = Queue()
        self.result_queue = Queue()
        self.process = None
        
    def start(self):
        """Start the instrument manager process"""
        self.process = multiprocessing.Process(
            target=self._run_manager,
            args=(self.command_queue, self.result_queue)
        )
        self.process.start()
    
    def stop(self):
        """Stop the instrument manager process"""
        if self.process and self.process.is_alive():
            self.command_queue.put(("STOP", None))
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
    
    def _run_manager(self, command_queue, result_queue):
        """Run the instrument manager, processing commands from the queue"""
        manager = InstrumentManager()
        
        running = True
        while running:
            try:
                if not command_queue.empty():
                    command, args = command_queue.get()
                    
                    if command == "STOP":
                        running = False
                    elif command == "SCAN":
                        instruments = manager.scan_for_instruments()
                        result_queue.put(("SCAN_RESULT", instruments))
                    elif command == "CONNECT":
                        success = manager.connect_to_instrument(args)
                        result_queue.put(("CONNECT_RESULT", success))
                    elif command == "DISCONNECT":
                        success = manager.disconnect_from_instrument(args)
                        result_queue.put(("DISCONNECT_RESULT", success))
                    elif command == "GET_CONNECTED":
                        connected = manager.get_connected_instruments()
                        result_queue.put(("CONNECTED_RESULT", connected))
                
                # Sleep a bit to avoid busy waiting
                time.sleep(0.01)
            
            except Exception as e:
                print(f"Error in instrument manager process: {e}")
                result_queue.put(("ERROR", str(e)))
    
    def scan_for_instruments(self):
        """Scan for connected instruments"""
        self.command_queue.put(("SCAN", None))
        
        # Wait for the result
        while True:
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "SCAN_RESULT":
                    return result
            time.sleep(0.01)
    
    def connect_to_instrument(self, instrument_id):
        """Connect to the specified instrument"""
        self.command_queue.put(("CONNECT", instrument_id))
        
        # Wait for the result
        while True:
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "CONNECT_RESULT":
                    return result
            time.sleep(0.01)
    
    def disconnect_from_instrument(self, instrument_id):
        """Disconnect from the specified instrument"""
        self.command_queue.put(("DISCONNECT", instrument_id))
        
        # Wait for the result
        while True:
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "DISCONNECT_RESULT":
                    return result
            time.sleep(0.01)
    
    def get_connected_instruments(self):
        """Get a dictionary of connected instruments"""
        self.command_queue.put(("GET_CONNECTED", None))
        
        # Wait for the result
        while True:
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "CONNECTED_RESULT":
                    return result
            time.sleep(0.01)


# Example usage
if __name__ == "__main__":
    # Start the manager in a separate process
    manager_process = InstrumentManagerProcess()
    manager_process.start()
    
    # Scan for instruments
    instruments = manager_process.scan_for_instruments()
    print(f"Found instruments: {instruments}")
    
    # Connect to the debug instrument
    success = manager_process.connect_to_instrument("InstDebug (Simulation)")
    print(f"Connected to debug instrument: {success}")
    
    # Get connected instruments
    connected = manager_process.get_connected_instruments()
    print(f"Connected instruments: {connected}")
    
    # Disconnect
    success = manager_process.disconnect_from_instrument("InstDebug (Simulation)")
    print(f"Disconnected from debug instrument: {success}")
    
    # Stop the manager process
    manager_process.stop() 