import sys
import os
import multiprocessing
from multiprocessing import Queue
import time
import importlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import experiment-related modules
from app.instruments.instrument_manager import InstrumentManagerProcess


class ExperimentLauncher:
    """
    Class to launch and manage experiments with a modular approach
    """
    
    def __init__(self):
        """
        Initialize the experiment launcher, discovering available experiments
        """
        # Initialize instrument manager
        self.instrument_manager = InstrumentManagerProcess()
        self.instrument_manager.start()
        
        # Dictionary of registered experiment modules
        self.available_experiments = {}
        
        # Discover available experiment modules
        self._discover_experiment_modules()
    
    def _discover_experiment_modules(self):
        """
        Discover and register available experiment modules by scanning
        the experiment directories
        """
        # Primary experiment directory
        experiment_dir = Path(project_root) / "app" / "instruments" / "arc2custom" / "experiments"
        
        # If the directory doesn't exist, create a minimal set of experiment placeholders
        if not experiment_dir.exists():
            print("Experiment directory not found, creating simulated experiments")
            self._create_placeholder_experiments()
            return
            
        # Track any failures for debugging
        failures = []
            
        if experiment_dir.exists():
            # Scan for Python modules
            for module_file in experiment_dir.glob("*.py"):
                if module_file.name == '__init__.py' or module_file.name == 'experiment.py':
                    continue
                    
                module_name = module_file.stem
                full_module_path = f"app.instruments.arc2custom.experiments.{module_name}"
                
                try:
                    # Attempt to import the module
                    module = importlib.import_module(full_module_path)
                    
                    # Look for experiment classes - assume main class matches module name in camel case
                    class_name = ''.join(word.capitalize() for word in module_name.split('_'))
                    
                    # Check if the class exists in the module
                    if hasattr(module, class_name):
                        self.available_experiments[module_name] = {
                            "module": full_module_path,
                            "class": class_name,
                            "description": getattr(module, class_name).__doc__ or f"No description available for {module_name}"
                        }
                        print(f"Registered experiment: {module_name}")
                    else:
                        print(f"Warning: Module {module_name} does not contain expected class {class_name}")
                except ImportError as e:
                    failures.append((module_name, str(e)))
                    print(f"Error loading experiment module {full_module_path}: {e}")
        
        # If we didn't find any experiments or had failures, create placeholders
        if not self.available_experiments or failures:
            print(f"Failed to load some or all experiment modules ({len(failures)} failures)")
            self._create_placeholder_experiments()
    
    def _create_placeholder_experiments(self):
        """Create placeholder experiment entries when real ones can't be loaded"""
        # Define a list of standard experiments that should be available
        standard_experiments = [
            ("ivmeasurement", "IVMeasurement", "Perform current-voltage measurements"),
            ("pulsemeasurement", "PulseMeasurement", "Perform pulse response measurements"),
            ("noisemeasurement", "NoiseMeasurement", "Measure noise characteristics"),
            ("memorycapacity", "MemoryCapacity", "Measure memory capacity"),
            ("activationpattern", "ActivationPattern", "Test activation patterns"),
            ("conductivitymatrix", "ConductivityMatrix", "Measure conductivity matrix"),
            ("reservoircomputing", "ReservoirComputing", "Test reservoir computing"),
            ("tomography", "Tomography", "Perform tomography measurements"),
            ("turnon", "TurnOn", "Turn on device test")
        ]
        
        # Register placeholder experiments
        for module_name, class_name, description in standard_experiments:
            # Only add if not already present
            if module_name not in self.available_experiments:
                self.available_experiments[module_name] = {
                    "module": "app.experiments.simulation",  # This will be handled specially in run_experiment
                    "class": class_name,
                    "description": description,
                    "is_simulation": True  # Mark as simulation
                }
                print(f"Registered placeholder experiment: {module_name}")
    
    def get_available_experiments(self) -> List[str]:
        """
        Get the list of available experiments
        
        Returns:
            List of experiment names
        """
        return list(self.available_experiments.keys())
    
    def get_experiment_description(self, experiment_name: str) -> str:
        """
        Get the description of an experiment
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Description of the experiment
        """
        if experiment_name not in self.available_experiments:
            return "Unknown experiment"
        
        return self.available_experiments[experiment_name].get("description", "No description available")
    
    def get_required_settings(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get the required settings for an experiment
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary of required settings
        """
        if experiment_name not in self.available_experiments:
            return {}
        
        return self.available_experiments[experiment_name].get("settings", {})
    
    def get_available_mappings(self) -> List[str]:
        """
        Get the list of available mappings
        
        Returns:
            List of mapping names
        """
        mappings = []
        
        # Get mappings from the mappings directory
        mappings_dir = Path(project_root) / "app" / "instruments" / "arc2custom" / "mappings"
        
        if mappings_dir.exists():
            for mapping_file in mappings_dir.glob("*.toml"):
                mappings.append(mapping_file.stem)
        
        return mappings
    
    def run_experiment(self, experiment_name: str, instrument_id: str, mapping: str, 
                       settings: Dict[str, Any] = None) -> bool:
        """
        Run an experiment with the specified settings
        
        Args:
            experiment_name: Name of the experiment to run
            instrument_id: ID of the instrument to use
            mapping: Name of the mapping to use
            settings: Dictionary of experiment-specific settings
            
        Returns:
            True if the experiment was started successfully, False otherwise
        """
        if experiment_name not in self.available_experiments:
            print(f"Unknown experiment: {experiment_name}")
            return False
        
        # Make sure we're connected to the instrument
        if not self.instrument_manager.is_connected(instrument_id):
            success = self.instrument_manager.connect_to_instrument(instrument_id)
            if not success:
                print(f"Failed to connect to instrument: {instrument_id}")
                return False
        
        # Get the experiment info
        experiment_info = self.available_experiments[experiment_name]
        
        # Check if this is a simulation placeholder
        if experiment_info.get("is_simulation", False):
            # For simulation, we'll use the debug instrument directly
            return self._run_simulation_experiment(experiment_name, instrument_id, mapping, settings)
        
        # Regular experiment handling
        module_path = experiment_info["module"]
        class_name = experiment_info["class"]
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the experiment class
            experiment_class = getattr(module, class_name)
            
            # Get the instrument reference
            instrument = self.instrument_manager.get_instrument(instrument_id)
            
            # Create a session - this will need to be implemented based on sessionmod.py
            from app.instruments.arc2custom import sessionmod
            session = sessionmod.Session()
            
            # If mapping is specified, load it
            if mapping:
                session.loadMapping(mapping)
            
            # Create the experiment instance
            experiment = experiment_class(instrument, experiment_name, session)
            
            # Apply settings if provided
            if settings:
                for key, value in settings.items():
                    if hasattr(experiment.settings, key):
                        setattr(experiment.settings, key, value)
            
            # Run the experiment in a separate process
            # This approach needs to be adapted based on how the experiment should be run
            process = multiprocessing.Process(
                target=self._run_experiment_process,
                args=(experiment,)
            )
            process.start()
            
            return True
        except Exception as e:
            print(f"Error launching experiment {experiment_name}: {e}")
            return False
    
    def _run_experiment_process(self, experiment):
        """
        Run an experiment in a separate process
        
        Args:
            experiment: The experiment instance to run
        """
        try:
            # Check connections
            if hasattr(experiment, "testConnections"):
                proceed = experiment.testConnections()
                if not proceed:
                    return
            
            # Run the experiment
            if hasattr(experiment, "run"):
                experiment.run()
            else:
                print(f"Experiment {experiment.name} does not have a run method")
        except Exception as e:
            print(f"Error running experiment: {e}")
    
    def stop_experiment(self) -> bool:
        """
        Stop the currently running experiment
        
        Returns:
            True if the experiment was stopped successfully, False otherwise
        """
        # This would need to be implemented to stop the running experiment
        # For now, just return True
        print("Stopping experiment (not implemented)")
        return True
    
    def shutdown(self):
        """Shut down the experiment launcher"""
        # Stop the instrument manager
        self.instrument_manager.stop()

    def _run_simulation_experiment(self, experiment_name: str, instrument_id: str, 
                                  mapping: str, settings: Dict[str, Any] = None) -> bool:
        """Run a simulation experiment using the InstDebug capabilities"""
        try:
            # Get the instrument
            instrument = self.instrument_manager.get_instrument(instrument_id)
            
            # Check if it has simulation capabilities
            if hasattr(instrument, "simulate_experiment"):
                # Create default settings if none provided
                if settings is None:
                    settings = {}
                
                # Run the experiment in a separate process
                process = multiprocessing.Process(
                    target=self._run_simulation_process,
                    args=(instrument, experiment_name, settings)
                )
                process.start()
                return True
            else:
                print(f"Instrument {instrument_id} does not support simulation")
                return False
        except Exception as e:
            print(f"Error launching simulation experiment: {e}")
            return False
    
    def _run_simulation_process(self, instrument, experiment_type: str, settings: Dict[str, Any]):
        """Run a simulation experiment in a separate process"""
        try:
            # Set CPU affinity if psutil is available
            try:
                import psutil
                current_process = psutil.Process()
                num_cpus = psutil.cpu_count(logical=True)
                if num_cpus > 1:
                    # Use cores other than core 0 (which is for GUI)
                    instrument_cores = list(range(1, min(4, num_cpus)))
                    current_process.cpu_affinity(instrument_cores)
                    print(f"Experiment process running on CPU cores {instrument_cores}")
            except ImportError:
                pass
            except Exception as e:
                print(f"Error setting CPU affinity: {e}")
            
            # Run the simulation
            print(f"Running simulation experiment: {experiment_type}")
            instrument.simulate_experiment(experiment_type, settings)
        except Exception as e:
            print(f"Error in simulation process: {e}")


class ExperimentLauncherProcess:
    """Runs an experiment launcher in a separate process"""
    
    def __init__(self):
        self.command_queue = Queue()
        self.result_queue = Queue()
        self.process = None
        
    def start(self):
        """Start the experiment launcher process"""
        self.process = multiprocessing.Process(
            target=self._run_launcher,
            args=(self.command_queue, self.result_queue)
        )
        self.process.start()
    
    def stop(self):
        """Stop the experiment launcher process"""
        if self.process and self.process.is_alive():
            self.command_queue.put(("STOP", None))
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
    
    def _run_launcher(self, command_queue, result_queue):
        """Run the experiment launcher, processing commands from the queue"""
        launcher = ExperimentLauncher()
        
        running = True
        while running:
            try:
                if not command_queue.empty():
                    command, args = command_queue.get()
                    
                    if command == "STOP":
                        launcher.shutdown()
                        running = False
                    elif command == "GET_EXPERIMENTS":
                        experiments = launcher.get_available_experiments()
                        result_queue.put(("EXPERIMENTS_RESULT", experiments))
                    elif command == "GET_DESCRIPTION":
                        description = launcher.get_experiment_description(args)
                        result_queue.put(("DESCRIPTION_RESULT", description))
                    elif command == "GET_SETTINGS":
                        settings = launcher.get_required_settings(args)
                        result_queue.put(("SETTINGS_RESULT", settings))
                    elif command == "GET_MAPPINGS":
                        mappings = launcher.get_available_mappings()
                        result_queue.put(("MAPPINGS_RESULT", mappings))
                    elif command == "RUN_EXPERIMENT":
                        exp_name, instrument_id, mapping, settings = args
                        success = launcher.run_experiment(exp_name, instrument_id, mapping, settings)
                        result_queue.put(("RUN_RESULT", success))
                    elif command == "STOP_EXPERIMENT":
                        success = launcher.stop_experiment()
                        result_queue.put(("STOP_RESULT", success))
                
                # Sleep a bit to avoid busy waiting
                time.sleep(0.01)
            
            except Exception as e:
                print(f"Error in experiment launcher process: {e}")
                result_queue.put(("ERROR", str(e)))
    
    def get_available_experiments(self) -> List[str]:
        """Get the list of available experiments"""
        self.command_queue.put(("GET_EXPERIMENTS", None))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "EXPERIMENTS_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error getting experiments: {result}")
                    return []
            time.sleep(0.01)
        
        print("Timeout waiting for experiment list")
        return []
    
    def get_experiment_description(self, experiment_name: str) -> str:
        """Get the description of an experiment"""
        self.command_queue.put(("GET_DESCRIPTION", experiment_name))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "DESCRIPTION_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error getting experiment description: {result}")
                    return "Error: " + str(result)
            time.sleep(0.01)
        
        print("Timeout waiting for experiment description")
        return "Timeout waiting for description"
    
    def get_required_settings(self, experiment_name: str) -> Dict[str, Any]:
        """Get the required settings for an experiment"""
        self.command_queue.put(("GET_SETTINGS", experiment_name))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "SETTINGS_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error getting experiment settings: {result}")
                    return {}
            time.sleep(0.01)
        
        print("Timeout waiting for experiment settings")
        return {}
    
    def get_available_mappings(self) -> List[str]:
        """Get the list of available mappings"""
        self.command_queue.put(("GET_MAPPINGS", None))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "MAPPINGS_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error getting mappings: {result}")
                    return []
            time.sleep(0.01)
        
        print("Timeout waiting for mappings list")
        return []
    
    def run_experiment(self, experiment_name: str, instrument_id: str, mapping: str,
                      settings: Dict[str, Any] = None) -> bool:
        """Run an experiment"""
        self.command_queue.put(("RUN_EXPERIMENT", (experiment_name, instrument_id, mapping, settings)))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "RUN_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error running experiment: {result}")
                    return False
            time.sleep(0.01)
        
        print("Timeout waiting for experiment run result")
        return False
    
    def stop_experiment(self) -> bool:
        """Stop the currently running experiment"""
        self.command_queue.put(("STOP_EXPERIMENT", None))
        
        # Wait for the result (with timeout)
        for _ in range(100):  # 1 second timeout
            if not self.result_queue.empty():
                command, result = self.result_queue.get()
                if command == "STOP_RESULT":
                    return result
                elif command == "ERROR":
                    print(f"Error stopping experiment: {result}")
                    return False
            time.sleep(0.01)
        
        print("Timeout waiting for experiment stop result")
        return False


# Example usage
if __name__ == "__main__":
    # Start the launcher in a separate process
    launcher_process = ExperimentLauncherProcess()
    launcher_process.start()
    
    # Get available experiments
    experiments = launcher_process.get_available_experiments()
    print(f"Available experiments: {experiments}")
    
    # Get available mappings
    mappings = launcher_process.get_available_mappings()
    print(f"Available mappings: {mappings}")
    
    # Get experiment description
    description = launcher_process.get_experiment_description("Activation Pattern")
    print(f"Activation Pattern description: {description}")
    
    # Run an experiment
    if mappings:
        success = launcher_process.run_experiment(
            "Activation Pattern",
            "InstDebug (Simulation)",
            mappings[0]
        )
        print(f"Started Activation Pattern experiment: {success}")
    
    # Stop the launcher process
    launcher_process.stop() 