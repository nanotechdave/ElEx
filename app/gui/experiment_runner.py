import sys
import os
import time
import multiprocessing
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox, QHBoxLayout
from PyQt6.QtCore import Qt

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from app.instruments.instrument_factory import InstrumentFactory


class ExperimentRunner(QWidget):
    """
    Widget for running experiments
    """
    
    def __init__(self, experiment_launcher, parent=None):
        super().__init__(parent)
        
        self.experiment_launcher = experiment_launcher
        self.selected_experiment = None
        self.experiment_settings = {}
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        
        # Experiment selection
        experiment_group = QGroupBox("Experiment")
        experiment_layout = QVBoxLayout(experiment_group)
        
        # Description label
        self.description_label = QLabel("No experiment selected")
        self.description_label.setWordWrap(True)
        experiment_layout.addWidget(self.description_label)
        
        # Status label
        self.status_label = QLabel("Ready")
        experiment_layout.addWidget(self.status_label)
        
        # Add button row
        button_layout = QHBoxLayout()
        
        # Run button
        self.run_button = QPushButton("Run Experiment")
        self.run_button.clicked.connect(self.run_experiment)
        self.run_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.clicked.connect(self.stop_experiment)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        experiment_layout.addLayout(button_layout)
        
        # Add to main layout
        layout.addWidget(experiment_group)
    
    def set_selected_experiment(self, experiment_name, description="No description"):
        """Set the selected experiment"""
        self.selected_experiment = experiment_name
        self.description_label.setText(description)
        self.run_button.setEnabled(True)
    
    def set_experiment_settings(self, settings):
        """Set the experiment settings"""
        self.experiment_settings = settings
    
    def run_experiment(self):
        """Run the selected experiment"""
        if not self.selected_experiment:
            self.status_label.setText("No experiment selected")
            return
        
        # Update UI
        self.status_label.setText(f"Running {self.selected_experiment}...")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Run the experiment
        success = self.experiment_launcher.run_experiment(
            self.selected_experiment,
            "InstDebug (Simulation)",  # Default to simulation instrument
            "default",  # Default mapping
            self.experiment_settings
        )
        
        if not success:
            self.status_label.setText(f"Failed to start {self.selected_experiment}")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_experiment(self):
        """Stop the running experiment"""
        success = self.experiment_launcher.stop_experiment()
        
        # Update UI
        if success:
            self.status_label.setText("Experiment stopped")
        else:
            self.status_label.setText("Failed to stop experiment")
        
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)


def run_experiment_process(experiment_type, settings, instrument_name, mapping_name, results_queue=None):
    """
    Runs an experiment based on the given parameters.
    
    Args:
        experiment_type (str): Type of experiment to run
        settings (dict): Dictionary of experiment settings
        instrument_name (str): Name of the instrument to use
        mapping_name (str): Name of the mapping to use
        results_queue (Queue, optional): Queue to send results to the GUI
    """
    try:
        # Try to set process CPU affinity if psutil is available
        # This helps ensure instrument control code runs on separate cores from GUI
        try:
            import psutil
            current_process = psutil.Process()
            
            # Get the number of available CPUs
            num_cpus = psutil.cpu_count(logical=True)
            
            if num_cpus > 1:
                # Run on cores other than the GUI core (0)
                instrument_cores = list(range(1, min(4, num_cpus)))
                current_process.cpu_affinity(instrument_cores)
                print(f"Experiment process running on CPU cores {instrument_cores}")
        except ImportError:
            # psutil not available, can't set affinity
            pass
        except Exception as e:
            print(f"Error setting CPU affinity: {e}")
            
        # Create the instrument factory
        instrument_factory = InstrumentFactory()
        
        # Get the instrument
        instrument = instrument_factory.get_instrument(instrument_name, mapping_name)
        
        # If we have a results queue from the GUI, use it
        if results_queue:
            instrument.set_result_queue(results_queue)
        
        # Create appropriate parameters based on experiment type
        experiment_params = get_default_params_for_experiment(experiment_type)
        
        # Update with any user-provided settings
        if settings:
            experiment_params.update(settings)
        
        print(f"Running {experiment_type} with parameters: {experiment_params}")
        
        # Run the experiment
        if hasattr(instrument, "simulate_experiment"):
            instrument.simulate_experiment(experiment_type, experiment_params)
        else:
            print(f"Instrument {instrument_name} does not support simulation")
            raise ValueError(f"Unsupported instrument: {instrument_name}")
        
    except Exception as e:
        print(f"Error in experiment process: {e}")
        if results_queue:
            # Send error to the GUI
            results_queue.put({
                "experiment_type": experiment_type,
                "status": "error",
                "error_message": str(e),
                "timestamp": time.time()
            })


def get_default_params_for_experiment(experiment_type):
    """
    Returns default parameters for different experiment types
    
    Args:
        experiment_type (str): Type of experiment
        
    Returns:
        dict: Default parameters for the experiment
    """
    # Default parameters for each experiment type
    if experiment_type == "IVMeasurement":
        return {
            "start_voltage": -1.0,
            "end_voltage": 1.0,
            "voltage_steps": 100,
            "channels": [0, 1, 2, 3]
        }
    elif experiment_type == "PulseMeasurement":
        return {
            "pulse_amplitude": 1.0,
            "pulse_width": 10,  # ms
            "num_pulses": 5,
            "channels": [0, 1, 2, 3]
        }
    elif experiment_type == "NoiseMeasurement":
        return {
            "duration": 10,  # seconds
            "sample_rate": 100,  # Hz
            "bias_voltage": 0.1,  # V
            "channels": [0, 1, 2, 3]
        }
    elif experiment_type == "MemoryCapacity":
        return {
            "input_length": 100,
            "warmup_time": 20,
            "bias_voltage": 0.5,
            "channels": [0, 1, 2, 3]
        }
    elif experiment_type == "ActivationPattern":
        return {
            "num_patterns": 10,
            "pattern_duration": 5,  # seconds
            "activation_voltage": 1.0,
            "channels": [0, 1, 2, 3]
        }
    elif experiment_type == "ConductivityMatrix":
        return {
            "voltage": 0.1,
            "matrix_size": 8,  # 8x8 matrix
            "channels": list(range(8))
        }
    elif experiment_type == "ReservoirComputing":
        return {
            "input_length": 100,
            "reservoir_size": 8,
            "bias_voltage": 0.5,
            "channels": list(range(8))
        }
    elif experiment_type == "Tomography":
        return {
            "num_angles": 12,
            "resolution": 32,
            "voltage_amplitude": 0.5,
            "channels": list(range(16))
        }
    elif experiment_type == "TurnOn":
        return {
            "voltage": 1.0,
            "duration": 10,  # seconds
            "channels": [0, 1, 2, 3]
        }
    else:
        # Generic default parameters
        return {
            "steps": 100,
            "channels": [0, 1, 2, 3]
        }

def setup_visualization(experiment_type, settings):
    """
    Set up visualization for the experiment results.
    For simulation, we'll just print to console, but this could be expanded
    to create real-time plots.
    
    Args:
        experiment_type: The type of experiment
        settings: Dictionary of experiment settings
    """
    print(f"\n{'='*80}")
    print(f"Starting {experiment_type} with the following settings:")
    
    # Print settings in a formatted way
    for key, value in settings.items():
        # Make lists more readable
        if isinstance(value, list) and len(value) > 10:
            print(f"  {key}: [{value[0]}, {value[1]}, ... {value[-2]}, {value[-1]}] (length: {len(value)})")
        else:
            print(f"  {key}: {value}")
    
    print(f"{'='*80}\n")
    
    # Different experiment types might need different visualization setups
    if experiment_type == "IVMeasurement":
        print("Setting up IV curve visualization...")
    elif experiment_type == "ConductivityMatrix":
        print("Setting up conductivity matrix visualization...")
    elif experiment_type == "PulseMeasurement":
        print("Setting up pulse response visualization...")
    # Add more experiment types as needed 