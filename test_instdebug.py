#!/usr/bin/env python3
"""
Test script for running the InstDebug simulation with the GUI
"""

import sys
import os
import multiprocessing
from multiprocessing import Queue
import time
import argparse

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from app.instruments.debug_instrument import DebugInstrument
from app.gui.elex_main import ElexMainWindow
from PyQt6.QtWidgets import QApplication


def test_instdebug_direct():
    """
    Test the InstDebug simulation directly without the GUI
    """
    from app.instruments.instdebug import InstDebug
    
    print("Creating InstDebug instance...")
    inst = InstDebug(serial_number="TEST001", simulation_mode="normal")
    
    # Connect to the instrument
    print("Connecting to instrument...")
    inst.connect()
    
    # Set up a simple channel mapper
    class SimpleChannelMapper:
        def __init__(self, name, channels):
            self.name = name
            self.nwords = channels
            self.nbits = channels
    
    mapper = SimpleChannelMapper("test_mapping", 16)
    inst.set_channel_mapper(mapper)
    
    # Set up a result queue
    result_queue = Queue()
    inst.set_result_queue(result_queue)
    
    # Create a consumer process that will display results
    def result_consumer(queue):
        print("Result consumer started")
        while True:
            try:
                result = queue.get(timeout=1)
                if result.get("type") == "status_update":
                    progress = result.get("progress", 0)
                    status = result.get("status", "unknown")
                    message = result.get("message", "")
                    print(f"Status: {status} | Progress: {progress}% | Message: {message}")
                elif result.get("type") == "data_update":
                    print(f"Received data update: {len(result.get('data', {}).get('timestamps', []))} data points")
            except:
                # Check if we should continue
                if not inst.is_connected():
                    break
    
    # Start the consumer
    consumer_process = multiprocessing.Process(target=result_consumer, args=(result_queue,))
    consumer_process.start()
    
    # Run a test experiment
    try:
        print("Running IV measurement experiment...")
        params = {
            "start_voltage": -1.0,
            "end_voltage": 1.0,
            "voltage_steps": 50,
            "channels": [0, 1, 2, 3]
        }
        inst.simulate_experiment("IVMeasurement", params)
        
        # Wait for a moment to let the results be processed
        time.sleep(1)
        
        print("Running pulse measurement experiment...")
        params = {
            "pulse_amplitude": 0.5,
            "pulse_width": 5,
            "num_pulses": 3,
            "channels": [0, 1]
        }
        inst.simulate_experiment("PulseMeasurement", params)
        
        # Wait to allow consumer to process results
        time.sleep(1)
    finally:
        # Disconnect the instrument
        print("Disconnecting from instrument...")
        inst.disconnect()
        
        # Wait for consumer to exit
        consumer_process.join(timeout=5)
        if consumer_process.is_alive():
            consumer_process.terminate()
    
    print("Test completed successfully")


def run_gui_with_instdebug():
    """
    Run the GUI with InstDebug as the default instrument
    """
    try:
        # Try to set process affinity if psutil is available
        try:
            import psutil
            current_process = psutil.Process()
            
            # Get the number of available CPUs
            num_cpus = psutil.cpu_count(logical=True)
            
            if num_cpus > 1:
                # Run the GUI on the first CPU core
                current_process.cpu_affinity([0])
                print(f"GUI process running on CPU core 0")
        except ImportError:
            # psutil not available, can't set affinity
            pass
        except Exception as e:
            print(f"Error setting CPU affinity: {e}")
        
        app = QApplication(sys.argv)
        window = ElexMainWindow()
        
        # Make sure we have a valid experiment launcher
        if not hasattr(window, 'experiment_launcher') or window.experiment_launcher is None:
            print("Error: GUI's experiment launcher not initialized")
            window.show()
            sys.exit(app.exec())
            return
            
        # Scan for instruments if the method exists
        if hasattr(window, 'scan_instruments'):
            window.scan_instruments()
        
        # First, ensure we connect to the debug instrument
        connected = False
        if hasattr(window, 'instrument_combo'):
            # Find the index of the debug instrument
            debug_index = window.instrument_combo.findText("InstDebug (Simulation)")
            if debug_index >= 0:
                window.instrument_combo.setCurrentIndex(debug_index)
                window.current_instrument = "InstDebug (Simulation)"
                connected = True
                print("Connected to InstDebug simulator in GUI")
        
        # Show the window
        window.show()
        
        # If successfully connected, automatically launch a test experiment after a delay
        if connected:
            # Wait a brief moment for the GUI to initialize
            from PyQt6.QtCore import QTimer
            
            def start_test_experiment():
                """Start a test experiment after the GUI is fully loaded"""
                try:
                    # First show the experiments page
                    if hasattr(window, 'stacked_widget') and window.stacked_widget.count() > 2:
                        window.stacked_widget.setCurrentIndex(2)  # Go to experiments page
                    
                    # Get available experiments
                    experiments = window.experiment_launcher.get_available_experiments()
                    if not experiments:
                        print("Error: No experiments available")
                        return
                    
                    # Select IVMeasurement if available, otherwise use the first experiment
                    test_experiment = None
                    for exp in experiments:
                        if exp.lower() == "ivmeasurement":
                            test_experiment = exp
                            break
                    
                    if test_experiment is None and experiments:
                        test_experiment = experiments[0]
                    
                    if test_experiment:
                        print(f"Starting test experiment: {test_experiment}")
                        
                        # Set the current experiment
                        window.current_experiment = test_experiment
                        
                        # Create test settings
                        settings = {
                            "start_voltage": -1.0,
                            "end_voltage": 1.0,
                            "voltage_steps": 100,
                            "channels": [0, 1, 2, 3]
                        }
                        
                        # Set default mapping
                        window.current_mapping = "default"
                        
                        # Start the experiment
                        if hasattr(window, 'run_experiment_in_separate_process'):
                            window.run_experiment_in_separate_process()
                        else:
                            window.experiment_launcher.run_experiment(
                                test_experiment, 
                                "InstDebug (Simulation)", 
                                "default", 
                                settings
                            )
                            # Switch to results view
                            if hasattr(window, 'stacked_widget') and window.stacked_widget.count() > 3:
                                window.stacked_widget.setCurrentIndex(3)  # Results page
                                
                            print("Experiment started. Results should be displayed in the GUI.")
                    else:
                        print("No suitable experiment found for testing")
                except Exception as e:
                    print(f"Error running test experiment: {e}")
            
            # Start the experiment after a delay
            QTimer.singleShot(2000, start_test_experiment)
        
        # Run the application
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error running GUI: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test InstDebug simulation")
    parser.add_argument("--mode", choices=["gui", "direct"], default="gui",
                      help="Test mode: 'gui' to run with GUI, 'direct' to test directly")
    args = parser.parse_args()
    
    if args.mode == "direct":
        test_instdebug_direct()
    else:
        run_gui_with_instdebug() 