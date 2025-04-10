#!/usr/bin/env python3
"""
ElEx - Electronic Experiments
Main application entry point
"""

import sys
import os
import multiprocessing
from multiprocessing import Queue
import time
import signal

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from app.instruments.instrument_manager import InstrumentManagerProcess
from app.experiments.experiment_launcher import ExperimentLauncherProcess
from app.gui.elex_main import ElexMainWindow
from PyQt6.QtWidgets import QApplication

# Try to import the psutil library for CPU affinity management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil library not available. CPU affinity control disabled.")


def set_process_affinity(process_id, cpu_ids):
    """
    Set CPU affinity for a process if psutil is available
    
    Args:
        process_id: ID of the process to set affinity for
        cpu_ids: List of CPU IDs to bind the process to
    """
    if not HAS_PSUTIL:
        return
        
    try:
        process = psutil.Process(process_id)
        process.cpu_affinity(cpu_ids)
        print(f"Process {process_id} affinity set to CPUs {cpu_ids}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")


def run_gui(command_queue, result_queue):
    """
    Run the GUI in a separate process
    
    Args:
        command_queue: Queue for sending commands to the GUI
        result_queue: Queue for receiving results from the GUI
    """
    # If psutil is available, set GUI to run on specific cores
    if HAS_PSUTIL:
        # Get the current process
        current_process = psutil.Process()
        
        # Get the number of available CPUs
        num_cpus = psutil.cpu_count(logical=True)
        
        if num_cpus > 1:
            # Run the GUI on the first CPU core
            current_process.cpu_affinity([0])
            print(f"GUI process running on CPU core 0")
    
    app = QApplication(sys.argv)
    
    # Create the main window
    window = ElexMainWindow()
    
    # Set up signal handling to process commands from the main process
    timer = window.startTimer(100)  # Check for commands every 100ms
    
    def timerEvent(event):
        # Process commands from the main process
        if not command_queue.empty():
            command, args = command_queue.get()
            if command == "CLOSE":
                window.close()
            elif command == "SET_STATUS":
                window.status_bar.showMessage(args)
            # Add more commands as needed
    
    # Hook up the timer event
    window.timerEvent = timerEvent
    
    # Show the window
    window.show()
    
    # Run the application
    app.exec()


def main():
    """
    Main application entry point
    """
    print("Starting ElEx - Electronic Experiments")
    
    # Set up communication queues
    gui_command_queue = Queue()
    gui_result_queue = Queue()
    
    # Create the processes
    gui_process = multiprocessing.Process(
        target=run_gui,
        args=(gui_command_queue, gui_result_queue)
    )
    
    instrument_manager = InstrumentManagerProcess()
    experiment_launcher = ExperimentLauncherProcess()
    
    try:
        # Start the processes
        gui_process.start()
        instrument_manager.start()
        experiment_launcher.start()
        
        # If psutil is available, assign CPU cores for the instrument control processes
        if HAS_PSUTIL:
            # Get the number of available CPUs
            num_cpus = psutil.cpu_count(logical=True)
            
            if num_cpus > 1:
                # Set instrument control processes to run on cores other than the first one
                # The first core (0) is reserved for the GUI
                instrument_cores = list(range(1, min(4, num_cpus)))
                
                # Set affinity for instrument manager process
                set_process_affinity(instrument_manager.process.pid, instrument_cores)
                
                # Set affinity for experiment launcher process
                set_process_affinity(experiment_launcher.process.pid, instrument_cores)
                
                print(f"Instrument control processes running on CPU cores {instrument_cores}")
        
        # Wait for the GUI to exit
        while gui_process.is_alive():
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    
    finally:
        # Clean up
        print("Shutting down processes...")
        
        # Stop the GUI process
        if gui_process.is_alive():
            gui_command_queue.put(("CLOSE", None))
            gui_process.join(timeout=5)
            if gui_process.is_alive():
                gui_process.terminate()
        
        # Stop the other processes
        instrument_manager.stop()
        experiment_launcher.stop()
        
        print("ElEx shutdown complete")


if __name__ == "__main__":
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore Ctrl+C in main process
    
    # Run the application
    main()
