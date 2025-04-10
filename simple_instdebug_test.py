#!/usr/bin/env python3
"""
Simple test script for the InstDebug simulation without the GUI
This avoids the dependencies on PyQt and other GUI components
"""

import sys
import os
import multiprocessing
from multiprocessing import Queue
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import the InstDebug class directly
from app.instruments.instdebug import InstDebug


class ResultVisualizer:
    """Class to visualize results from the InstDebug simulation"""
    
    def __init__(self, max_points=1000):
        """
        Initialize the visualizer
        
        Args:
            max_points: Maximum number of points to display
        """
        self.max_points = max_points
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.voltage_ax = self.axes[0]
        self.current_ax = self.axes[1]
        
        # Set up empty lines for each channel (up to 4 channels)
        self.voltage_lines = []
        self.current_lines = []
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, color in enumerate(colors[:4]):
            voltage_line, = self.voltage_ax.plot([], [], color=color, label=f'Ch {i}')
            current_line, = self.current_ax.plot([], [], color=color, label=f'Ch {i}')
            self.voltage_lines.append(voltage_line)
            self.current_lines.append(current_line)
        
        # Set up axes
        self.voltage_ax.set_title('Voltage vs Time')
        self.voltage_ax.set_xlabel('Time (s)')
        self.voltage_ax.set_ylabel('Voltage (V)')
        self.voltage_ax.grid(True)
        self.voltage_ax.legend()
        
        self.current_ax.set_title('Current vs Time')
        self.current_ax.set_xlabel('Time (s)')
        self.current_ax.set_ylabel('Current (A)')
        self.current_ax.grid(True)
        self.current_ax.legend()
        
        self.fig.tight_layout()
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.01, "Status: Ready", ha='center')
        
        # Data storage
        self.time_data = {}  # Dict of deques for each channel
        self.voltage_data = {}  # Dict of deques for each channel
        self.current_data = {}  # Dict of deques for each channel
        
        # Animation control
        self.anim = None
        self.experiment_type = None
        self.paused = False
    
    def start_animation(self):
        """Start the animation to update the plot"""
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        plt.show(block=False)
    
    def process_result(self, result):
        """Process a result from the queue"""
        if result.get("type") == "status_update":
            status = result.get("status", "unknown")
            progress = result.get("progress", 0)
            message = result.get("message", "")
            self.update_status(f"Status: {status} | Progress: {progress}% | {message}")
            
            # Store experiment type
            self.experiment_type = result.get("experiment_type", "unknown")
            
        elif result.get("type") == "data_update":
            data = result.get("data", {})
            self.update_data(data)
    
    def update_status(self, status_text):
        """Update the status text"""
        self.status_text.set_text(status_text)
    
    def update_data(self, data):
        """Update the data with new values"""
        timestamps = data.get("timestamps", [])
        voltages = data.get("voltages", {})
        currents = data.get("currents", {})
        
        if not timestamps:
            return
        
        # Get the base time if this is the first data point
        if not hasattr(self, 'base_time'):
            self.base_time = timestamps[0]
        
        # Process each channel
        for ch, voltage_values in voltages.items():
            # Initialize deques if this is a new channel
            if ch not in self.time_data:
                self.time_data[ch] = deque(maxlen=self.max_points)
                self.voltage_data[ch] = deque(maxlen=self.max_points)
                self.current_data[ch] = deque(maxlen=self.max_points)
            
            # Add new data points
            for i, t in enumerate(timestamps):
                if i < len(voltage_values):
                    self.time_data[ch].append(t - self.base_time)
                    self.voltage_data[ch].append(voltage_values[i])
                    
                    # Make sure currents exist for this channel and index
                    if ch in currents and i < len(currents[ch]):
                        self.current_data[ch].append(currents[ch][i])
                    else:
                        self.current_data[ch].append(0)
    
    def update_plot(self, frame):
        """Update the plot with the latest data"""
        if self.paused:
            return
        
        # Update voltage lines
        for i, line in enumerate(self.voltage_lines):
            ch = i  # Match channel to line index
            if ch in self.time_data and self.time_data[ch]:
                line.set_data(list(self.time_data[ch]), list(self.voltage_data[ch]))
        
        # Update current lines
        for i, line in enumerate(self.current_lines):
            ch = i  # Match channel to line index
            if ch in self.time_data and self.time_data[ch]:
                line.set_data(list(self.time_data[ch]), list(self.current_data[ch]))
        
        # Adjust axis limits
        all_times = []
        all_voltages = []
        all_currents = []
        
        for ch in self.time_data:
            if self.time_data[ch]:
                all_times.extend(self.time_data[ch])
                all_voltages.extend(self.voltage_data[ch])
                all_currents.extend(self.current_data[ch])
        
        if all_times:
            # Update voltage axis
            self.voltage_ax.set_xlim(min(all_times), max(all_times) + 0.1)
            if all_voltages:
                v_min = min(all_voltages)
                v_max = max(all_voltages)
                v_range = max(v_max - v_min, 0.1)  # Ensure a minimum range
                self.voltage_ax.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)
            
            # Update current axis
            self.current_ax.set_xlim(min(all_times), max(all_times) + 0.1)
            if all_currents:
                i_min = min(all_currents)
                i_max = max(all_currents)
                i_range = max(i_max - i_min, 1e-9)  # Ensure a minimum range
                self.current_ax.set_ylim(i_min - 0.1 * i_range, i_max + 0.1 * i_range)
        
        self.fig.canvas.draw_idle()
    
    def toggle_pause(self):
        """Toggle pausing of the animation"""
        self.paused = not self.paused
        status = "Paused" if self.paused else "Running"
        print(f"Animation {status}")


def test_iv_measurement():
    """Run an IV measurement test"""
    print("\n=== Testing IV Measurement ===")
    
    # Create InstDebug instance
    inst = InstDebug(serial_number="TEST001", simulation_mode="normal")
    inst.connect()
    
    # Set up a simple channel mapper
    class SimpleChannelMapper:
        def __init__(self, name, channels):
            self.name = name
            self.nwords = channels
            self.nbits = channels
    
    mapper = SimpleChannelMapper("test_mapping", 16)
    inst.set_channel_mapper(mapper)
    
    # Set up result queue and visualizer
    result_queue = Queue()
    inst.set_result_queue(result_queue)
    visualizer = ResultVisualizer()
    visualizer.start_animation()
    
    # Run experiment in a separate thread to avoid blocking the UI
    def run_experiment():
        try:
            params = {
                "start_voltage": -1.0,
                "end_voltage": 1.0,
                "voltage_steps": 100,
                "channels": [0, 1, 2, 3]
            }
            inst.simulate_experiment("IVMeasurement", params)
        finally:
            inst.disconnect()
    
    import threading
    exp_thread = threading.Thread(target=run_experiment)
    exp_thread.daemon = True
    exp_thread.start()
    
    # Process results from the queue and update the visualizer
    try:
        while exp_thread.is_alive() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.1)
                visualizer.process_result(result)
            except:
                pass
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Keep the plot window open until user closes it
    print("Experiment complete. Close the plot window to exit.")
    plt.show()


def test_pulse_measurement():
    """Run a pulse measurement test"""
    print("\n=== Testing Pulse Measurement ===")
    
    # Create InstDebug instance
    inst = InstDebug(serial_number="TEST001", simulation_mode="normal")
    inst.connect()
    
    # Set up a simple channel mapper
    class SimpleChannelMapper:
        def __init__(self, name, channels):
            self.name = name
            self.nwords = channels
            self.nbits = channels
    
    mapper = SimpleChannelMapper("test_mapping", 16)
    inst.set_channel_mapper(mapper)
    
    # Set up result queue and visualizer
    result_queue = Queue()
    inst.set_result_queue(result_queue)
    visualizer = ResultVisualizer()
    visualizer.start_animation()
    
    # Run experiment in a separate thread to avoid blocking the UI
    def run_experiment():
        try:
            params = {
                "pulse_amplitude": 1.0,
                "pulse_width": 10,  # ms
                "num_pulses": 5,
                "channels": [0, 1, 2, 3]
            }
            inst.simulate_experiment("PulseMeasurement", params)
        finally:
            inst.disconnect()
    
    import threading
    exp_thread = threading.Thread(target=run_experiment)
    exp_thread.daemon = True
    exp_thread.start()
    
    # Process results from the queue and update the visualizer
    try:
        while exp_thread.is_alive() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.1)
                visualizer.process_result(result)
            except:
                pass
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Keep the plot window open until user closes it
    print("Experiment complete. Close the plot window to exit.")
    plt.show()


def test_noise_measurement():
    """Run a noise measurement test"""
    print("\n=== Testing Noise Measurement ===")
    
    # Create InstDebug instance
    inst = InstDebug(serial_number="TEST001", simulation_mode="normal")
    inst.connect()
    
    # Set up a simple channel mapper
    class SimpleChannelMapper:
        def __init__(self, name, channels):
            self.name = name
            self.nwords = channels
            self.nbits = channels
    
    mapper = SimpleChannelMapper("test_mapping", 16)
    inst.set_channel_mapper(mapper)
    
    # Set up result queue and visualizer
    result_queue = Queue()
    inst.set_result_queue(result_queue)
    visualizer = ResultVisualizer()
    visualizer.start_animation()
    
    # Run experiment in a separate thread to avoid blocking the UI
    def run_experiment():
        try:
            params = {
                "duration": 5,  # seconds
                "sample_rate": 100,  # Hz
                "bias_voltage": 0.1,  # V
                "channels": [0, 1]
            }
            inst.simulate_experiment("NoiseMeasurement", params)
        finally:
            inst.disconnect()
    
    import threading
    exp_thread = threading.Thread(target=run_experiment)
    exp_thread.daemon = True
    exp_thread.start()
    
    # Process results from the queue and update the visualizer
    try:
        while exp_thread.is_alive() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.1)
                visualizer.process_result(result)
            except:
                pass
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Keep the plot window open until user closes it
    print("Experiment complete. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test InstDebug simulation functionality")
    parser.add_argument("--test", choices=["iv", "pulse", "noise"], default="iv",
                      help="Test to run: 'iv' for IV measurement, 'pulse' for pulse measurement, 'noise' for noise measurement")
    
    args = parser.parse_args()
    
    if args.test == "iv":
        test_iv_measurement()
    elif args.test == "pulse":
        test_pulse_measurement()
    elif args.test == "noise":
        test_noise_measurement() 