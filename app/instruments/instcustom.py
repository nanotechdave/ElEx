import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import multiprocessing
import queue
import os
import sys

# Add the project root to Python path to help with imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import InstDebug
try:
    from app.instruments.instdebug import InstDebug
except ImportError:
    # Try relative import as fallback
    try:
        from .instdebug import InstDebug
    except ImportError:
        print("Error: Unable to import InstDebug. Make sure it's available in the same directory.")
        # Define a minimal InstDebug class to avoid errors
        class InstDebug:
            def __init__(self, serial_number="DEBUG001", simulation_mode="normal"):
                pass

class InstCustom(InstDebug):
    """
    A custom measurement instrument that extends InstDebug with additional
    functionality to simulate experimental routines from arc2custom.
    """
    
    def __init__(self, serial_number: str = "CUSTOM001", simulation_mode: str = "normal"):
        super().__init__(serial_number, simulation_mode)
        
        # Additional parameters for simulation
        self.channels = 32  # Extend to 32 channels to match ARC2Custom
        self.resistances = {ch: 1000 * (ch + 1) for ch in range(self.channels)}
        self.applied_voltage = {ch: 0.0 for ch in range(self.channels)}
        self.applied_current = {ch: 0.0 for ch in range(self.channels)}
        self.source_mode = {ch: "voltage" for ch in range(self.channels)}
        
        # Memristor model parameters
        self.memristor_state = {ch: 0.5 for ch in range(self.channels)}  # State variable (0 to 1)
        self.memristor_resistance_on = 1e3  # Resistance in ON state (ohms)
        self.memristor_resistance_off = 1e6  # Resistance in OFF state (ohms)
        
        # Result queues for multiprocessing communication
        self.result_queue = multiprocessing.Queue()
        self.command_queue = multiprocessing.Queue()
        
        # Experiment process
        self.experiment_process = None
    
    def set_result_queue(self, queue: multiprocessing.Queue) -> None:
        """
        Set an external result queue for sending experiment updates.
        
        Args:
            queue: The multiprocessing queue to send results to
        """
        self.result_queue = queue
    
    def set_channel_mapper(self, channel_mapper) -> None:
        """
        Set the channel mapper for this instrument.
        
        Args:
            channel_mapper: A ChannelMapper object that defines channel mappings
        """
        self.channel_mapper = channel_mapper
        print(f"Channel mapper set: {channel_mapper.name}")
        
        # Update channels based on mapper configuration
        self.channels = max(channel_mapper.nwords, channel_mapper.nbits)
        
        # Expand arrays if needed
        if self.channels > len(self.memristor_state):
            for ch in range(len(self.memristor_state), self.channels):
                self.memristor_state[ch] = 0.5
                self.resistances[ch] = self._calculate_memristor_resistance(ch)
                self.applied_voltage[ch] = 0.0
                self.applied_current[ch] = 0.0
                self.source_mode[ch] = "voltage"
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the instrument"""
        info = super().get_info()
        info["name"] = "InstCustom"
        info["channels"] = self.channels
        return info
    
    def _calculate_memristor_resistance(self, channel: int) -> float:
        """Calculate the resistance of a memristor based on its state"""
        state = self.memristor_state[channel]
        # Logarithmic interpolation between ON and OFF states
        log_r_on = np.log10(self.memristor_resistance_on)
        log_r_off = np.log10(self.memristor_resistance_off)
        log_r = log_r_off - state * (log_r_off - log_r_on)
        return 10**log_r
    
    def update_memristor_state(self, channel: int, voltage: float, dt: float = 0.001) -> None:
        """Update the state of a memristor based on applied voltage"""
        if abs(voltage) < 0.1:
            # No significant change at low voltages
            return
        
        state = self.memristor_state[channel]
        
        # Simple memristor model: state changes faster at higher voltages
        # and saturates at boundaries
        v_thresh = 0.5  # Threshold voltage
        
        if voltage > v_thresh:
            # SET operation (increasing conductance)
            dstate = 0.1 * (voltage - v_thresh) * dt * (1 - state)
            self.memristor_state[channel] = min(1.0, state + dstate)
        elif voltage < -v_thresh:
            # RESET operation (decreasing conductance)
            dstate = 0.1 * (-voltage - v_thresh) * dt * state
            self.memristor_state[channel] = max(0.0, state - dstate)
    
    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set the voltage for a channel and update memristor state"""
        super().set_voltage(channel, voltage)
        
        # Update memristor state
        if 0 <= channel < self.channels:
            self.update_memristor_state(channel, voltage)
            
            # Update resistance based on new state
            self.resistances[channel] = self._calculate_memristor_resistance(channel)
            
            # Recalculate current
            if self.source_mode[channel] == "voltage":
                ideal_current = voltage / self.resistances[channel]
                if abs(ideal_current) > self.current_compliance:
                    self.applied_current[channel] = self.current_compliance if ideal_current > 0 else -self.current_compliance
                else:
                    self.applied_current[channel] = ideal_current
    
    def simulate_experiment(self, experiment_type: str, params: dict) -> None:
        """
        Simulate an experiment and send updates to the result queue.
        This version runs directly in the current process without spawning a new process,
        which avoids serialization issues.
        
        Args:
            experiment_type: Type of experiment to simulate
            params: Dictionary of experiment parameters
        """
        print(f"Starting {experiment_type} simulation...")
        
        # Initialize results dictionary
        results = {
            "experiment_type": experiment_type,
            "status": "running",
            "progress": 0,
            "data": {},
            "timestamp": time.time(),
            "params": params
        }
        
        # Send initial status
        self.result_queue.put(results.copy())
        
        try:
            # Simulate different experiment types
            if experiment_type == "IVMeasurement":
                self._simulate_iv_measurement(params, results, self.result_queue)
            elif experiment_type == "ConductivityMatrix":
                self._simulate_conductivity_matrix(params, results, self.result_queue)
            elif experiment_type == "NoiseMeasurement":
                self._simulate_noise_measurement(params, results, self.result_queue)
            elif experiment_type == "PulseMeasurement":
                self._simulate_pulse_measurement(params, results, self.result_queue)
            elif experiment_type == "MemoryCapacity":
                self._simulate_memory_capacity(params, results, self.result_queue)
            elif experiment_type == "ReservoirComputing":
                self._simulate_reservoir_computing(params, results, self.result_queue)
            elif experiment_type == "Tomography":
                self._simulate_tomography(params, results, self.result_queue)
            elif experiment_type == "TurnOn":
                self._simulate_turn_on(params, results, self.result_queue)
            else:
                results["status"] = "error"
                results["error_message"] = f"Unknown experiment type: {experiment_type}"
                self.result_queue.put(results.copy())
                return
            
            # Mark experiment as completed
            results["status"] = "completed"
            results["progress"] = 100
            self.result_queue.put(results.copy())
            print(f"Simulation completed: {experiment_type}")
            
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            results = {
                "experiment_type": experiment_type,
                "status": "error",
                "error_message": str(e),
                "timestamp": time.time()
            }
            self.result_queue.put(results)
    
    def _simulate_iv_measurement(self, params: dict, results: dict, result_queue: multiprocessing.Queue) -> None:
        """Simulate IV measurement experiment"""
        mask_to_bias = params.get("mask_to_bias", [])
        mask_to_gnd = params.get("mask_to_gnd", [])
        mask_to_read_v = params.get("mask_to_read_v", [])
        mask_to_read_i = params.get("mask_to_read_i", [])
        
        start_voltage = params.get("start_voltage", 0.01)
        end_voltage = params.get("end_voltage", 1.0)
        voltage_step = params.get("voltage_step", 0.01)
        sample_time = params.get("sample_time", 0.01)
        
        # Calculate voltage points
        voltage_points = np.arange(start_voltage, end_voltage, voltage_step)
        num_points = len(voltage_points)
        
        # Initialize results data
        results["data"] = {
            "voltage": [],
            "current": {},
            "resistance": {},
            "timestamp": []
        }
        
        # Initialize current and resistance dictionaries for each read channel
        for ch in mask_to_read_i:
            results["data"]["current"][ch] = []
            results["data"]["resistance"][ch] = []
        
        # Simulate measurement for each voltage point
        for i, voltage in enumerate(voltage_points):
            # Update progress
            results["progress"] = int((i / num_points) * 100)
            
            # Apply voltage to bias channels
            for ch in mask_to_bias:
                # In a real implementation, this would update the internal state of the device
                pass
                
            # Simulate measurement delay
            time.sleep(sample_time * 0.01)  # Scale down for simulation speed
            
            # Record timestamp
            timestamp = time.time()
            results["data"]["timestamp"].append(timestamp)
            results["data"]["voltage"].append(voltage)
            
            # Simulate measuring current on read channels
            for ch in mask_to_read_i:
                # Calculate current based on simple model
                if ch in mask_to_bias:
                    # Channel is biased
                    resistance = 1000 + 500 * np.sin(voltage * 3.14)  # Simple nonlinear response
                    current = voltage / resistance
                elif ch in mask_to_gnd:
                    # Channel is grounded
                    current = -voltage / 1000  # Simple linear response
                else:
                    # Channel is floating
                    current = voltage / 10000 * random.random() * 0.1
                
                # Add noise
                current += (random.random() - 0.5) * 1e-9
                
                # Record values
                results["data"]["current"][ch].append(current)
                
                # Calculate resistance
                if abs(current) > 1e-12:  # Avoid division by zero
                    resistance = voltage / current
                else:
                    resistance = float('inf')
                results["data"]["resistance"][ch].append(resistance)
            
            # Send update every 10% progress or every 5 data points for real-time visualization
            if i % 5 == 0 or i % (num_points // 10) == 0 or i == num_points - 1:
                # Create a copy of the results with just the latest data points
                update = {
                    "experiment_type": results["experiment_type"],
                    "status": results["status"],
                    "progress": results["progress"],
                    "data": {
                        "timestamp": results["data"]["timestamp"][-5:],
                        "voltage": results["data"]["voltage"][-5:],
                        "current": {ch: results["data"]["current"][ch][-5:] for ch in results["data"]["current"]},
                        "resistance": {ch: results["data"]["resistance"][ch][-5:] for ch in results["data"]["resistance"]}
                    }
                }
                
                result_queue.put(update)
    
    def _simulate_conductivity_matrix(self, params: dict, results: dict, result_queue: multiprocessing.Queue) -> None:
        """Simulate conductivity matrix experiment"""
        mask = params.get("mask", list(range(8, 24)))  # Default to channels 8-23
        v_read = params.get("v_read", 0.05)
        sample_time = params.get("sample_time", 0.01)
        n_reps_avg = params.get("n_reps_avg", 10)
        
        # Create all possible channel pairs
        channel_pairs = [(i, j) for i in mask for j in mask if i < j]
        num_pairs = len(channel_pairs)
        
        # Initialize conductivity matrix
        conductivity_matrix = np.zeros((len(mask), len(mask)))
        
        # Initialize results data
        results["data"] = {
            "conductivity_matrix": conductivity_matrix.tolist(),
            "channel_pairs": channel_pairs,
            "measurements": {}
        }
        
        # Simulate measurement for each channel pair
        for i, (ch1, ch2) in enumerate(channel_pairs):
            # Update progress
            results["progress"] = int((i / num_pairs) * 100)
            
            # Simulate measurement
            time.sleep(sample_time * 0.01 * n_reps_avg * 0.1)  # Scale down for simulation speed
            
            # Calculate conductance between the two channels
            # In a real implementation, this would use the device's internal state
            conductance = 0.01 + 0.05 * random.random()
            if abs(ch1 - ch2) == 1:  # Adjacent channels have higher conductance
                conductance *= 2
            
            # Update conductivity matrix
            conductivity_matrix[mask.index(ch1), mask.index(ch2)] = conductance
            conductivity_matrix[mask.index(ch2), mask.index(ch1)] = conductance
            
            # Record measurement
            results["data"]["measurements"][f"{ch1}_{ch2}"] = {
                "voltage": v_read,
                "current": v_read * conductance,
                "conductance": conductance
            }
            
            # Update results
            results["data"]["conductivity_matrix"] = conductivity_matrix.tolist()
            
            # Send update every 10% progress
            if i % (num_pairs // 10) == 0 or i == num_pairs - 1:
                result_queue.put(results.copy())
    
    def _simulate_pulse_measurement(self, params: dict, results: dict, result_queue: multiprocessing.Queue) -> None:
        """Simulate pulse measurement experiment"""
        mask_to_bias = params.get("mask_to_bias", [])
        mask_to_gnd = params.get("mask_to_gnd", [])
        mask_to_read_v = params.get("mask_to_read_v", [])
        mask_to_read_i = params.get("mask_to_read_i", [])
        
        pre_pulse_time = params.get("pre_pulse_time", 10)
        pulse_time = params.get("pulse_time", 10)
        post_pulse_time = params.get("post_pulse_time", 300)
        pulse_voltage = params.get("pulse_voltage", [1])
        interpulse_voltage = params.get("interpulse_voltage", 0.05)
        sample_time = params.get("sample_time", 0.2)
        
        total_time = pre_pulse_time + pulse_time + post_pulse_time
        samples_per_second = 1 / sample_time
        total_samples = int(total_time * samples_per_second)
        
        # Initialize results data
        results["data"] = {
            "time": [],
            "voltage": [],
            "current": {},
            "resistance": {}
        }
        
        # Initialize current and resistance dictionaries for each read channel
        for ch in mask_to_read_i:
            results["data"]["current"][ch] = []
            results["data"]["resistance"][ch] = []
        
        # Simulate measurement
        start_time = time.time()
        
        for i in range(total_samples):
            # Calculate current time
            current_time = i * sample_time
            
            # Update progress
            results["progress"] = int((i / total_samples) * 100)
            
            # Determine voltage based on time
            if current_time < pre_pulse_time:
                # Pre-pulse phase
                voltage = interpulse_voltage
            elif current_time < pre_pulse_time + pulse_time:
                # Pulse phase
                voltage = pulse_voltage[0]
            else:
                # Post-pulse phase
                voltage = interpulse_voltage
            
            # Record time and voltage
            results["data"]["time"].append(current_time)
            results["data"]["voltage"].append(voltage)
            
            # Simulate measuring current on read channels
            for ch in mask_to_read_i:
                # Calculate current based on simple model
                if ch in mask_to_bias:
                    # Channel is biased
                    # Simulate memristor-like relaxation behavior
                    if current_time >= pre_pulse_time and current_time < pre_pulse_time + pulse_time:
                        # During pulse: resistance decreases
                        resistance = 10000 * (1 - 0.5 * (current_time - pre_pulse_time) / pulse_time)
                    elif current_time >= pre_pulse_time + pulse_time:
                        # After pulse: resistance slowly recovers (relaxation)
                        t_after_pulse = current_time - (pre_pulse_time + pulse_time)
                        recovery_rate = 0.01  # Slower recovery
                        resistance = 5000 * (1 + recovery_rate * t_after_pulse)
                    else:
                        # Before pulse: stable resistance
                        resistance = 10000
                    
                    current = voltage / resistance
                else:
                    # Other channels
                    resistance = 100000
                    current = voltage / resistance
                
                # Add noise
                current += (random.random() - 0.5) * current * 0.05
                
                # Record values
                results["data"]["current"][ch].append(current)
                results["data"]["resistance"][ch].append(resistance)
            
            # Simulate measurement delay (reduced for simulation speed)
            time.sleep(sample_time * 0.01)
            
            # Send update every 10% progress
            if i % (total_samples // 10) == 0 or i == total_samples - 1:
                result_queue.put(results.copy())
    
    def _simulate_memory_capacity(self, params: dict, results: dict, result_queue: multiprocessing.Queue) -> None:
        """Simulate memory capacity experiment"""
        n_samples = params.get("n_samples", 3000)
        sample_time = params.get("sample_time", 0.1)
        v_read = params.get("v_read", 0.05)
        v_bias_vector = params.get("v_bias_vector", None)
        
        # Generate random bias vector if not provided
        if v_bias_vector is None:
            v_bias_vector = np.random.uniform(0.5, 1.5, n_samples).tolist()
        
        # Initialize results data
        results["data"] = {
            "sample_index": list(range(n_samples)),
            "v_bias": v_bias_vector,
            "response": [],
            "memory_capacity": 0
        }
        
        # Simulate memory capacity measurement
        for i in range(n_samples):
            # Update progress
            results["progress"] = int((i / n_samples) * 100)
            
            # Generate response based on input history
            # Simple model: response depends on current input and previous inputs with decay
            if i == 0:
                response = v_bias_vector[i] * 0.8 + 0.1
            else:
                history_weight = min(10, i) / 10  # More history weight as we progress
                current_weight = 1 - history_weight
                
                # Calculate weighted sum of previous inputs
                history_sum = 0
                for j in range(1, min(10, i)+1):
                    history_sum += v_bias_vector[i-j] * (0.8 ** j)  # Exponential decay
                
                history_response = history_sum / min(10, i)
                current_response = v_bias_vector[i] * 0.8 + 0.1
                
                response = current_weight * current_response + history_weight * history_response
            
            # Add noise
            response += (random.random() - 0.5) * 0.05
            
            # Ensure response is in reasonable range
            response = max(0, min(1, response))
            
            # Record response
            results["data"]["response"].append(response)
            
            # Simulate measurement delay (reduced for simulation speed)
            time.sleep(sample_time * 0.001)
            
            # Send update every 10% progress
            if i % (n_samples // 10) == 0 or i == n_samples - 1:
                # Calculate estimated memory capacity (simplified)
                if i > 100:
                    # Simple estimation based on autocorrelation
                    # In a real implementation, this would be a more complex calculation
                    memory_capacity = random.uniform(0.4, 0.8)  # Placeholder
                    results["data"]["memory_capacity"] = memory_capacity
                
                result_queue.put(results.copy())
        
        # Final memory capacity calculation
        memory_capacity = random.uniform(0.5, 0.9)  # Placeholder
        results["data"]["memory_capacity"] = memory_capacity
        result_queue.put(results.copy())
    
    def _simulate_other_experiments(self, experiment_type: str, params: dict, results: dict, result_queue: multiprocessing.Queue) -> None:
        """Simple simulation for other experiment types"""
        total_steps = 100
        
        # Initialize results data
        results["data"] = {
            "time": [],
            "values": []
        }
        
        # Simulate experiment steps
        for i in range(total_steps):
            # Update progress
            results["progress"] = int((i / total_steps) * 100)
            
            # Simulate measurement
            time.sleep(0.05)  # Simulation delay
            
            # Record simulated data
            results["data"]["time"].append(time.time())
            results["data"]["values"].append(random.random())
            
            # Send update every 10% progress
            if i % (total_steps // 10) == 0 or i == total_steps - 1:
                result_queue.put(results.copy())
    
    # Use the simpler simulation for remaining experiment types
    _simulate_noise_measurement = _simulate_other_experiments
    _simulate_reservoir_computing = _simulate_other_experiments
    _simulate_tomography = _simulate_other_experiments
    _simulate_turn_on = _simulate_other_experiments
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get the status of the current experiment"""
        if not self.experiment_process or not self.experiment_process.is_alive():
            return {"status": "idle"}
        
        # Check if there are any results in the queue
        try:
            result = self.result_queue.get_nowait()
            # Put the result back in the queue for other consumers
            self.result_queue.put(result)
            return result
        except queue.Empty:
            return {"status": "running", "progress": 0}
    
    def stop_experiment(self) -> bool:
        """Stop the current experiment"""
        if self.experiment_process and self.experiment_process.is_alive():
            self.experiment_process.terminate()
            self.experiment_process.join(timeout=1)
            return True
        return False 