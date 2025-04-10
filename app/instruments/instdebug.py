import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import multiprocessing

class InstDebug:
    """
    A dummy measurement instrument that simulates a 16-channel measurement device
    with current and voltage reading capabilities.
    """
    
    def __init__(self, serial_number: str = "DEBUG001", simulation_mode: str = "normal"):
        self.serial_number = serial_number
        self.simulation_mode = simulation_mode
        self.channels = 16
        self.connected = True
        
        # Default parameters
        self.voltage_range = 10.0  # Voltage range in V
        self.current_range = 1e-3  # Current range in A
        self.integration_time = 0.1  # Integration time in s
        self.noise_level = 0.01  # Noise level (fraction of reading)
        
        # Channel resistances (Ohms) - different for each channel to simulate different behaviors
        self.resistances = {ch: 1000 * (ch + 1) for ch in range(self.channels)}
        
        # Applied voltages/currents per channel
        self.applied_voltage = {ch: 0.0 for ch in range(self.channels)}
        self.applied_current = {ch: 0.0 for ch in range(self.channels)}
        
        # Mode per channel (voltage or current source)
        self.source_mode = {ch: "voltage" for ch in range(self.channels)}
        
        # Compliance values
        self.voltage_compliance = 10.0
        self.current_compliance = 0.1
        
        # Result queue for experiment communication
        self.result_queue = None
    
    def set_result_queue(self, queue: multiprocessing.Queue) -> None:
        """
        Set the result queue for sending experiment updates.
        
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
        
        # Expand resistances and other per-channel arrays if needed
        if hasattr(self, 'resistances') and self.channels > len(self.resistances):
            for ch in range(len(self.resistances), self.channels):
                self.resistances[ch] = 1000 * (ch + 1)
                self.applied_voltage[ch] = 0.0
                self.applied_current[ch] = 0.0
                self.source_mode[ch] = "voltage"
                
    def connect(self) -> bool:
        """Simulate connection to the instrument."""
        time.sleep(0.5)  # Simulate connection delay
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """Simulate disconnection from the instrument."""
        time.sleep(0.2)  # Simulate disconnection delay
        self.connected = False
        return True
    
    def is_connected(self) -> bool:
        """Check if the instrument is connected."""
        return self.connected
    
    def set_source_mode(self, channel: int, mode: str) -> None:
        """Set the source mode (voltage or current) for a channel."""
        if 0 <= channel < self.channels:
            if mode.lower() in ["voltage", "current"]:
                self.source_mode[channel] = mode.lower()
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'voltage' or 'current'")
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set the voltage for a channel."""
        if 0 <= channel < self.channels:
            if abs(voltage) <= self.voltage_range:
                self.applied_voltage[channel] = voltage
                if self.source_mode[channel] == "voltage":
                    # Calculate current based on Ohm's law
                    ideal_current = voltage / self.resistances[channel]
                    # Apply compliance limit
                    if abs(ideal_current) > self.current_compliance:
                        self.applied_current[channel] = self.current_compliance if ideal_current > 0 else -self.current_compliance
                    else:
                        self.applied_current[channel] = ideal_current
            else:
                raise ValueError(f"Voltage must be within ±{self.voltage_range}V")
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def set_current(self, channel: int, current: float) -> None:
        """Set the current for a channel."""
        if 0 <= channel < self.channels:
            if abs(current) <= self.current_range:
                self.applied_current[channel] = current
                if self.source_mode[channel] == "current":
                    # Calculate voltage based on Ohm's law
                    ideal_voltage = current * self.resistances[channel]
                    # Apply compliance limit
                    if abs(ideal_voltage) > self.voltage_compliance:
                        self.applied_voltage[channel] = self.voltage_compliance if ideal_voltage > 0 else -self.voltage_compliance
                    else:
                        self.applied_voltage[channel] = ideal_voltage
            else:
                raise ValueError(f"Current must be within ±{self.current_range}A")
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def measure_voltage(self, channel: int) -> float:
        """Measure the voltage on a channel."""
        if 0 <= channel < self.channels:
            # Add noise to the measurement
            noise = self.applied_voltage[channel] * self.noise_level * (2 * random.random() - 1)
            time.sleep(self.integration_time)  # Simulate measurement time
            return self.applied_voltage[channel] + noise
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def measure_current(self, channel: int) -> float:
        """Measure the current on a channel."""
        if 0 <= channel < self.channels:
            # Add noise to the measurement
            noise = self.applied_current[channel] * self.noise_level * (2 * random.random() - 1)
            time.sleep(self.integration_time)  # Simulate measurement time
            return self.applied_current[channel] + noise
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def measure_resistance(self, channel: int) -> float:
        """Measure the resistance on a channel."""
        voltage = self.measure_voltage(channel)
        current = self.measure_current(channel)
        if abs(current) < 1e-12:  # Avoid division by zero
            return float('inf')
        return voltage / current
    
    def set_integration_time(self, time_s: float) -> None:
        """Set the integration time in seconds."""
        if time_s > 0:
            self.integration_time = time_s
        else:
            raise ValueError("Integration time must be positive")
    
    def set_noise_level(self, level: float) -> None:
        """Set the noise level as a fraction of the reading."""
        if 0 <= level <= 1:
            self.noise_level = level
        else:
            raise ValueError("Noise level must be between 0 and 1")
    
    def set_resistance(self, channel: int, resistance: float) -> None:
        """Set the simulated resistance for a channel."""
        if 0 <= channel < self.channels:
            if resistance > 0:
                self.resistances[channel] = resistance
            else:
                raise ValueError("Resistance must be positive")
        else:
            raise ValueError(f"Channel must be between 0 and {self.channels-1}")
    
    def get_id(self) -> str:
        """Get the instrument ID."""
        return f"InstDebug {self.serial_number} (Simulation Mode: {self.simulation_mode})"
    
    def set_voltage_compliance(self, value: float) -> None:
        """Set the voltage compliance limit."""
        if value > 0:
            self.voltage_compliance = value
        else:
            raise ValueError("Voltage compliance must be positive")
    
    def set_current_compliance(self, value: float) -> None:
        """Set the current compliance limit."""
        if value > 0:
            self.current_compliance = value
        else:
            raise ValueError("Current compliance must be positive")
    
    def autorange_current(self, channel: int) -> None:
        """Simulate auto-ranging for current measurements."""
        pass  # Just a placeholder, doesn't need to do anything in simulation
    
    def autorange_voltage(self, channel: int) -> None:
        """Simulate auto-ranging for voltage measurements."""
        pass  # Just a placeholder, doesn't need to do anything in simulation
    
    def simulate_experiment(self, experiment_type: str, params: dict) -> None:
        """
        Simulate a basic experiment and send updates to the result queue.
        This is a simplified version that will be overridden by more complex instruments.
        
        Args:
            experiment_type: Type of experiment to simulate
            params: Dictionary of experiment parameters
        """
        if self.result_queue is None:
            print("Warning: No result queue set. Results will not be sent to GUI.")
            return
            
        print(f"Starting {experiment_type} simulation in InstDebug...")
        
        # Initialize results dictionary
        results = {
            "type": "status_update",
            "experiment_type": experiment_type,
            "status": "running",
            "progress": 0,
            "message": f"Starting {experiment_type} experiment",
            "timestamp": time.time()
        }
        
        # Send initial status
        self.result_queue.put(results.copy())
        
        try:
            # Choose simulation method based on experiment type
            if experiment_type == "IVMeasurement":
                self._simulate_iv_measurement(params)
            elif experiment_type == "MemoryCapacity":
                self._simulate_memory_capacity(params)
            elif experiment_type == "NoiseMeasurement":
                self._simulate_noise_measurement(params)
            elif experiment_type == "ActivationPattern":
                self._simulate_activation_pattern(params)
            elif experiment_type == "PulseMeasurement":
                self._simulate_pulse_measurement(params)
            elif experiment_type == "ConductivityMatrix":
                self._simulate_conductivity_matrix(params)
            elif experiment_type == "ReservoirComputing":
                self._simulate_reservoir_computing(params)
            elif experiment_type == "Tomography":
                self._simulate_tomography(params)
            elif experiment_type == "TurnOn":
                self._simulate_turn_on(params)
            else:
                # Generic simulation for unknown experiment types
                self._simulate_generic_experiment(params)
            
            # Send completion status
            self.result_queue.put({
                "type": "status_update",
                "experiment_type": experiment_type,
                "status": "completed",
                "progress": 100,
                "message": f"Completed {experiment_type} experiment",
                "timestamp": time.time()
            })
            
        except Exception as e:
            # Send error status
            self.result_queue.put({
                "type": "status_update",
                "experiment_type": experiment_type,
                "status": "error",
                "progress": 0,
                "message": f"Error: {str(e)}",
                "timestamp": time.time()
            })
            print(f"Error simulating experiment: {e}")

    def _simulate_generic_experiment(self, params: dict) -> None:
        """Generic simulation for unknown experiment types"""
        total_steps = params.get("steps", 100)
        
        # Initialize data
        timestamps = []
        voltages = {}
        currents = {}
        resistances = {}
        
        # Use a few random channels
        channels = random.sample(range(self.channels), min(4, self.channels))
        
        for step in range(total_steps):
            # Simulate measurement time
            time.sleep(0.02)  # 20ms per step
            
            # Current timestamp
            current_time = time.time()
            timestamps.append(current_time)
            
            # Generate random data for each channel
            for ch in channels:
                # First time, initialize the data arrays
                if step == 0:
                    voltages[ch] = []
                    currents[ch] = []
                    resistances[ch] = []
                
                # Generate random data (realistic values)
                voltage = 0.1 + 0.9 * step / total_steps  # 0.1V to 1.0V
                current = voltage / self.resistances[ch]  # Follow Ohm's law
                
                # Add noise
                voltage += voltage * self.noise_level * (2 * random.random() - 1)
                current += current * self.noise_level * (2 * random.random() - 1)
                
                # Store the data
                voltages[ch].append(voltage)
                currents[ch].append(current)
                resistances[ch].append(voltage / current if current != 0 else float('inf'))
            
            # Update progress
            progress = int(100 * (step + 1) / total_steps)
            
            # Check and fix any length mismatches
            for ch in channels:
                if len(timestamps) != len(voltages[ch]) or len(timestamps) != len(currents[ch]):
                    print(f"Length mismatch in generic data: timestamps[{len(timestamps)}], voltages[{ch}][{len(voltages[ch])}], currents[{ch}][{len(currents[ch])}]")
                    # Fix the lengths by truncating to the shortest
                    min_len = min(len(timestamps), len(voltages[ch]), len(currents[ch]))
                    timestamps = timestamps[:min_len]
                    voltages[ch] = voltages[ch][:min_len]
                    currents[ch] = currents[ch][:min_len]
                    resistances[ch] = resistances[ch][:min_len]
            
            # Send update to the GUI
            self.result_queue.put({
                "type": "data_update",
                "experiment_type": "generic",
                "status": "running",
                "progress": progress,
                "message": f"Step {step+1}/{total_steps}",
                "timestamp": current_time,
                "data": {
                    "timestamps": timestamps.copy(),
                    "voltages": {ch: voltages[ch].copy() for ch in channels},
                    "currents": {ch: currents[ch].copy() for ch in channels},
                    "resistances": {ch: resistances[ch].copy() for ch in channels}
                }
            })

    def _simulate_iv_measurement(self, params: dict) -> None:
        """Simulate I-V measurement with realistic IV curves"""
        start_voltage = params.get("start_voltage", -1.0)
        end_voltage = params.get("end_voltage", 1.0)
        voltage_steps = params.get("voltage_steps", 100)
        channels = params.get("channels", random.sample(range(self.channels), min(4, self.channels)))
        
        # Create voltage array
        voltages_array = np.linspace(start_voltage, end_voltage, voltage_steps)
        
        # Initialize data
        timestamps = []
        voltages = {ch: [] for ch in channels}
        currents = {ch: [] for ch in channels}
        resistances = {ch: [] for ch in channels}
        
        for step, voltage in enumerate(voltages_array):
            # Simulate measurement time
            time.sleep(0.02)  # 20ms per step
            
            # Current timestamp
            current_time = time.time()
            timestamps.append(current_time)
            
            # For each channel
            for ch in channels:
                # Apply voltage
                self.set_voltage(ch, voltage)
                
                # Measure current (using Ohm's law with non-linearity for realism)
                # Simulate non-linear resistance that depends on voltage
                effective_resistance = self.resistances[ch] * (1 + 0.2 * abs(voltage))
                ideal_current = voltage / effective_resistance
                
                # Add noise
                measured_voltage = voltage + voltage * self.noise_level * (2 * random.random() - 1)
                measured_current = ideal_current + ideal_current * self.noise_level * (2 * random.random() - 1)
                
                # Store the data
                voltages[ch].append(measured_voltage)
                currents[ch].append(measured_current)
                
                # Calculate resistance
                if abs(measured_current) > 1e-12:  # Avoid division by zero
                    resistances[ch].append(measured_voltage / measured_current)
                else:
                    resistances[ch].append(float('inf'))
            
            # Update progress
            progress = int(100 * (step + 1) / voltage_steps)
            
            # Send the data only if there are no length mismatches
            for ch in channels:
                if len(timestamps) != len(voltages[ch]) or len(timestamps) != len(currents[ch]):
                    print(f"Length mismatch in data: timestamps[{len(timestamps)}], voltages[{ch}][{len(voltages[ch])}], currents[{ch}][{len(currents[ch])}]")
                    # Fix the lengths by truncating to the shortest
                    min_len = min(len(timestamps), len(voltages[ch]), len(currents[ch]))
                    timestamps = timestamps[:min_len]
                    voltages[ch] = voltages[ch][:min_len]
                    currents[ch] = currents[ch][:min_len]
                    resistances[ch] = resistances[ch][:min_len]
            
            # Send update to the GUI
            self.result_queue.put({
                "type": "data_update",
                "experiment_type": "IV",
                "status": "running",
                "progress": progress,
                "message": f"Measuring at {voltage:.2f}V ({progress}%)",
                "timestamp": current_time,
                "data": {
                    "timestamps": timestamps.copy(),
                    "voltages": {ch: voltages[ch].copy() for ch in channels},
                    "currents": {ch: currents[ch].copy() for ch in channels},
                    "resistances": {ch: resistances[ch].copy() for ch in channels}
                }
            })

    def _simulate_pulse_measurement(self, params: dict) -> None:
        """Simulate pulse measurement with realistic transient responses"""
        pulse_amplitude = params.get("pulse_amplitude", 1.0)
        pulse_width = params.get("pulse_width", 10)  # ms
        num_pulses = params.get("num_pulses", 5)
        channels = params.get("channels", random.sample(range(self.channels), min(4, self.channels)))
        
        # Time points (in ms)
        time_points = np.linspace(0, num_pulses * pulse_width * 3, 500)
        
        # Initialize data
        timestamps = []
        voltages = {ch: [] for ch in channels}
        currents = {ch: [] for ch in channels}
        
        # Generate pulse train
        for step, t in enumerate(time_points):
            # Simulate measurement time
            time.sleep(0.01)  # 10ms per step
            
            # Current timestamp
            current_time = time.time()
            timestamps.append(current_time)
            
            # For each channel
            for ch in channels:
                # Calculate pulse voltage at this time
                pulse_state = 0
                for p in range(num_pulses):
                    pulse_start = p * pulse_width * 3
                    pulse_end = pulse_start + pulse_width
                    if pulse_start <= t <= pulse_end:
                        pulse_state = pulse_amplitude
                        break
                
                # Apply the voltage
                self.set_voltage(ch, pulse_state)
                
                # Calculate current response with capacitive effects
                # Simulate a simple RC circuit response
                tau = 0.2 * pulse_width  # RC time constant (in ms)
                
                # Find the closest pulse transition
                if pulse_state > 0:
                    # Rising edge - track how long into the pulse we are
                    for p in range(num_pulses):
                        pulse_start = p * pulse_width * 3
                        if pulse_start <= t:
                            time_since_edge = t - pulse_start
                            break
                    # Exponential charging equation
                    response_factor = 1 - np.exp(-time_since_edge / tau)
                else:
                    # Falling edge - find the most recent pulse end
                    time_since_edge = 0
                    for p in range(num_pulses):
                        pulse_start = p * pulse_width * 3
                        pulse_end = pulse_start + pulse_width
                        if t > pulse_end:
                            time_since_edge = t - pulse_end
                    # Exponential discharging equation
                    response_factor = np.exp(-time_since_edge / tau)
                    if time_since_edge == 0:
                        response_factor = 0
                
                # Calculate the current
                steady_current = pulse_state / self.resistances[ch]
                current = steady_current * response_factor
                
                # Add noise
                measured_voltage = pulse_state + pulse_state * self.noise_level * (2 * random.random() - 1)
                measured_current = current + current * self.noise_level * (2 * random.random() - 1)
                
                # Store the data
                voltages[ch].append(measured_voltage)
                currents[ch].append(measured_current)
            
            # Update progress
            progress = int(100 * (step + 1) / len(time_points))
            
            # Check and fix any length mismatches
            for ch in channels:
                if len(timestamps) != len(voltages[ch]) or len(timestamps) != len(currents[ch]):
                    print(f"Length mismatch in pulse data: timestamps[{len(timestamps)}], voltages[{ch}][{len(voltages[ch])}], currents[{ch}][{len(currents[ch])}]")
                    # Fix the lengths by truncating to the shortest
                    min_len = min(len(timestamps), len(voltages[ch]), len(currents[ch]))
                    timestamps = timestamps[:min_len]
                    voltages[ch] = voltages[ch][:min_len]
                    currents[ch] = currents[ch][:min_len]
            
            # Send update to the GUI
            self.result_queue.put({
                "type": "data_update",
                "experiment_type": "Pulse",
                "status": "running",
                "progress": progress,
                "message": f"Measuring pulse response ({progress}%)",
                "timestamp": current_time,
                "data": {
                    "timestamps": timestamps.copy(),
                    "voltages": {ch: voltages[ch].copy() for ch in channels},
                    "currents": {ch: currents[ch].copy() for ch in channels},
                    "time_points": time_points[:step+1].tolist()
                }
            })

    def _simulate_noise_measurement(self, params: dict) -> None:
        """Simulate noise measurement with realistic noise characteristics"""
        duration = params.get("duration", 10)  # seconds
        sample_rate = params.get("sample_rate", 100)  # Hz
        channels = params.get("channels", random.sample(range(self.channels), min(4, self.channels)))
        bias_voltage = params.get("bias_voltage", 0.1)  # V
        
        # Total number of samples
        num_samples = int(duration * sample_rate)
        
        # Time points
        time_points = np.linspace(0, duration, num_samples)
        
        # Initialize data
        timestamps = []
        voltages = {ch: [] for ch in channels}
        currents = {ch: [] for ch in channels}
        
        # Generate pink noise (1/f noise) characteristics
        # Using numpy's random capabilities to simulate realistic electronic noise
        for step, t in enumerate(time_points):
            # Only sleep occasionally to speed up simulation
            if step % 10 == 0:
                time.sleep(0.01)
            
            # Current timestamp
            current_time = time.time()
            timestamps.append(current_time)
            
            # For each channel
            for ch in channels:
                # Set bias voltage
                self.set_voltage(ch, bias_voltage)
                
                # Base current from Ohm's law
                base_current = bias_voltage / self.resistances[ch]
                
                # Generate noise components:
                # 1. White noise (thermal noise)
                white_noise = np.random.normal(0, 0.005 * base_current)
                
                # 2. Pink noise (1/f noise)
                # Simplified approximation using low frequency sinusoids
                pink_noise = 0
                for f in [0.1, 0.3, 0.7, 1.3, 2.1, 3.5]:
                    pink_noise += (0.01 * base_current / np.sqrt(f)) * np.sin(2 * np.pi * f * t + random.random() * 2 * np.pi)
                
                # 3. Random telegraph noise (RTN) - sudden jumps
                if random.random() < 0.01:  # 1% chance of RTN event
                    rtn_amplitude = 0.02 * base_current * (2 * random.random() - 1)
                else:
                    rtn_amplitude = 0
                
                # Combined noise
                noise = white_noise + pink_noise + rtn_amplitude
                
                # Measured values with noise
                measured_voltage = bias_voltage + bias_voltage * 0.001 * (2 * random.random() - 1)  # Very small voltage noise
                measured_current = base_current + noise
                
                # Store the data
                voltages[ch].append(measured_voltage)
                currents[ch].append(measured_current)
            
            # Update progress
            if step % 10 == 0 or step == num_samples - 1:
                progress = int(100 * (step + 1) / num_samples)
                
                # Check and fix any length mismatches
                for ch in channels:
                    if len(timestamps) != len(voltages[ch]) or len(timestamps) != len(currents[ch]):
                        print(f"Length mismatch in noise data: timestamps[{len(timestamps)}], voltages[{ch}][{len(voltages[ch])}], currents[{ch}][{len(currents[ch])}]")
                        # Fix the lengths by truncating to the shortest
                        min_len = min(len(timestamps), len(voltages[ch]), len(currents[ch]))
                        timestamps = timestamps[:min_len]
                        voltages[ch] = voltages[ch][:min_len]
                        currents[ch] = currents[ch][:min_len]
                
                # Send update to the GUI
                self.result_queue.put({
                    "type": "data_update",
                    "experiment_type": "Noise",
                    "status": "running",
                    "progress": progress,
                    "message": f"Measuring noise ({progress}%)",
                    "timestamp": current_time,
                    "data": {
                        "timestamps": timestamps.copy(),
                        "voltages": {ch: voltages[ch].copy() for ch in channels},
                        "currents": {ch: currents[ch].copy() for ch in channels},
                        "time_points": time_points[:step+1].tolist()
                    }
                })

    # Placeholder methods for other experiment types
    def _simulate_memory_capacity(self, params: dict) -> None:
        """Simulate memory capacity experiment"""
        self._simulate_generic_experiment(params)
        
    def _simulate_activation_pattern(self, params: dict) -> None:
        """Simulate activation pattern experiment"""
        self._simulate_generic_experiment(params)
        
    def _simulate_conductivity_matrix(self, params: dict) -> None:
        """Simulate conductivity matrix experiment"""
        self._simulate_generic_experiment(params)
        
    def _simulate_reservoir_computing(self, params: dict) -> None:
        """Simulate reservoir computing experiment"""
        self._simulate_generic_experiment(params)
        
    def _simulate_tomography(self, params: dict) -> None:
        """Simulate tomography experiment"""
        self._simulate_generic_experiment(params)
        
    def _simulate_turn_on(self, params: dict) -> None:
        """Simulate turn on experiment"""
        self._simulate_generic_experiment(params)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the instrument"""
        return {
            "name": "InstDebug",
            "serial_number": self.serial_number,
            "channels": self.channels,
            "firmware_version": "1.0.0",
            "simulation_mode": self.simulation_mode
        }
    
    def connect_to_gnd(self, channel_list: List[int]) -> bool:
        """Connect channels to ground"""
        for ch in channel_list:
            if 0 <= ch < self.channels:
                self.source_mode[ch] = "gnd"
        return True
    
    def connect_to_bias(self, channel_list: List[int], voltage: float) -> bool:
        """Connect channels to bias voltage"""
        for ch in channel_list:
            if 0 <= ch < self.channels:
                self.source_mode[ch] = "bias"
                self.applied_voltage[ch] = voltage
        return True
    
    def set_to_float(self, channel_list: List[int]) -> bool:
        """Set channels to float"""
        for ch in channel_list:
            if 0 <= ch < self.channels:
                self.source_mode[ch] = "float"
        return True
    
    def vread_channels(self, channel_list: List[int], execute: bool = True) -> Dict[int, float]:
        """Read voltage from specified channels"""
        result = {}
        
        # Calculate voltages based on network
        for ch in channel_list:
            if 0 <= ch < self.channels:
                # For readability, store the channel mode
                mode = self.source_mode[ch]
                
                if mode == "gnd":
                    v = 0.0
                elif mode == "bias":
                    v = self.applied_voltage[ch]
                elif mode == "float":
                    # Calculate based on neighboring voltages
                    v = self._calculate_float_voltage(ch)
                else:
                    v = 0.0
                
                # Add noise
                if self.simulation_mode == "noisy":
                    noise = (random.random() - 0.5) * self.noise_level * self.voltage_range
                    v += noise
                
                result[ch] = v
        
        # Simulate measurement delay
        if execute:
            time.sleep(self.integration_time)
            self.execute()
        
        return result
    
    def iread_channels(self, channel_list: List[int], execute: bool = True) -> Dict[int, float]:
        """Read current from specified channels"""
        result = {}
        
        # Calculate currents based on network
        for ch in channel_list:
            if 0 <= ch < self.channels:
                # Calculate current based on Ohm's law and the resistance network
                i = self._calculate_channel_current(ch)
                
                # Add noise
                if self.simulation_mode == "noisy":
                    noise = (random.random() - 0.5) * self.noise_level * self.current_range
                    i += noise
                
                result[ch] = i
        
        # Simulate measurement delay
        if execute:
            time.sleep(self.integration_time)
            self.execute()
        
        return result
    
    def execute(self) -> bool:
        """Execute pending commands"""
        # Simulate execution delay
        time.sleep(0.05)
        
        # Update the channel states based on network interactions
        self._update_channel_states()
        
        return True
    
    def _calculate_float_voltage(self, channel: int) -> float:
        """
        Calculate the voltage of a floating channel based on connected channels
        
        This simulates how a floating channel's voltage is influenced by
        nearby channels that are at GND or bias potentials.
        """
        # Sum of conductances
        total_conductance = 0.0
        # Sum of voltage * conductance products
        weighted_voltage_sum = 0.0
        
        for other_ch in range(self.channels):
            if other_ch == channel:
                continue
                
            other_mode = self.source_mode[other_ch]
            
            # Only consider channels with defined voltage (GND or bias)
            if other_mode in ["gnd", "bias"]:
                # Get resistance between channels
                r = self.resistances[channel] if other_mode == "gnd" else self.resistances[other_ch]
                # Calculate conductance (1/R)
                g = 1.0 / r if r > 0 else 0.0
                
                total_conductance += g
                weighted_voltage_sum += g * self.applied_voltage[other_ch]
        
        # If no connections, return 0
        if total_conductance == 0:
            return 0.0
            
        # Calculate voltage using voltage divider principle
        return weighted_voltage_sum / total_conductance
    
    def _calculate_channel_current(self, channel: int) -> float:
        """
        Calculate the current flowing through a channel
        
        Uses Ohm's law and the resistance network to calculate current.
        """
        current = 0.0
        ch_mode = self.source_mode[channel]
        ch_voltage = self.applied_voltage[channel]
        
        # Current only flows if the channel is at a defined potential
        if ch_mode in ["gnd", "bias"]:
            for other_ch in range(self.channels):
                if other_ch == channel:
                    continue
                    
                other_mode = self.source_mode[other_ch]
                
                # Current flows between channels at different potentials
                if other_mode in ["gnd", "bias"]:
                    other_voltage = self.applied_voltage[other_ch]
                    voltage_diff = ch_voltage - other_voltage
                    
                    # Get resistance between channels
                    r = self.resistances[channel] if other_mode == "gnd" else self.resistances[other_ch]
                    
                    # Calculate current using Ohm's law (I = V/R)
                    if r > 0:
                        current += voltage_diff / r
        
        return current
    
    def _update_channel_states(self):
        """Update the state of all channels based on network interactions"""
        # For each channel, update voltage and current
        for ch in range(self.channels):
            mode = self.source_mode[ch]
            
            if mode == "float":
                # Update floating channel voltage
                v = self._calculate_float_voltage(ch)
                self.applied_voltage[ch] = v
            
            # Update channel current
            i = self._calculate_channel_current(ch)
            self.applied_current[ch] = i
    
    def pulse_slice_fast(self, pulse_params: List[Tuple], delays: List[int], execute: bool = True) -> bool:
        """
        Simulate fast pulsing
        
        Args:
            pulse_params: List of (channel, voltage, duration) tuples
            delays: List of delay times in microseconds
            execute: Whether to execute immediately
        """
        # Simulate the pulsing operation
        for params in pulse_params:
            channel, voltage, duration = params
            if 0 <= channel < self.channels:
                self.source_mode[channel] = "bias"
                self.applied_voltage[channel] = voltage
        
        # Simulate the delay
        time.sleep(sum(delays) / 1e6)  # Convert microseconds to seconds
        
        # Execute if requested
        if execute:
            self.execute()
        
        return True
    
    def pulse_slice_fast_open(self, pulse_params: List[Tuple], delays: List[int], execute: bool = True) -> bool:
        """
        Simulate fast pulsing with floating end
        
        Args:
            pulse_params: List of (channel, voltage, duration) tuples
            delays: List of delay times in microseconds
            execute: Whether to execute immediately
        """
        # First apply pulses
        result = self.pulse_slice_fast(pulse_params, delays, False)
        
        # Then float the channels at the end
        for params in pulse_params:
            channel, _, _ = params
            if 0 <= channel < self.channels:
                self.source_mode[channel] = "float"
        
        # Execute if requested
        if execute:
            self.execute()
        
        return result
    
    def delay(self, microseconds: int) -> bool:
        """
        Simulate a delay
        
        Args:
            microseconds: Delay in microseconds
        """
        time.sleep(microseconds / 1e6)  # Convert microseconds to seconds
        return True
    
    def measure(self, channel_list: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Measure voltage and current on specified channels
        
        Returns:
            Tuple of (voltage_dict, current_dict)
        """
        voltages = self.vread_channels(channel_list, False)
        currents = self.iread_channels(channel_list, False)
        self.execute()
        return voltages, currents 