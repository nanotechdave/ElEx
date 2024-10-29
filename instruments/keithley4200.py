# keithley4200.py
from .instrument_base import Instrument
from connections.keithley4200_communication import Keithley4200Communication
from utils.functions_pulsing import numSearch, fileInit, fileUpdate

class Keithley4200(Instrument):
    def __init__(self, address):
        # Initialize communication specifically for the Keithley 4200
        self.connection = Keithley4200Communication(address)
    
    def read_voltage(self):
        # Command for reading voltage from the Keithley 4200
        return self.connection.send_command("MEAS:VOLT?")

    def read_current(self):
        # Command for reading current from the Keithley 4200
        return self.connection.send_command("MEAS:CURR?")

    def set_voltage(self, voltage: float):
        self.connection.send_command(f"SOURCE:VOLT {voltage}")

    def set_current(self, current: float):
        self.connection.send_command(f"SOURCE:CURR {current}")

    def set_pulse_parameters(self, pulse_width, pulse_delay):
        # Set specific pulse parameters for pulsed measurements
        self.connection.send_command(f"PULSE:WIDTH {pulse_width}")
        self.connection.send_command(f"PULSE:DELAY {pulse_delay}")

    def run_pulse_measurement(self, savepath, measurement_name):
        # Prepare data saving
        fileInit(savepath, measurement_name)
        
        # Begin pulsed measurement process
        self.connection.send_command("PULSE:INIT")
        
        # Capture and save data at each step
        for i in range(10):  # example loop for data collection
            voltage = self.read_voltage()
            current = self.read_current()
            fileUpdate(savepath, measurement_name, voltage, current)
            
            # Optionally include a delay or pacing logic
