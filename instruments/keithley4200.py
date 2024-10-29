# keithley4200.py
from .instrument_base import Instrument
from connections.keithley4200_communication import Keithley4200Communications
from utils.functions_pulsing import numSearch, fileInit, fileUpdate, dataInit

class Keithley4200(Instrument):
    def __init__(self, address):
        self.connection = Keithley4200Communications(address)
    
    def read_voltage(self):
        return self.connection.send_command("MEAS:VOLT?")

    def read_current(self):
        return self.connection.send_command("MEAS:CURR?")

    def set_voltage(self, voltage: float):
        self.connection.send_command(f"SOURCE:VOLT {voltage}")

    def set_current(self, current: float):
        self.connection.send_command(f"SOURCE:CURR {current}")

    def set_pulse_parameters(self, pulse_width, pulse_delay):
        self.connection.send_command(f"PULSE:WIDTH {pulse_width}")
        self.connection.send_command(f"PULSE:DELAY {pulse_delay}")

    def run_pulse_measurement(self, savepath, measurement_name):
        # Initialize file for data storage
        fileInit(savepath, measurement_name)
        
        # Set initial conditions or configurations as needed
        gen_num = numSearch(savepath)
        data = dataInit(gen_num)
        
        # Example pulsed measurement loop
        for i in range(10):  # Adjust number of pulses as needed
            voltage = self.read_voltage()
            current = self.read_current()
            fileUpdate(savepath, measurement_name, voltage, current)
