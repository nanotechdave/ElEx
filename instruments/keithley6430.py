# keithley6430.py
from .instrument_base import Instrument
from connections.keithley_communication import KeithleyCommunications
from utils.functions_pulsing import numSearch, fileInit, fileUpdate, dataInit

class Keithley6430(Instrument):
    def __init__(self, address):
        # Set up communication for the Keithley 6430
        self.connection = KeithleyCommunications(address)
    
    def read_voltage(self):
        # Device-specific command for reading voltage
        return self.connection.send_command("MEAS:VOLT?")

    def read_current(self):
        # Device-specific command for reading current
        return self.connection.send_command("MEAS:CURR?")

    def set_voltage(self, voltage: float):
        self.connection.send_command(f"SOURCE:VOLT {voltage}")

    def set_current(self, current: float):
        self.connection.send_command(f"SOURCE:CURR {current}")

    # Additional 6430-specific methods, if any, can go here
