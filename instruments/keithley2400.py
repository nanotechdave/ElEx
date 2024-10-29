# keithley2400.py
from .instrument_base import Instrument

class Keithley2400(Instrument):
    def __init__(self, connection):
        self.connection = connection

    def read_voltage(self):
        # Device-specific command for reading voltage
        return self.connection.send("MEAS:VOLT?")

    # Implement other methods similarly
