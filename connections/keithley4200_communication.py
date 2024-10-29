# keithley_communication.py
import pyvisa as visa
import pyvisa.constants as pyconst

class Keithley4200Communications:
    """This class offers a collection of wrapper methods for PyVisa communication."""
    
    def __init__(self, address):
        self.rm = visa.ResourceManager()
        self.instrument = self.rm.open_resource(address)
        self.instrument.timeout = 5000
        self.instrument.write_termination = '\n'
        self.instrument.read_termination = '\n'

    def send_command(self, command):
        """Send a command to the instrument."""
        self.instrument.write(command)

    def read_response(self):
        """Read the instrument's response."""
        return self.instrument.read()
