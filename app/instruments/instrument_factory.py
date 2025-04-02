# instrument_factory.py

from .keithley6430 import Keithley6430
from.keithley4200 import Keithley4200
from connections.connection_manager import ConnectionManager

def instrument_factory(model, address, protocol="GPIB"):
    connection = ConnectionManager(address, protocol)
    
    if model == "Keithley6430":
        return Keithley6430(connection)
    elif model == "Keithley4200":
        return Keithley4200(connection)
    else:
        raise ValueError("Unsupported instrument model")
