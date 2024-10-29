# instrument_factory.py
from .keithley2400 import Keithley2400
from .keithley2450 import Keithley2450
from connections.connection_manager import ConnectionManager

def instrument_factory(model, address, protocol="GPIB"):
    connection = ConnectionManager(address, protocol)
    if model == "Keithley2400":
        return Keithley2400(connection)
    elif model == "Keithley2450":
        return Keithley2450(connection)
    else:
        raise ValueError("Unsupported instrument model")
