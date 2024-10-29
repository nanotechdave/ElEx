# pulsed_measurement.py
from instruments.keithley4200 import Keithley4200

def pulsed_measurement_experiment():
    # Example usage of the Keithley4200 instrument for pulsed measurement
    address = "GPIB::24"  # Instrument address
    keithley = Keithley4200(address)

    # Set up pulse parameters
    keithley.set_pulse_parameters(pulse_width=0.5, pulse_delay=0.1)

    # Run pulsed measurement and save results
    savepath = "01_Raw_data/pulsing/Sample1"
    measurement_name = "sample_measurement"
    keithley.run_pulse_measurement(savepath, measurement_name)
