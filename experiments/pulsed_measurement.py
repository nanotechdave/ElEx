# pulsed_measurement.py
from instruments.keithley4200 import Keithley4200

# pulsed_measurement.py
def pulsed_measurement_experiment(instrument, pulse_width, pulse_delay, num_pulses, savepath, measurement_name):
    # Set up pulse parameters on the instrument
    instrument.set_pulse_parameters(pulse_width, pulse_delay)

    # Run the pulsed measurement, saving results to the specified path
    instrument.run_pulse_measurement(savepath, measurement_name)

