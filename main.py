# main.py
import json
from instruments.instrument_factory import instrument_factory
from experiments.pulsed_measurement import pulsed_measurement_experiment

def load_config():
    """Load settings from config.json."""
    with open("config.json", "r") as file:
        return json.load(file)

if __name__ == "__main__":
    # Load configuration settings
    config = load_config()
    
    # Instrument settings from config.json
    instrument_model = config["instrument"]["model"]
    address = config["instrument"]["address"]
    protocol = config["instrument"]["protocol"]
    
    # Initialize the instrument based on configuration
    instrument = instrument_factory(instrument_model, address, protocol)

    # Experiment parameters
    pulse_width = config["experiment"]["pulse_width"]
    pulse_delay = config["experiment"]["pulse_delay"]
    num_pulses = config["experiment"]["num_pulses"]
    savepath = config["experiment"]["savepath"]
    measurement_name = config["experiment"]["measurement_name"]
    
    # Run the experiment with configured parameters
    pulsed_measurement_experiment(instrument, pulse_width, pulse_delay, num_pulses, savepath, measurement_name)
