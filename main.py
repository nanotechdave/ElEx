# main.py
from instruments.instrument_factory import instrument_factory
from experiments.run_experiment import run_experiment

# Example usage
if __name__ == "__main__":
    instrument_2400 = instrument_factory("Keithley2400", "GPIB::24")
    instrument_2450 = instrument_factory("Keithley2450", "USB::0x05E6::0x2450::12345::INSTR")

    run_experiment(instrument_2400)
    run_experiment(instrument_2450)





