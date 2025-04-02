# run_experiment.py
def run_experiment(instrument):
    instrument.set_voltage(5.0)
    voltage = instrument.read_voltage()
    current = instrument.read_current()
    print(f"Voltage: {voltage}, Current: {current}")
