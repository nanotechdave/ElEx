import os

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Common settings
DEFAULT_INTEGRATION_TIME = 0.01  # seconds
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_VOLTAGE_RANGE = 1.0  # Volts
DEFAULT_CURRENT_RANGE = 1e-3  # Amps

# Experiment defaults
EXPERIMENT_DEFAULTS = {
    "IVMeasurement": {
        "start_voltage": -1.0,
        "end_voltage": 1.0,
        "voltage_step": 0.1,
        "sample_time": 0.01
    },
    "PulseMeasurement": {
        "pre_pulse_time": 10,
        "pulse_time": 10,
        "post_pulse_time": 100,
        "pulse_voltage": [1.0],
        "interpulse_voltage": 0.05,
        "sample_time": 0.2
    },
    "ConductivityMatrix": {
        "v_read": 0.05,
        "sample_time": 0.01,
        "n_reps_avg": 10
    }
}

# Mapping defaults
DEFAULT_MAPPING = "crossbar32"

# GUI settings
GUI_REFRESH_RATE = 100  # milliseconds
GUI_PLOT_MAX_POINTS = 10000  # Maximum number of points to keep in plot 