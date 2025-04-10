# InstDebug Simulation

This document explains how to use the InstDebug simulation for testing electronic measurement routines without requiring physical hardware.

## Overview

The InstDebug simulation provides a software-based simulation of an electronic measurement instrument with multiple channels. It can simulate various experiments such as:

- IV Measurements
- Pulse Measurements
- Noise Measurements
- Memory Capacity Tests
- And more...

The simulation is designed to run on a separate CPU core from the GUI to better reflect the behavior of a real hardware setup.

## Requirements

- Python 3.6 or higher
- NumPy
- Matplotlib (for visualization in the simple test script)
- Optional: psutil (for CPU core management)

Install with:
```bash
pip install numpy matplotlib psutil
```

## Testing the Simulation

There are two ways to test the InstDebug simulation:

### 1. Simple Test Script

The `simple_instdebug_test.py` script provides direct testing of the InstDebug simulation without requiring the full application's GUI infrastructure.

Run it with:

```bash
# Test IV Measurement (default)
python simple_instdebug_test.py

# Test Pulse Measurement
python simple_instdebug_test.py --test pulse

# Test Noise Measurement
python simple_instdebug_test.py --test noise
```

This script creates real-time plots of voltage and current measurements.

### 2. Full Application Test

The `test_instdebug.py` script attempts to load the full application GUI with the InstDebug simulation.

Run it with:

```bash
# Test with GUI
python test_instdebug.py

# Test directly without GUI
python test_instdebug.py --mode direct
```

## CPU Separation

The application is designed to run the GUI on one CPU core (usually core 0) and the instrument control code on separate cores. This helps ensure that the GUI remains responsive even during intensive measurements.

If the `psutil` library is available, core assignment is handled automatically. Otherwise, both components will run on the same CPU core.

## Realistic Simulation

The InstDebug simulation provides realistic simulated results for various experiment types:

1. **IV Measurement**:
   - Non-linear IV characteristics
   - Realistic noise profiles
   - Proper resistance calculations

2. **Pulse Measurement**:
   - RC-like transient responses
   - Charging/discharging effects
   - Multiple pulse handling

3. **Noise Measurement**:
   - White noise (thermal)
   - Pink noise (1/f)
   - Random Telegraph Noise (RTN)

## Troubleshooting

### "Missing Module" Errors

If you see errors about missing modules (e.g., "No module named 'pyarc2'"), this is expected when running in simulation mode. The application has fallbacks for these cases.

### GUI Issues

If the GUI fails to start, try running the simple test script instead:

```bash
python simple_instdebug_test.py
```

This bypasses the GUI infrastructure and tests the simulation directly.

### CPU Affinity Issues

If you see errors related to CPU affinity or psutil, you may need to install the psutil library:

```bash
pip install psutil
```

Alternatively, the application will run without CPU separation if psutil is unavailable.

## Advanced Usage

The InstDebug simulation can be customized by modifying the parameters in the test scripts or by directly instantiating and configuring the InstDebug class:

```python
from app.instruments.instdebug import InstDebug

# Create a custom instance
inst = InstDebug(serial_number="CUSTOM001", simulation_mode="normal")

# Customize parameters
inst.set_noise_level(0.02)  # 2% noise
inst.set_integration_time(0.2)  # 200ms integration time
inst.set_resistance(0, 2000)  # Set channel 0 to 2K ohms
``` 