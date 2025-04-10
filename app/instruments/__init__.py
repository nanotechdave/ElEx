# Try importing each instrument class, but don't fail if dependencies are missing
# This allows us to use the simulation classes even if real instrument support is unavailable

try:
    from .keithley6430 import Keithley6430
except ImportError:
    # Define a placeholder class for documentation purposes
    class Keithley6430:
        """This class requires additional dependencies that are not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Keithley6430 requires external dependencies that are not installed.")

try:
    from .keithley4200 import Keithley4200
except ImportError:
    # Define a placeholder class for documentation purposes
    class Keithley4200:
        """This class requires additional dependencies that are not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Keithley4200 requires external dependencies that are not installed.")

# These should always be available as they're part of the base package
from .instdebug import InstDebug
from .instcustom import InstCustom

# Note: instrument_factory is imported when needed, not here
# This avoids circular dependencies
