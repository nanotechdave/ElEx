# instrument_base.py
from abc import ABC, abstractmethod

class Instrument(ABC):
    @abstractmethod
    def read_voltage(self):
        pass

    @abstractmethod
    def read_current(self):
        pass

    @abstractmethod
    def set_voltage(self, voltage: float):
        pass

    @abstractmethod
    def set_current(self, current: float):
        pass
