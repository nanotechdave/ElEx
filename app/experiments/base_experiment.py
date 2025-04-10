import abc
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from PyQt6 import QtCore, QtWidgets


class BaseExperimentOperation(QtCore.QThread):
    """
    A background operation for experiments. This is similar to BaseOperation in arc2custom.
    
    All experiment operations should inherit from this class to provide standardized
    behavior and signal handling. When a thread based on this operation is started, 
    the `run` method will be called and must be implemented by all subclasses.
    """
    
    operationFinished = QtCore.pyqtSignal()
    operationProgress = QtCore.pyqtSignal(float, str)  # progress percentage, status message
    
    def __init__(self, parent=None):
        if not isinstance(parent, BaseExperiment):
            raise TypeError("Parent is not a subclass of `BaseExperiment`")
        super().__init__(parent=parent)
        
        self._logger = parent.logger
        self.parent = parent
    
    @property
    def instrument(self):
        """
        Reference to the currently connected instrument (if any)
        """
        return self.parent.instrument
    
    @property
    def settings(self):
        """
        Reference to the experiment settings
        """
        return self.parent.settings
    
    @property
    def logger(self):
        """
        Returns the logger for this operation
        """
        return self._logger
    
    @abc.abstractmethod
    def run(self):
        """
        Implement the logic of the operation by overriding this method
        """
        pass


class BaseExperiment(QtWidgets.QWidget):
    """
    Base class for all experiment modules. Custom experiment modules should
    inherit from this class to provide standardized behavior and signal handling.
    
    This class provides common functionality for experiment modules, including
    signals, instrument management, settings, and UI interactions.
    """
    
    # Signals
    experimentStarted = QtCore.pyqtSignal()
    experimentFinished = QtCore.pyqtSignal()
    experimentProgress = QtCore.pyqtSignal(float, str)  # progress percentage, status message
    experimentError = QtCore.pyqtSignal(str)  # error message
    
    # Class attributes
    description = "Base experiment class"
    required_settings = {}
    
    def __init__(self, instrument, name, session=None, parent=None):
        """
        Initialize the experiment module
        
        Args:
            instrument: Reference to the instrument to use
            name: Name of the experiment
            session: Session information (if applicable)
            parent: Parent widget
        """
        super().__init__(parent=parent)
        
        self.name = name
        self._instrument = instrument
        self._session = session
        self._logger = logging.getLogger(f"experiment.{name}")
        
        # Initialize UI
        self.setupUi()
    
    def setupUi(self):
        """
        Set up the UI components for this experiment.
        This should be overridden by subclasses to create custom UIs.
        
        The default implementation creates a simple layout with a label
        showing the experiment's description.
        """
        layout = QtWidgets.QVBoxLayout()
        
        # Add description label
        description_label = QtWidgets.QLabel(self.description)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        
        # Set the layout
        self.setLayout(layout)
    
    @property
    def instrument(self):
        """
        Reference to the instrument
        """
        return self._instrument
    
    @property
    def session(self):
        """
        Reference to the session
        """
        return self._session
    
    @property
    def logger(self):
        """
        Returns the logger for this experiment
        """
        return self._logger
    
    def exportToJson(self, fname):
        """
        Export the experiment settings to a JSON file
        
        Args:
            fname: Name of the file to export to
        """
        settings_dict = {}
        
        # Export settings
        if hasattr(self, 'settings'):
            for key, value in vars(self.settings).items():
                # Skip private attributes
                if not key.startswith('_'):
                    settings_dict[key] = value
        
        # Add module information
        settings_dict['module'] = self.__class__.__module__
        settings_dict['class'] = self.__class__.__name__
        
        # Write to file
        with open(fname, 'w') as f:
            json.dump(settings_dict, f, indent=4)
    
    def loadFromJson(self, fname):
        """
        Load experiment settings from a JSON file
        
        Args:
            fname: Name of the file to load from
        """
        with open(fname, 'r') as f:
            settings_dict = json.load(f)
        
        # Apply settings
        if hasattr(self, 'settings'):
            for key, value in settings_dict.items():
                # Skip module information and private attributes
                if key not in ['module', 'class'] and not key.startswith('_'):
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
    
    @abc.abstractmethod
    def run(self):
        """
        Run the experiment. This method should be implemented by all subclasses.
        """
        pass


def load_experiment(experiment_name, experiment_path=None):
    """
    Load an experiment module by name
    
    Args:
        experiment_name: Name of the experiment to load
        experiment_path: Optional path to the experiment module
        
    Returns:
        The experiment class
    """
    if experiment_path is None:
        # Try to find the experiment in the standard locations
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        experiment_dir = Path(project_root) / "app" / "instruments" / "arc2custom" / "experiments"
        
        # Check if the experiment exists
        for module_file in experiment_dir.glob("*.py"):
            if module_file.stem.lower() == experiment_name.lower():
                experiment_path = f"app.instruments.arc2custom.experiments.{module_file.stem}"
                break
    
    if experiment_path is None:
        raise ValueError(f"Could not find experiment {experiment_name}")
    
    # Import the module
    module = importlib.import_module(experiment_path)
    
    # Try to find the experiment class
    # First, look for a class with the same name as the experiment
    class_name = ''.join(word.capitalize() for word in experiment_name.split('_'))
    if hasattr(module, class_name):
        return getattr(module, class_name)
    
    # If not found, look for any class that inherits from BaseExperiment
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and 
            attr.__module__ == module.__name__ and
            hasattr(attr, '__bases__') and
            any('Experiment' in base.__name__ for base in attr.__bases__)):
            return attr
    
    raise ValueError(f"Could not find experiment class in module {experiment_path}") 