import os
import importlib
from pathlib import Path
from typing import Dict, Type, List

from PyQt6 import QtCore, QtWidgets
from PyQt6.uic import loadUi

# Cached widgets by type
_widget_registry = {}


def register_widget(widget_type: str, widget_class: Type[QtWidgets.QWidget]):
    """
    Register a widget class for a specific type
    
    Args:
        widget_type: Type identifier for the widget
        widget_class: The widget class to register
    """
    _widget_registry[widget_type] = widget_class


def get_widget_class(widget_type: str) -> Type[QtWidgets.QWidget]:
    """
    Get a widget class for a specific type
    
    Args:
        widget_type: Type identifier for the widget
        
    Returns:
        The widget class
    """
    if widget_type not in _widget_registry:
        raise ValueError(f"Widget type {widget_type} not registered")
    
    return _widget_registry[widget_type]


def create_widget(widget_type: str, parent=None, **kwargs) -> QtWidgets.QWidget:
    """
    Create a widget of the specified type
    
    Args:
        widget_type: Type identifier for the widget
        parent: Parent widget
        **kwargs: Additional parameters to pass to the widget constructor
        
    Returns:
        The created widget
    """
    widget_class = get_widget_class(widget_type)
    return widget_class(parent=parent, **kwargs)


def discover_widgets(directories: List[str] = None):
    """
    Discover and register widgets from the specified directories
    
    Args:
        directories: List of directories to search for widgets
    """
    if directories is None:
        # Default to the arc2custom widgets directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        directories = [
            Path(project_root) / "app" / "instruments" / "arc2custom" / "widgets",
            Path(project_root) / "app" / "experiments" / "widgets"
        ]
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for widget_file in Path(directory).glob("*_widget.py"):
            module_name = widget_file.stem
            # Determine the module path based on the directory
            if "arc2custom" in str(directory):
                module_path = f"app.instruments.arc2custom.widgets.{module_name}"
            else:
                module_path = f"app.experiments.widgets.{module_name}"
                
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Look for widget classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, QtWidgets.QWidget) and
                        attr.__module__ == module.__name__):
                        # This looks like a widget class
                        widget_type = module_name.replace("_widget", "")
                        register_widget(widget_type, attr)
            except ImportError as e:
                print(f"Error loading widget module {module_path}: {e}")


def load_ui_widget(ui_file: str, base_class: Type[QtWidgets.QWidget] = None) -> Type[QtWidgets.QWidget]:
    """
    Load a widget from a UI file
    
    Args:
        ui_file: Path to the UI file
        base_class: Base class for the widget
        
    Returns:
        The widget class
    """
    if base_class is None:
        base_class = QtWidgets.QWidget
        
    class UiWidget(base_class):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            loadUi(ui_file, self)
            
            # Apply any additional initialization from kwargs
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    return UiWidget


def load_ui_from_string(ui_string: str, base_class: Type[QtWidgets.QWidget] = None) -> Type[QtWidgets.QWidget]:
    """
    Load a widget from a UI string
    
    Args:
        ui_string: UI definition as a string
        base_class: Base class for the widget
        
    Returns:
        The widget class
    """
    if base_class is None:
        base_class = QtWidgets.QWidget
        
    # Save the UI string to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ui', delete=False) as f:
        f.write(ui_string.encode('utf-8'))
        ui_file = f.name
    
    # Load the UI from the temporary file
    widget_class = load_ui_widget(ui_file, base_class)
    
    # Clean up the temporary file
    os.unlink(ui_file)
    
    return widget_class


# Initialize the widget registry
discover_widgets() 