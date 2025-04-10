"""
IV Measurement Experiment Template

This module demonstrates how to create an experiment using the new
modular experiment framework.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from PyQt6 import QtCore, QtWidgets

from app.experiments.base_experiment import BaseExperiment, BaseExperimentOperation
from app.instruments.arc2custom import dparclib as dparc
from app.instruments.arc2custom import dplib as dp
from app.instruments.arc2custom import measurementsettings


class IVMeasurementSettings(measurementsettings.MeasurementSettings):
    """
    Settings specific to IV measurement experiments
    """
    
    def __init__(self):
        super().__init__()
        
        # IV-specific settings
        self.v_start = -0.5  # Starting voltage
        self.v_stop = 0.5    # Ending voltage
        self.v_step = 0.01   # Voltage step
        self.delay = 0.1     # Delay between measurements (s)
        self.compliance = 1e-6  # Compliance current (A)
        self.bidirectional = True  # Whether to sweep in both directions


class IVSweepOperation(BaseExperimentOperation):
    """
    Operation to perform an IV sweep
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Initialize data storage
        self.voltages = []
        self.currents = {}  # Dictionary mapping channels to current lists
    
    def run(self):
        """
        Run the IV sweep operation
        """
        try:
            settings = self.settings
            
            # Create voltage sweep points
            v_points = np.arange(settings.v_start, settings.v_stop, settings.v_step)
            if settings.bidirectional:
                v_points = np.concatenate([v_points, np.flip(v_points)])
            
            # Initialize storage
            self.voltages = []
            self.currents = {ch: [] for ch in settings.mask}
            
            # Set up the instrument
            dparc.setAllChannelsToFloat(self.instrument)
            self.instrument.connect_to_gnd(settings.mask_to_gnd)
            
            # Perform the sweep
            total_points = len(v_points)
            for i, v in enumerate(v_points):
                # Update progress
                progress = 100.0 * i / total_points
                self.operationProgress.emit(progress, f"Measuring at {v:.3f}V")
                
                # Create bias mask (all bias channels set to the same voltage)
                bias_mask = [(ch, v) for ch in settings.mask_to_bias]
                
                # Set bias voltages
                self.instrument.bias_channels(bias_mask)
                
                # Wait for settling
                time.sleep(settings.delay)
                
                # Measure
                timestamp = time.time()
                voltage_data = self.instrument.vread_channels(settings.mask_to_read_v)
                current_data = self.instrument.iread_channels(settings.mask_to_read_i)
                
                # Store data
                self.voltages.append(v)
                for ch in settings.mask:
                    if ch in settings.mask_to_read_i:
                        self.currents[ch].append(current_data[ch])
                    else:
                        # If not a current measurement channel, store NaN
                        self.currents[ch].append(float('nan'))
            
            # Clear instrument state
            dparc.setAllChannelsToFloat(self.instrument)
            
            # Signal completion
            self.operationFinished.emit()
            
        except Exception as e:
            self.logger.error(f"Error during IV sweep: {e}")
            dparc.setAllChannelsToFloat(self.instrument)  # Safety measure
            raise


class IVMeasurementExperiment(BaseExperiment):
    """
    IV Measurement experiment module
    
    This experiment performs an IV sweep between selected electrodes,
    ramping voltage while measuring current.
    """
    
    # Class attributes
    description = "Performs an IV sweep between selected electrodes, ramping voltage while measuring current."
    required_settings = {
        "v_start": (-2.0, 2.0, 0.01),  # min, max, step
        "v_stop": (-2.0, 2.0, 0.01),
        "v_step": (0.001, 0.1, 0.001),
        "delay": (0.001, 1.0, 0.001),
        "compliance": (1e-9, 1e-3, 1e-9),
        "bidirectional": (True, False)
    }
    
    def __init__(self, instrument, name, session=None, parent=None):
        """
        Initialize the IV measurement experiment
        
        Args:
            instrument: Reference to the instrument to use
            name: Name of the experiment
            session: Session information (if applicable)
            parent: Parent widget
        """
        super().__init__(instrument, name, session, parent)
        
        # Create settings
        self.settings = IVMeasurementSettings()
        
        # Initialize data storage
        self.results = None
    
    def setupUi(self):
        """Set up the UI components for this experiment"""
        layout = QtWidgets.QVBoxLayout()
        
        # Add description
        description_label = QtWidgets.QLabel(self.description)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        
        # Create settings form
        form_layout = QtWidgets.QFormLayout()
        
        # Add settings controls
        self.v_start_spinbox = QtWidgets.QDoubleSpinBox()
        self.v_start_spinbox.setRange(-2.0, 2.0)
        self.v_start_spinbox.setSingleStep(0.01)
        self.v_start_spinbox.setValue(self.settings.v_start)
        self.v_start_spinbox.valueChanged.connect(self._update_settings)
        form_layout.addRow("Start Voltage (V):", self.v_start_spinbox)
        
        self.v_stop_spinbox = QtWidgets.QDoubleSpinBox()
        self.v_stop_spinbox.setRange(-2.0, 2.0)
        self.v_stop_spinbox.setSingleStep(0.01)
        self.v_stop_spinbox.setValue(self.settings.v_stop)
        self.v_stop_spinbox.valueChanged.connect(self._update_settings)
        form_layout.addRow("Stop Voltage (V):", self.v_stop_spinbox)
        
        self.v_step_spinbox = QtWidgets.QDoubleSpinBox()
        self.v_step_spinbox.setRange(0.001, 0.1)
        self.v_step_spinbox.setSingleStep(0.001)
        self.v_step_spinbox.setValue(self.settings.v_step)
        self.v_step_spinbox.valueChanged.connect(self._update_settings)
        form_layout.addRow("Voltage Step (V):", self.v_step_spinbox)
        
        self.delay_spinbox = QtWidgets.QDoubleSpinBox()
        self.delay_spinbox.setRange(0.001, 1.0)
        self.delay_spinbox.setSingleStep(0.001)
        self.delay_spinbox.setValue(self.settings.delay)
        self.delay_spinbox.valueChanged.connect(self._update_settings)
        form_layout.addRow("Delay (s):", self.delay_spinbox)
        
        self.compliance_spinbox = QtWidgets.QDoubleSpinBox()
        self.compliance_spinbox.setRange(1e-9, 1e-3)
        self.compliance_spinbox.setSingleStep(1e-9)
        self.compliance_spinbox.setValue(self.settings.compliance)
        self.compliance_spinbox.setDecimals(9)
        self.compliance_spinbox.valueChanged.connect(self._update_settings)
        form_layout.addRow("Compliance (A):", self.compliance_spinbox)
        
        self.bidirectional_checkbox = QtWidgets.QCheckBox()
        self.bidirectional_checkbox.setChecked(self.settings.bidirectional)
        self.bidirectional_checkbox.stateChanged.connect(self._update_settings)
        form_layout.addRow("Bidirectional:", self.bidirectional_checkbox)
        
        layout.addLayout(form_layout)
        
        # Add run button
        self.run_button = QtWidgets.QPushButton("Run Experiment")
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)
        
        # Add progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Set the layout
        self.setLayout(layout)
    
    def _update_settings(self):
        """Update settings from UI controls"""
        self.settings.v_start = self.v_start_spinbox.value()
        self.settings.v_stop = self.v_stop_spinbox.value()
        self.settings.v_step = self.v_step_spinbox.value()
        self.settings.delay = self.delay_spinbox.value()
        self.settings.compliance = self.compliance_spinbox.value()
        self.settings.bidirectional = self.bidirectional_checkbox.isChecked()
    
    def run(self):
        """Run the IV measurement experiment"""
        if not self.instrument:
            self.status_label.setText("Error: No instrument connected")
            return
        
        # Disable UI during experiment
        self.run_button.setEnabled(False)
        self.status_label.setText("Running...")
        
        # Signal experiment start
        self.experimentStarted.emit()
        
        # Create and run the operation
        self.operation = IVSweepOperation(self)
        self.operation.operationProgress.connect(self._handle_progress)
        self.operation.operationFinished.connect(self._handle_finished)
        self.operation.start()
    
    def _handle_progress(self, progress, message):
        """Handle progress updates from the operation"""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)
        
        # Forward the progress
        self.experimentProgress.emit(progress, message)
    
    def _handle_finished(self):
        """Handle operation completion"""
        # Store results
        self.results = {
            "voltages": self.operation.voltages,
            "currents": self.operation.currents
        }
        
        # Update UI
        self.progress_bar.setValue(100)
        self.status_label.setText("Completed")
        self.run_button.setEnabled(True)
        
        # Signal experiment completion
        self.experimentFinished.emit()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the experiment results
        
        Returns:
            Dictionary containing the experiment results
        """
        return self.results


# For testing
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create the experiment widget
    experiment = IVMeasurementExperiment(None, "IV Measurement")
    experiment.show()
    
    sys.exit(app.exec()) 