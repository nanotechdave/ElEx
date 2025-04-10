import sys
import os
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                           QComboBox, QGroupBox, QFormLayout, QSpinBox,
                           QDoubleSpinBox, QLineEdit, QListWidget, QCheckBox,
                           QDialog, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from app.gui.electrode_matrix import ElectrodeMatrixWidget

class MeasurementSettingsWindow(QDialog):
    """Window for configuring measurement settings for experiments"""
    
    def __init__(self, experiment_type, mapping, callback=None, parent=None):
        super().__init__(parent)
        
        self.experiment_type = experiment_type
        self.mapping = mapping
        self.callback = callback
        
        self.setWindowTitle(f"{experiment_type} Settings")
        self.setMinimumSize(600, 500)
        
        self.initUI()
            
    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different setting categories
        self.tab_widget = QTabWidget()
        
        # Always include common settings
        self.create_common_settings_tab()
        
        # Add experiment-specific settings tab
        if self.experiment_type == "IVMeasurement":
            self.create_iv_settings_tab()
        elif self.experiment_type == "ConductivityMatrix":
            self.create_conductivity_matrix_settings_tab()
        elif self.experiment_type == "PulseMeasurement":
            self.create_pulse_settings_tab()
        elif self.experiment_type == "MemoryCapacity":
            self.create_memory_capacity_settings_tab()
        elif self.experiment_type == "NoiseMeasurement":
            self.create_noise_settings_tab()
        # Add more experiment tabs as needed
        
        main_layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        run_btn = QPushButton("Run Experiment")
        run_btn.clicked.connect(self.accept_and_run)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(run_btn)
        
        main_layout.addLayout(button_layout)
    
    def create_common_settings_tab(self):
        """Create tab with common settings for all experiments"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Channel selection group
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        
        # Create electrode matrix widget for channel selection
        self.electrode_matrix = ElectrodeMatrixWidget()
        channel_layout.addWidget(self.electrode_matrix)
        
        # Channel role selection
        roles_group = QGroupBox("Channel Roles")
        roles_layout = QFormLayout(roles_group)
        
        # Add inputs for different channel roles
        self.bias_channels = QLineEdit("8")
        self.gnd_channels = QLineEdit("17")
        self.read_v_channels = QLineEdit("8-23")
        self.read_i_channels = QLineEdit("8, 17")
        
        roles_layout.addRow("Bias Channels:", self.bias_channels)
        roles_layout.addRow("Ground Channels:", self.gnd_channels)
        roles_layout.addRow("Read Voltage Channels:", self.read_v_channels)
        roles_layout.addRow("Read Current Channels:", self.read_i_channels)
        
        # Connect electrode matrix selection to update channel lists
        self.electrode_matrix.bias_selection_changed.connect(self.update_bias_channels)
        self.electrode_matrix.gnd_selection_changed.connect(self.update_gnd_channels)
        self.electrode_matrix.read_selection_changed.connect(self.update_read_channels)
        
        channel_layout.addWidget(roles_group)
        layout.addWidget(channel_group)
        
        # Add sample time setting (common to most experiments)
        timing_group = QGroupBox("Timing Settings")
        timing_layout = QFormLayout(timing_group)
        
        self.sample_time = QDoubleSpinBox()
        self.sample_time.setRange(0.001, 10.0)
        self.sample_time.setValue(0.01)
        self.sample_time.setSuffix(" s")
        
        self.n_reps_avg = QSpinBox()
        self.n_reps_avg.setRange(1, 100)
        self.n_reps_avg.setValue(10)
        
        timing_layout.addRow("Sample Time:", self.sample_time)
        timing_layout.addRow("Repetitions for Averaging:", self.n_reps_avg)
        
        layout.addWidget(timing_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Common Settings")
    
    def create_iv_settings_tab(self):
        """Create tab with IV measurement specific settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Voltage settings
        voltage_group = QGroupBox("Voltage Settings")
        voltage_layout = QFormLayout(voltage_group)
        
        self.start_voltage = QDoubleSpinBox()
        self.start_voltage.setRange(0.0, 10.0)
        self.start_voltage.setValue(0.01)
        self.start_voltage.setSuffix(" V")
        
        self.end_voltage = QDoubleSpinBox()
        self.end_voltage.setRange(0.0, 10.0)
        self.end_voltage.setValue(4.0)
        self.end_voltage.setSuffix(" V")
        
        self.voltage_step = QDoubleSpinBox()
        self.voltage_step.setRange(0.001, 1.0)
        self.voltage_step.setValue(0.01)
        self.voltage_step.setSuffix(" V")
        
        voltage_layout.addRow("Start Voltage:", self.start_voltage)
        voltage_layout.addRow("End Voltage:", self.end_voltage)
        voltage_layout.addRow("Voltage Step:", self.voltage_step)
        
        layout.addWidget(voltage_group)
        
        # Compliance settings
        compliance_group = QGroupBox("Compliance Settings")
        compliance_layout = QFormLayout(compliance_group)
        
        self.g_stop = QDoubleSpinBox()
        self.g_stop.setRange(0.0, 200.0)
        self.g_stop.setValue(1.25)
        
        self.g_interval = QDoubleSpinBox()
        self.g_interval.setRange(0.0, 10.0)
        self.g_interval.setValue(0.05)
        
        self.g_points = QSpinBox()
        self.g_points.setRange(1, 100)
        self.g_points.setValue(10)
        
        self.float_at_end = QCheckBox()
        self.float_at_end.setChecked(True)
        
        compliance_layout.addRow("G Stop:", self.g_stop)
        compliance_layout.addRow("G Interval:", self.g_interval)
        compliance_layout.addRow("G Points:", self.g_points)
        compliance_layout.addRow("Float at End:", self.float_at_end)
        
        layout.addWidget(compliance_group)
        
        # Waveform settings
        waveform_group = QGroupBox("Waveform Settings")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_type = QComboBox()
        self.waveform_type.addItems(["Standard", "Constant", "Step", "Sine", "Triangle"])
        
        waveform_params_layout = QFormLayout()
        
        self.waveform_amplitude = QDoubleSpinBox()
        self.waveform_amplitude.setRange(0.0, 10.0)
        self.waveform_amplitude.setValue(0.3)
        self.waveform_amplitude.setSuffix(" V")
        
        self.waveform_dc_bias = QDoubleSpinBox()
        self.waveform_dc_bias.setRange(0.0, 10.0)
        self.waveform_dc_bias.setValue(0.5)
        self.waveform_dc_bias.setSuffix(" V")
        
        self.waveform_frequency = QDoubleSpinBox()
        self.waveform_frequency.setRange(0.1, 10.0)
        self.waveform_frequency.setValue(1.0)
        self.waveform_frequency.setSuffix(" Hz")
        
        self.waveform_periods = QSpinBox()
        self.waveform_periods.setRange(1, 1000)
        self.waveform_periods.setValue(100)
        
        self.waveform_total_time = QDoubleSpinBox()
        self.waveform_total_time.setRange(1.0, 1000.0)
        self.waveform_total_time.setValue(100.0)
        self.waveform_total_time.setSuffix(" s")
        
        waveform_params_layout.addRow("Amplitude:", self.waveform_amplitude)
        waveform_params_layout.addRow("DC Bias:", self.waveform_dc_bias)
        waveform_params_layout.addRow("Frequency:", self.waveform_frequency)
        waveform_params_layout.addRow("Periods:", self.waveform_periods)
        waveform_params_layout.addRow("Total Time:", self.waveform_total_time)
        
        waveform_layout.addWidget(self.waveform_type)
        waveform_layout.addLayout(waveform_params_layout)
        
        layout.addWidget(waveform_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "IV Settings")
    
    def create_conductivity_matrix_settings_tab(self):
        """Create tab with conductivity matrix specific settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Measurement settings
        measurement_group = QGroupBox("Measurement Settings")
        measurement_layout = QFormLayout(measurement_group)
        
        self.v_read = QDoubleSpinBox()
        self.v_read.setRange(0.01, 1.0)
        self.v_read.setValue(0.05)
        self.v_read.setSuffix(" V")
        
        measurement_layout.addRow("Read Voltage:", self.v_read)
        
        layout.addWidget(measurement_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Conductivity Matrix Settings")
    
    def create_pulse_settings_tab(self):
        """Create tab with pulse measurement specific settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Pulse settings
        pulse_group = QGroupBox("Pulse Settings")
        pulse_layout = QFormLayout(pulse_group)
        
        self.pre_pulse_time = QDoubleSpinBox()
        self.pre_pulse_time.setRange(0.1, 100.0)
        self.pre_pulse_time.setValue(10.0)
        self.pre_pulse_time.setSuffix(" s")
        
        self.pulse_time = QDoubleSpinBox()
        self.pulse_time.setRange(0.1, 100.0)
        self.pulse_time.setValue(10.0)
        self.pulse_time.setSuffix(" s")
        
        self.post_pulse_time = QDoubleSpinBox()
        self.post_pulse_time.setRange(0.1, 1000.0)
        self.post_pulse_time.setValue(300.0)
        self.post_pulse_time.setSuffix(" s")
        
        self.pulse_voltage = QDoubleSpinBox()
        self.pulse_voltage.setRange(0.1, 10.0)
        self.pulse_voltage.setValue(1.0)
        self.pulse_voltage.setSuffix(" V")
        
        self.interpulse_voltage = QDoubleSpinBox()
        self.interpulse_voltage.setRange(0.01, 1.0)
        self.interpulse_voltage.setValue(0.05)
        self.interpulse_voltage.setSuffix(" V")
        
        pulse_layout.addRow("Pre-Pulse Time:", self.pre_pulse_time)
        pulse_layout.addRow("Pulse Time:", self.pulse_time)
        pulse_layout.addRow("Post-Pulse Time:", self.post_pulse_time)
        pulse_layout.addRow("Pulse Voltage:", self.pulse_voltage)
        pulse_layout.addRow("Interpulse Voltage:", self.interpulse_voltage)
        
        layout.addWidget(pulse_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Pulse Settings")
    
    def create_memory_capacity_settings_tab(self):
        """Create tab with memory capacity specific settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Memory capacity settings
        mc_group = QGroupBox("Memory Capacity Settings")
        mc_layout = QFormLayout(mc_group)
        
        self.n_samples = QSpinBox()
        self.n_samples.setRange(100, 10000)
        self.n_samples.setValue(3000)
        
        self.v_read_mc = QDoubleSpinBox()
        self.v_read_mc.setRange(0.01, 1.0)
        self.v_read_mc.setValue(0.05)
        self.v_read_mc.setSuffix(" V")
        
        self.crit_bias = QDoubleSpinBox()
        self.crit_bias.setRange(0.1, 10.0)
        self.crit_bias.setValue(0.8)
        self.crit_bias.setSuffix(" V")
        
        self.crit_amp = QDoubleSpinBox()
        self.crit_amp.setRange(0.1, 5.0)
        self.crit_amp.setValue(0.3)
        self.crit_amp.setSuffix(" V")
        
        mc_layout.addRow("Number of Samples:", self.n_samples)
        mc_layout.addRow("Read Voltage:", self.v_read_mc)
        mc_layout.addRow("Critical Bias:", self.crit_bias)
        mc_layout.addRow("Critical Amplitude:", self.crit_amp)
        
        layout.addWidget(mc_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Memory Capacity Settings")
    
    def create_noise_settings_tab(self):
        """Create tab with noise measurement specific settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Noise settings
        noise_group = QGroupBox("Noise Measurement Settings")
        noise_layout = QFormLayout(noise_group)
        
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1.0, 3600.0)
        self.duration.setValue(360.0)
        self.duration.setSuffix(" s")
        
        self.bias_voltage = QDoubleSpinBox()
        self.bias_voltage.setRange(0.01, 1.0)
        self.bias_voltage.setValue(0.05)
        self.bias_voltage.setSuffix(" V")
        
        noise_layout.addRow("Duration:", self.duration)
        noise_layout.addRow("Bias Voltage:", self.bias_voltage)
        
        layout.addWidget(noise_group)
        
        # Add the tab
        self.tab_widget.addTab(tab, "Noise Settings")
    
    def update_bias_channels(self, channels):
        """Update bias channels text from electrode matrix selection"""
        self.bias_channels.setText(", ".join(map(str, channels)))
    
    def update_gnd_channels(self, channels):
        """Update ground channels text from electrode matrix selection"""
        self.gnd_channels.setText(", ".join(map(str, channels)))
    
    def update_read_channels(self, channels):
        """Update read channels text from electrode matrix selection"""
        self.read_v_channels.setText(", ".join(map(str, channels)))
        # By default, also set read current channels to the same as read voltage
        self.read_i_channels.setText(", ".join(map(str, channels[:2])) if len(channels) >= 2 else "")
    
    def parse_channel_list(self, text):
        """Parse a string of comma-separated channel numbers, including ranges"""
        channels = []
        parts = text.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle ranges like '8-23'
                start, end = map(int, part.split('-'))
                channels.extend(range(start, end + 1))
            else:
                try:
                    # Handle single numbers
                    channels.append(int(part))
                except ValueError:
                    # Skip invalid entries
                    pass
        
        return channels
    
    def get_settings(self):
        """Get all settings as a dictionary"""
        settings = {
            "mask_to_bias": self.parse_channel_list(self.bias_channels.text()),
            "mask_to_gnd": self.parse_channel_list(self.gnd_channels.text()),
            "mask_to_read_v": self.parse_channel_list(self.read_v_channels.text()),
            "mask_to_read_i": self.parse_channel_list(self.read_i_channels.text()),
            "sample_time": self.sample_time.value(),
            "n_reps_avg": self.n_reps_avg.value()
        }
        
        # Add experiment-specific settings
        if self.experiment_type == "IVMeasurement":
            settings.update({
                "start_voltage": self.start_voltage.value(),
                "end_voltage": self.end_voltage.value(),
                "voltage_step": self.voltage_step.value(),
                "g_stop": self.g_stop.value(),
                "g_interval": self.g_interval.value(),
                "g_points": self.g_points.value(),
                "float_at_end": self.float_at_end.isChecked(),
                "waveform_type": self.waveform_type.currentText(),
                "waveform_amplitude": self.waveform_amplitude.value(),
                "waveform_dc_bias": self.waveform_dc_bias.value(),
                "waveform_frequency": self.waveform_frequency.value(),
                "waveform_periods": self.waveform_periods.value(),
                "waveform_total_time": self.waveform_total_time.value()
            })
        elif self.experiment_type == "ConductivityMatrix":
            settings.update({
                "v_read": self.v_read.value()
            })
        elif self.experiment_type == "PulseMeasurement":
            settings.update({
                "pre_pulse_time": self.pre_pulse_time.value(),
                "pulse_time": self.pulse_time.value(),
                "post_pulse_time": self.post_pulse_time.value(),
                "pulse_voltage": [self.pulse_voltage.value()],
                "interpulse_voltage": self.interpulse_voltage.value()
            })
        elif self.experiment_type == "MemoryCapacity":
            settings.update({
                "n_samples": self.n_samples.value(),
                "v_read": self.v_read_mc.value(),
                "crit_bias": self.crit_bias.value(),
                "crit_amp": self.crit_amp.value()
            })
        elif self.experiment_type == "NoiseMeasurement":
            settings.update({
                "duration": self.duration.value(),
                "bias_voltage": self.bias_voltage.value()
            })
        
        return settings
    
    def accept_and_run(self):
        """Accept the dialog and run the experiment with the current settings"""
        settings = self.get_settings()
        
        if self.callback:
            self.callback(self.experiment_type, settings)
        
        self.accept() 