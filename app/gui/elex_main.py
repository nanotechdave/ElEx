import sys
import os
from pathlib import Path
import multiprocessing
from multiprocessing import Queue

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                            QComboBox, QGroupBox, QFormLayout, QStatusBar,
                            QTabWidget, QSplitter, QToolBar, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSettings, QSize
from PyQt6.QtGui import QFont, QIcon

# Import local modules
from app.gui.plot_widget import RealTimePlotWidget
from app.gui.experiment_runner import ExperimentRunner
from app.gui.electrode_matrix import ElectrodeMatrixWidget
from app.gui.measurement_settings_window import MeasurementSettingsWindow

# Import experiment management modules
from app.experiments.experiment_launcher import ExperimentLauncherProcess
from app.experiments.base_experiment import BaseExperiment

class ElexMainWindow(QMainWindow):
    """Main window for the ElEx application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ElEx - Electronic Experiments")
        self.setMinimumSize(1200, 800)
        
        # Initialize state variables
        self.current_instrument = None
        self.current_mapping = None
        self.current_experiment = None
        
        # Create a queue for sharing results between processes
        self.results_queue = Queue()
        
        # Timer for updating plots
        self.update_timer = None
        
        # Initialize settings
        self.settings = QSettings("ElEx", "ElectronicExperiments")
        self.restore_window_geometry()
        
        # Initialize experiment launcher
        self.experiment_launcher = ExperimentLauncherProcess()
        self.experiment_launcher.start()
        
        # Initialize available experiments list
        self.available_experiments = []
        self.refresh_available_experiments()
        
        self.initUI()
        
    def initUI(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("ElEx - Electronic Experiments")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        header_label.setFont(font)
        main_layout.addWidget(header_label)
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.create_instrument_page()
        self.create_mapping_page()
        self.create_experiment_page()
        self.create_results_page()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Start on instrument selection page
        self.stacked_widget.setCurrentIndex(0)
    
    def create_instrument_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title_label = QLabel("Instrument Selection")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Instrument detection group
        detect_group = QGroupBox("Available Instruments")
        detect_layout = QVBoxLayout(detect_group)
        
        # Add refresh button
        refresh_btn = QPushButton("Scan for Instruments")
        refresh_btn.clicked.connect(self.scan_instruments)
        detect_layout.addWidget(refresh_btn)
        
        # Add instrument list (will be populated by scan_instruments)
        self.instrument_combo = QComboBox()
        detect_layout.addWidget(self.instrument_combo)
        
        # Add InstDebug for debugging
        self.instrument_combo.addItem("InstDebug (Simulation)")
        
        layout.addWidget(detect_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add next button
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.on_instrument_selected)
        layout.addWidget(next_btn)
        
        self.stacked_widget.addWidget(page)
    
    def create_mapping_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title_label = QLabel("Mapping Selection")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Mapping selection group
        mapping_group = QGroupBox("Select Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        
        # Add mapping selection combobox
        self.mapping_combo = QComboBox()
        self.populate_mapping_combo()
        mapping_layout.addWidget(self.mapping_combo)
        
        layout.addWidget(mapping_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add navigation buttons
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.on_mapping_selected)
        
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(next_btn)
        layout.addLayout(nav_layout)
        
        self.stacked_widget.addWidget(page)
    
    def create_experiment_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title_label = QLabel("Experiment Selection")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Experiment selection group
        experiment_group = QGroupBox("Select Experiment")
        experiment_layout = QVBoxLayout(experiment_group)
        
        # Add experiment selection combobox
        self.experiment_combo = QComboBox()
        self.populate_experiment_combo()
        experiment_layout.addWidget(self.experiment_combo)
        
        layout.addWidget(experiment_group)
        
        # Add experiment description
        self.experiment_description = QLabel("Select an experiment to see its description")
        self.experiment_description.setWordWrap(True)
        self.experiment_description.setMinimumHeight(100)
        layout.addWidget(self.experiment_description)
        
        # Connect experiment combo to description update
        self.experiment_combo.currentTextChanged.connect(self.update_experiment_description)
        
        # Add spacer
        layout.addStretch()
        
        # Add navigation buttons
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        run_btn = QPushButton("Run Experiment")
        run_btn.clicked.connect(self.run_experiment)
        
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(run_btn)
        layout.addLayout(nav_layout)
        
        self.stacked_widget.addWidget(page)
    
    def create_results_page(self):
        """Create the results page with real-time plots"""
        from app.gui.plot_widget import RealTimePlotWidget
        
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title_label = QLabel("Experiment Results")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Status information
        self.result_status = QLabel("No experiment running")
        layout.addWidget(self.result_status)
        
        # Add real-time plot widget
        self.plot_widget = RealTimePlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Add back button
        back_btn = QPushButton("Back to Experiment Selection")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        layout.addWidget(back_btn)
        
        self.stacked_widget.addWidget(page)
    
    def scan_instruments(self):
        """Scans for connected instruments"""
        self.status_bar.showMessage("Scanning for instruments...")
        
        # Clear the combo box except for the debug instrument
        self.instrument_combo.clear()
        self.instrument_combo.addItem("InstDebug (Simulation)")
        
        # Here we would actually scan for instruments
        # For now, just add a dummy instrument
        self.instrument_combo.addItem("DummyInstrument (Not Connected)")
        
        self.status_bar.showMessage("Scan complete", 3000)
    
    def populate_mapping_combo(self):
        """Populates the mapping combo box with available mappings"""
        # Clear the combo box
        self.mapping_combo.clear()
        
        # Get mappings from the mappings directory
        mappings_dir = Path(project_root) / "app" / "instruments" / "arc2custom" / "mappings"
        
        if mappings_dir.exists():
            for mapping_file in mappings_dir.glob("*.toml"):
                self.mapping_combo.addItem(mapping_file.stem)
    
    def populate_experiment_combo(self):
        """Populates the experiment combo box with available experiments"""
        # Clear the combo box
        self.experiment_combo.clear()
        
        # Add available experiments
        self.experiment_combo.addItems([
            "IV Measurement",
            "Memory Capacity",
            "Noise Measurement",
            "Activation Pattern",
            "Pulse Measurement",
            "Conductivity Matrix"
        ])
    
    def update_experiment_description(self, experiment_name):
        """Updates the experiment description based on the selected experiment"""
        descriptions = {
            "IV Measurement": "Performs an IV sweep between selected electrodes, ramping voltage while measuring current.",
            "Memory Capacity": "Measures memory capacity by applying specific voltage patterns.",
            "Noise Measurement": "Measures noise characteristics across electrodes.",
            "Activation Pattern": "Creates connections between electrodes in a specified sequence.",
            "Pulse Measurement": "Applies voltage pulses and measures response.",
            "Conductivity Matrix": "Maps conductivity between all electrode pairs."
        }
        
        self.experiment_description.setText(descriptions.get(experiment_name, "No description available"))
    
    def on_instrument_selected(self):
        """Handle instrument selection"""
        self.current_instrument = self.instrument_combo.currentText()
        self.status_bar.showMessage(f"Selected instrument: {self.current_instrument}")
        
        # Move to mapping selection
        self.stacked_widget.setCurrentIndex(1)
    
    def on_mapping_selected(self):
        """Handle mapping selection"""
        self.current_mapping = self.mapping_combo.currentText()
        self.status_bar.showMessage(f"Selected mapping: {self.current_mapping}")
        
        # Move to experiment selection
        self.stacked_widget.setCurrentIndex(2)
    
    def run_experiment(self):
        """Runs the selected experiment"""
        self.current_experiment = self.experiment_combo.currentText()
        
        # Display summary
        summary = (
            f"Running experiment with the following settings:\n"
            f"  - Instrument: {self.current_instrument}\n"
            f"  - Mapping: {self.current_mapping}\n"
            f"  - Experiment: {self.current_experiment}"
        )
        
        print(summary)
        self.status_bar.showMessage(f"Running {self.current_experiment}...")
        
        # Launch the experiment in a separate process
        self.run_experiment_in_separate_process()
    
    def run_experiment_in_separate_process(self):
        """Runs the experiment in a separate process"""
        # Create a dictionary to map friendly names to actual experiment types
        experiment_mapping = {
            "IV Measurement": "IVMeasurement",
            "Memory Capacity": "MemoryCapacity",
            "Noise Measurement": "NoiseMeasurement",
            "Activation Pattern": "ActivationPattern",
            "Pulse Measurement": "PulseMeasurement",
            "Conductivity Matrix": "ConductivityMatrix",
            "Reservoir Computing": "ReservoirComputing",
            "Tomography": "Tomography",
            "Turn On": "TurnOn"
        }
        
        # Get the experiment type identifier
        experiment_type = experiment_mapping.get(self.current_experiment)
        if not experiment_type:
            self.status_bar.showMessage(f"Unknown experiment type: {self.current_experiment}", 3000)
            return
            
        # First, show the measurement settings window to configure the experiment
        from app.gui.measurement_settings_window import MeasurementSettingsWindow
        
        # Create and show the measurement settings window
        # Pass a callback that will be called when settings are confirmed
        self.settings_window = MeasurementSettingsWindow(
            experiment_type=experiment_type,
            mapping=self.current_mapping,
            callback=self._start_experiment_with_settings
        )
        self.settings_window.show()
    
    def _start_experiment_with_settings(self, experiment_type, settings):
        """Start the experiment with the provided settings"""
        # Create a new process for the experiment runner
        from app.gui.experiment_runner import run_experiment_process
        
        # Reset plot data
        self.plot_widget.reset()
        self.plot_widget.set_title(f"{experiment_type} Experiment")
        
        # Show the results page
        self.stacked_widget.setCurrentIndex(3)
        self.result_status.setText(f"Starting {experiment_type} experiment...")
        
        # Start the process
        process = multiprocessing.Process(
            target=run_experiment_process,
            args=(experiment_type, settings, self.current_instrument, self.current_mapping, self.results_queue)
        )
        
        process.start()
        self.status_bar.showMessage(f"Experiment {experiment_type} started in separate process (PID: {process.pid})")
        
        # Start a timer to periodically check experiment status and update plots
        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(lambda: self._check_experiment_updates(process.pid))
        self.experiment_timer.start(100)  # Check every 100ms for real-time updates
    
    def _check_experiment_updates(self, process_id):
        """Check for experiment updates and update the plot"""
        # Check if there are updates in the queue
        try:
            # Process all available updates
            while not self.results_queue.empty():
                try:
                    update = self.results_queue.get_nowait()
                    self._process_experiment_update(update)
                except Exception as e:
                    print(f"Error processing update: {str(e)}")
            
            # Check if the process is still running
            import os
            import signal
            try:
                os.kill(process_id, 0)
                # Process is still running
            except OSError:
                # Process has ended
                self.status_bar.showMessage("Experiment completed", 3000)
                self.experiment_timer.stop()
                self.result_status.setText("Experiment completed")
        except Exception as e:
            print(f"Error checking experiment updates: {str(e)}")
            self.experiment_timer.stop()
    
    def _process_experiment_update(self, update):
        """Process an experiment update and update the plot"""
        try:
            if not isinstance(update, dict):
                print(f"Warning: Received non-dictionary update: {update}")
                return
                
            # Handle status updates
            if update.get('type') == 'status_update':
                status = update.get('status', 'unknown')
                progress = update.get('progress', 0)
                message = update.get('message', '')
                experiment_type = update.get('experiment_type', 'unknown')
                
                # Update status display
                self.result_status.setText(f"{experiment_type}: {message} ({progress}%)")
                return
                
            # Handle data updates
            if update.get('type') == 'data_update':
                data = update.get('data', {})
                experiment_type = update.get('experiment_type', 'unknown')
                
                # Debug print to understand data structure
                if 'timestamps' in data:
                    timestamps_len = len(data['timestamps'])
                    voltage_keys = list(data.get('voltages', {}).keys())
                    
                    if voltage_keys and 'voltages' in data:
                        first_channel = voltage_keys[0]
                        voltage_lens = len(data['voltages'][first_channel])
                        print(f"Data update: timestamps[{timestamps_len}], voltages[{first_channel}][{voltage_lens}]")
                
                # Extract data
                timestamps = data.get('timestamps', [])
                
                # Handle different data formats
                # Format 1: data contains 'voltages', 'currents', etc. dictionaries
                # Each with channel numbers as keys
                if 'voltages' in data and isinstance(data['voltages'], dict):
                    voltages = data['voltages']
                    currents = data.get('currents', {})
                    
                    for channel, voltage_values in voltages.items():
                        channel_str = f"Ch {channel}"
                        
                        # Create voltage series
                        if len(timestamps) == len(voltage_values):
                            self.plot_widget.add_data_series(channel_str, "Voltage", 
                                                            timestamps, voltage_values)
                        else:
                            print(f"Length mismatch: timestamps[{len(timestamps)}] vs voltages[{channel}][{len(voltage_values)}]")
                        
                        # Create current series if available
                        if channel in currents and len(timestamps) == len(currents[channel]):
                            self.plot_widget.add_data_series(channel_str, "Current", 
                                                            timestamps, currents[channel])
                        elif channel in currents:
                            print(f"Length mismatch: timestamps[{len(timestamps)}] vs currents[{channel}][{len(currents[channel])}]")
                            
                # Format 2: older format with 'voltage', 'current' dictionaries
                # Each contains nested dictionaries with channel numbers as keys
                elif 'voltage' in data and isinstance(data['voltage'], dict):
                    for channel, values in data['voltage'].items():
                        channel_str = f"Ch {channel}"
                        
                        # Create voltage series
                        if 'timestamp' in data and len(data['timestamp']) == len(values):
                            self.plot_widget.add_data_series(channel_str, "Voltage", 
                                                            data['timestamp'], values)
                        else:
                            if 'timestamp' in data:
                                print(f"Length mismatch: timestamp[{len(data['timestamp'])}] vs voltage[{channel}][{len(values)}]")
                        
                        # Create current series if available
                        if 'current' in data and channel in data['current']:
                            current_values = data['current'][channel]
                            if 'timestamp' in data and len(data['timestamp']) == len(current_values):
                                self.plot_widget.add_data_series(channel_str, "Current", 
                                                                data['timestamp'], current_values)
                            else:
                                if 'timestamp' in data:
                                    print(f"Length mismatch: timestamp[{len(data['timestamp'])}] vs current[{channel}][{len(current_values)}]")
                
                # Update the plot
                self.plot_widget.update_plot()
                return
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing experiment update: {str(e)}")

    def refresh_available_experiments(self):
        """Refresh the list of available experiments"""
        self.available_experiments = self.experiment_launcher.get_available_experiments()
        
        # Update the experiment selector if it exists
        if hasattr(self, 'experiment_combo'):
            # Save the current selection
            current_experiment = self.experiment_combo.currentText()
            
            # Update the items
            self.experiment_combo.clear()
            self.experiment_combo.addItems(self.available_experiments)
            
            # Restore the selection if possible
            if current_experiment in self.available_experiments:
                self.experiment_combo.setCurrentText(current_experiment)
    
    def on_experiment_changed(self, experiment_name):
        """Handle experiment selection change"""
        if not experiment_name or experiment_name not in self.available_experiments:
            return
        
        # Get the experiment description
        description = self.experiment_launcher.get_experiment_description(experiment_name)
        
        # Update the experiment runner
        self.experiment_runner.set_selected_experiment(experiment_name, description)
        
        # Update the experiment UI tab
        self.update_experiment_ui_tab(experiment_name)
    
    def update_experiment_ui_tab(self, experiment_name):
        """Update the experiment UI tab with settings for the selected experiment"""
        # Clear the current UI
        for i in reversed(range(self.experiment_layout.count())): 
            item = self.experiment_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Get the experiment settings
        settings = self.experiment_launcher.get_required_settings(experiment_name)
        
        # If there are no settings, show a placeholder
        if not settings:
            self.experiment_placeholder = QLabel(f"No settings available for {experiment_name}")
            self.experiment_layout.addWidget(self.experiment_placeholder)
            return
        
        # Otherwise, create a form with settings controls
        from PyQt6.QtWidgets import QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox
        
        form_layout = QFormLayout()
        self.experiment_layout.addLayout(form_layout)
        
        # Create controls for each setting
        self.setting_controls = {}
        for setting_name, setting_info in settings.items():
            if isinstance(setting_info, tuple):
                # Numeric setting
                if len(setting_info) >= 2:
                    min_val, max_val = setting_info[0:2]
                    step = setting_info[2] if len(setting_info) > 2 else 1
                    
                    if isinstance(min_val, float) or isinstance(max_val, float) or isinstance(step, float):
                        # Float setting
                        control = QDoubleSpinBox()
                        control.setDecimals(9)  # Support for very small values
                    else:
                        # Integer setting
                        control = QSpinBox()
                    
                    control.setRange(min_val, max_val)
                    control.setSingleStep(step)
                    
                elif isinstance(setting_info[0], bool):
                    # Boolean setting
                    control = QCheckBox()
                    control.setChecked(setting_info[0])
            else:
                # Unknown setting type
                control = QLabel(f"Unsupported setting type: {setting_info}")
            
            form_layout.addRow(f"{setting_name}:", control)
            self.setting_controls[setting_name] = control
        
        # Add a button to apply settings
        apply_button = QPushButton("Apply Settings")
        apply_button.clicked.connect(self.on_apply_experiment_settings)
        self.experiment_layout.addWidget(apply_button)
    
    def on_apply_experiment_settings(self):
        """Apply the current experiment settings"""
        experiment_name = self.experiment_combo.currentText()
        if not experiment_name:
            return
        
        # Collect the settings from the UI
        settings = {}
        for setting_name, control in self.setting_controls.items():
            if isinstance(control, (QDoubleSpinBox, QSpinBox)):
                settings[setting_name] = control.value()
            elif isinstance(control, QCheckBox):
                settings[setting_name] = control.isChecked()
        
        # Update the experiment runner with the settings
        self.experiment_runner.set_experiment_settings(settings)
        
        # Show confirmation
        self.status_bar.showMessage(f"Applied settings for {experiment_name}")
    
    def on_connect(self):
        """Connect to instruments"""
        # To be implemented
        self.status_bar.showMessage("Connect functionality not yet implemented")
    
    def on_settings(self):
        """Open the settings dialog"""
        settings_window = MeasurementSettingsWindow(self)
        settings_window.exec()
    
    def on_load_experiment(self):
        """Load experiment settings from a file"""
        # To be implemented
        self.status_bar.showMessage("Load experiment functionality not yet implemented")
    
    def on_save_experiment(self):
        """Save experiment settings to a file"""
        # To be implemented
        self.status_bar.showMessage("Save experiment functionality not yet implemented")
    
    def on_run_experiment(self):
        """Run the current experiment"""
        # Forward to the experiment runner
        self.experiment_runner.run_experiment()
    
    def on_stop_experiment(self):
        """Stop the current experiment"""
        # Forward to the experiment runner
        success = self.experiment_launcher.stop_experiment()
        if success:
            self.status_bar.showMessage("Experiment stopped")
        else:
            self.status_bar.showMessage("Failed to stop experiment")
    
    def update_ui(self):
        """Update the UI periodically"""
        # Update status indicators, etc.
        pass
    
    def restore_window_geometry(self):
        """Restore window size and position from saved settings"""
        geometry = self.settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Default size and position
            self.resize(1200, 800)
            self.center_window()
    
    def center_window(self):
        """Center the window on the screen"""
        frame_geometry = self.frameGeometry()
        screen = QApplication.primaryScreen()
        center_point = screen.availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save window geometry
        self.settings.setValue("windowGeometry", self.saveGeometry())
        
        # Clean up
        self.experiment_launcher.stop()
        
        # Accept the close event
        event.accept()


def launch_gui():
    """Launches the GUI in a separate process"""
    app = QApplication(sys.argv)
    window = ElexMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Create a separate process for the GUI
    gui_process = multiprocessing.Process(target=launch_gui)
    gui_process.start()
    # The main process could handle other tasks, communication
    # with the GUI would need to be set up via IPC mechanisms
    gui_process.join()  # Wait for GUI to close 