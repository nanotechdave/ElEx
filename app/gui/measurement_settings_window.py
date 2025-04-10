import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QInputDialog, QMessageBox)
from PyQt5.QtCore import pyqtSignal
from app.gui.electrode_matrix import ElectrodeMatrix

class MeasurementSettingsWindow(QMainWindow):
    # Signal that will be emitted when sequence is confirmed
    sequenceConfirmed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Electrode Sequencer")
        self.confirmed_sequence = []  # Store the confirmed sequence
        self.patterns_file = Path(__file__).parent / "patterns.json"
        self.patterns = self.load_patterns()
        self.initUI()
        
    def load_patterns(self):
        """Load patterns from JSON file"""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                return json.load(f)['patterns']
        return {}
        
    def save_patterns(self):
        """Save patterns to JSON file"""
        with open(self.patterns_file, 'w') as f:
            json.dump({'patterns': self.patterns}, f, indent=4)
            
    def initUI(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create pattern selector
        pattern_layout = QHBoxLayout()
        pattern_label = QLabel("Select Pattern:")
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Custom"] + list(self.patterns.keys()))
        self.pattern_combo.currentTextChanged.connect(self.onPatternSelected)
        
        # Add save pattern button
        save_pattern_btn = QPushButton("Save Current as Pattern")
        save_pattern_btn.clicked.connect(self.saveCurrentPattern)
        
        pattern_layout.addWidget(pattern_label)
        pattern_layout.addWidget(self.pattern_combo)
        pattern_layout.addWidget(save_pattern_btn)
        
        # Create electrode matrix
        self.electrode_matrix = ElectrodeMatrix()
        self.electrode_matrix.sequenceChanged.connect(self.onSequenceChanged)
        self.electrode_matrix.sequenceConfirmed.connect(self.onSequenceConfirmed)
        
        # Add widgets to main layout
        main_layout.addLayout(pattern_layout)
        main_layout.addWidget(self.electrode_matrix)
        
        # Set window size
        self.setMinimumSize(400, 500)
        
    def onPatternSelected(self, pattern_name):
        if pattern_name == "Custom":
            return
            
        if pattern_name in self.patterns:
            # Clear current selection
            self.electrode_matrix.clearSelection()
            # Apply the selected pattern
            for electrode in self.patterns[pattern_name]:
                self.electrode_matrix.toggleElectrode(electrode)
                
    def saveCurrentPattern(self):
        if not self.electrode_matrix.getSequence():
            QMessageBox.warning(self, "Warning", "No sequence selected to save!")
            return
            
        name, ok = QInputDialog.getText(self, 'Save Pattern', 
                                      'Enter a name for this pattern:')
        if ok and name:
            if name in self.patterns:
                reply = QMessageBox.question(self, 'Confirm Overwrite',
                                          f'Pattern "{name}" already exists. Overwrite?',
                                          QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    return
                    
            self.patterns[name] = self.electrode_matrix.getSequence()
            self.save_patterns()
            
            # Update combo box
            current_text = self.pattern_combo.currentText()
            self.pattern_combo.clear()
            self.pattern_combo.addItems(["Custom"] + list(self.patterns.keys()))
            self.pattern_combo.setCurrentText(current_text)
        
    def onSequenceChanged(self, sequence):
        # This method will be called whenever the electrode sequence changes
        print(f"Selected electrode sequence: {sequence}")
        
    def onSequenceConfirmed(self, sequence):
        # Store the confirmed sequence
        self.confirmed_sequence = sequence
        print(f"Confirmed electrode sequence: {sequence}")
        # Emit the signal with the confirmed sequence
        self.sequenceConfirmed.emit(sequence)
        # Close the window
        self.close()
        
    def getConfirmedSequence(self):
        # Method to get the last confirmed sequence
        return self.confirmed_sequence 