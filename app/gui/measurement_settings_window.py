import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel)
from PyQt5.QtCore import pyqtSignal
from app.gui.electrode_matrix import ElectrodeMatrix

class MeasurementSettingsWindow(QMainWindow):
    # Signal that will be emitted when sequence is confirmed
    sequenceConfirmed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Electrode Sequencer")
        self.confirmed_sequence = []  # Store the confirmed sequence
        self.initUI()
        
    def initUI(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create electrode matrix
        self.electrode_matrix = ElectrodeMatrix()
        self.electrode_matrix.sequenceChanged.connect(self.onSequenceChanged)
        self.electrode_matrix.sequenceConfirmed.connect(self.onSequenceConfirmed)
        
        # Add widgets to main layout
        main_layout.addWidget(self.electrode_matrix)
        
        # Set window size
        self.setMinimumSize(400, 500)
        
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