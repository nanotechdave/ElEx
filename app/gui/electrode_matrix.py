import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (QWidget, QGridLayout, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QLabel, QButtonGroup, QRadioButton)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

class ElectrodeButton(QPushButton):
    """A custom button representing an electrode"""
    
    def __init__(self, number, parent=None):
        super().__init__(str(number), parent)
        self.number = number
        self.setCheckable(True)
        self.setMinimumSize(40, 40)
        self.role = "none"  # none, bias, gnd, read
        self.updateStyleSheet()
    
    def setRole(self, role):
        """Set the role of this electrode"""
        self.role = role
        self.updateStyleSheet()
    
    def updateStyleSheet(self):
        """Update button appearance based on role"""
        self.setChecked(self.role != "none")
        
        if self.role == "bias":
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff5252;
                }
            """)
        elif self.role == "gnd":
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196f3;
                    color: white;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #64b5f6;
                }
            """)
        elif self.role == "read":
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4caf50;
                    color: white;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #81c784;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    color: black;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #bdbdbd;
                }
            """)

class ElectrodeMatrixWidget(QWidget):
    """A widget displaying a matrix of electrodes that can be selected for different roles"""
    
    # Signals for when electrode selections change
    bias_selection_changed = pyqtSignal(list)
    gnd_selection_changed = pyqtSignal(list)
    read_selection_changed = pyqtSignal(list)
    
    def __init__(self, rows=4, cols=8, parent=None):
        super().__init__(parent)
        
        self.rows = rows
        self.cols = cols
        self.electrodes = {}  # Dictionary of electrode buttons
        self.bias_electrodes = []
        self.gnd_electrodes = []
        self.read_electrodes = []
        
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
        
        # Create role selection controls
        role_layout = QHBoxLayout()
        self.role_group = QButtonGroup()
        
        self.bias_radio = QRadioButton("Bias")
        self.bias_radio.setChecked(True)
        self.gnd_radio = QRadioButton("Ground")
        self.read_radio = QRadioButton("Read")
        self.none_radio = QRadioButton("None")
        
        self.role_group.addButton(self.bias_radio)
        self.role_group.addButton(self.gnd_radio)
        self.role_group.addButton(self.read_radio)
        self.role_group.addButton(self.none_radio)
        
        role_layout.addWidget(self.bias_radio)
        role_layout.addWidget(self.gnd_radio)
        role_layout.addWidget(self.read_radio)
        role_layout.addWidget(self.none_radio)
        
        # Add legend
        legend_layout = QHBoxLayout()
        
        bias_label = QLabel("Bias")
        bias_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        gnd_label = QLabel("Ground")
        gnd_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        
        read_label = QLabel("Read")
        read_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        
        legend_layout.addWidget(bias_label)
        legend_layout.addWidget(gnd_label)
        legend_layout.addWidget(read_label)
        legend_layout.addStretch()
        
        # Create electrode matrix
        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)
        
        electrode_number = 8  # Starting at channel 8 to match ARC2 numbering
        
        for row in range(self.rows):
            for col in range(self.cols):
                button = ElectrodeButton(electrode_number)
                button.clicked.connect(self.onElectrodeClicked)
                
                self.electrodes[electrode_number] = button
                grid_layout.addWidget(button, row, col)
                
                electrode_number += 1
        
        # Add layouts to main layout
        main_layout.addLayout(role_layout)
        main_layout.addLayout(legend_layout)
        main_layout.addLayout(grid_layout)
        main_layout.addStretch()
    
    def onElectrodeClicked(self):
        """Handle electrode button click"""
        button = self.sender()
        if not isinstance(button, ElectrodeButton):
            return
        
        # Determine which role is selected
        if self.bias_radio.isChecked():
            new_role = "bias"
        elif self.gnd_radio.isChecked():
            new_role = "gnd"
        elif self.read_radio.isChecked():
            new_role = "read"
        else:  # None
            new_role = "none"
        
        # Toggle role (if already has this role, remove it)
        if button.role == new_role:
            button.setRole("none")
        else:
            # Remove from previous role lists
            if button.role == "bias" and button.number in self.bias_electrodes:
                self.bias_electrodes.remove(button.number)
            elif button.role == "gnd" and button.number in self.gnd_electrodes:
                self.gnd_electrodes.remove(button.number)
            elif button.role == "read" and button.number in self.read_electrodes:
                self.read_electrodes.remove(button.number)
            
            # Set new role
            button.setRole(new_role)
            
            # Add to new role list
            if new_role == "bias" and button.number not in self.bias_electrodes:
                self.bias_electrodes.append(button.number)
            elif new_role == "gnd" and button.number not in self.gnd_electrodes:
                self.gnd_electrodes.append(button.number)
            elif new_role == "read" and button.number not in self.read_electrodes:
                self.read_electrodes.append(button.number)
        
        # Emit signals for role changes
        self.bias_selection_changed.emit(sorted(self.bias_electrodes))
        self.gnd_selection_changed.emit(sorted(self.gnd_electrodes))
        self.read_selection_changed.emit(sorted(self.read_electrodes))
    
    def clearSelection(self):
        """Clear all selected electrodes"""
        for button in self.electrodes.values():
            button.setRole("none")
        
        self.bias_electrodes = []
        self.gnd_electrodes = []
        self.read_electrodes = []
        
        # Emit signals for role changes
        self.bias_selection_changed.emit([])
        self.gnd_selection_changed.emit([])
        self.read_selection_changed.emit([])
    
    def setElectrodeRoles(self, bias_list=None, gnd_list=None, read_list=None):
        """Set electrode roles from lists"""
        # Clear current selection
        self.clearSelection()
        
        # Set bias electrodes
        if bias_list:
            for electrode_num in bias_list:
                if electrode_num in self.electrodes:
                    self.electrodes[electrode_num].setRole("bias")
                    self.bias_electrodes.append(electrode_num)
        
        # Set ground electrodes
        if gnd_list:
            for electrode_num in gnd_list:
                if electrode_num in self.electrodes:
                    self.electrodes[electrode_num].setRole("gnd")
                    self.gnd_electrodes.append(electrode_num)
        
        # Set read electrodes
        if read_list:
            for electrode_num in read_list:
                if electrode_num in self.electrodes:
                    self.electrodes[electrode_num].setRole("read")
                    self.read_electrodes.append(electrode_num)
        
        # Emit signals for role changes
        self.bias_selection_changed.emit(sorted(self.bias_electrodes))
        self.gnd_selection_changed.emit(sorted(self.gnd_electrodes))
        self.read_selection_changed.emit(sorted(self.read_electrodes))
    
    def getBiasElectrodes(self):
        """Get list of bias electrodes"""
        return sorted(self.bias_electrodes)
    
    def getGndElectrodes(self):
        """Get list of ground electrodes"""
        return sorted(self.gnd_electrodes)
    
    def getReadElectrodes(self):
        """Get list of read electrodes"""
        return sorted(self.read_electrodes) 