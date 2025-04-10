import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (QWidget, QGridLayout, QPushButton, 
                             QVBoxLayout, QLabel, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor
import numpy as np

class ElectrodeMatrix(QWidget):
    # Signals
    sequenceChanged = pyqtSignal(list)
    sequenceConfirmed = pyqtSignal(list)  # Signal for confirmed sequence
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_electrodes = []
        self.arrow_points = []  # Store points for drawing arrows
        self.initUI()
        
    def initUI(self):
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create grid for electrode buttons
        self.grid = QGridLayout()
        self.buttons = []
        
        # Create 4x4 matrix of buttons
        for i in range(4):
            for j in range(4):
                # Calculate electrode number (1-based)
                electrode_num = i * 4 + j + 1
                btn = QPushButton(str(electrode_num))
                btn.setFixedSize(60, 60)
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, num=electrode_num: self.toggleElectrode(num))
                self.grid.addWidget(btn, i, j)
                self.buttons.append(btn)
        
        # Create sequence display
        self.sequence_label = QLabel("Selected sequence: []")
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Add clear button
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clearSelection)
        
        # Add confirm button
        confirm_btn = QPushButton("Confirm Sequence")
        confirm_btn.clicked.connect(self.confirmSequence)
        confirm_btn.setStyleSheet("background-color: #4CAF50; color: white;")  # Green color
        
        # Add buttons to layout
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(confirm_btn)
        
        # Add layouts to main layout
        main_layout.addLayout(self.grid)
        main_layout.addWidget(self.sequence_label)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
    def toggleElectrode(self, electrode_num):
        if electrode_num in self.selected_electrodes:
            # Remove electrode and all subsequent electrodes
            idx = self.selected_electrodes.index(electrode_num)
            self.selected_electrodes = self.selected_electrodes[:idx]
            # Uncheck all buttons after this one
            for i in range(idx, len(self.buttons)):
                self.buttons[i].setChecked(False)
        else:
            self.selected_electrodes.append(electrode_num)
            self.buttons[electrode_num - 1].setChecked(True)
            
        # Update display
        self.sequence_label.setText(f"Selected sequence: {self.selected_electrodes}")
        # Update arrow points
        self.updateArrowPoints()
        # Emit signal with current sequence
        self.sequenceChanged.emit(self.selected_electrodes)
        
    def clearSelection(self):
        self.selected_electrodes = []
        self.arrow_points = []
        for btn in self.buttons:
            btn.setChecked(False)
        self.sequence_label.setText("Selected sequence: []")
        self.sequenceChanged.emit([])
        self.update()
        
    def confirmSequence(self):
        if self.selected_electrodes:
            self.sequenceConfirmed.emit(self.selected_electrodes)
            
    def getSequence(self):
        return self.selected_electrodes
        
    def updateArrowPoints(self):
        self.arrow_points = []
        if len(self.selected_electrodes) < 2:
            return
            
        for i in range(len(self.selected_electrodes) - 1):
            # Get button positions for current and next electrode
            current_btn = self.buttons[self.selected_electrodes[i] - 1]
            next_btn = self.buttons[self.selected_electrodes[i + 1] - 1]
            
            # Calculate center points of buttons
            start_point = current_btn.mapTo(self, current_btn.rect().center())
            end_point = next_btn.mapTo(self, next_btn.rect().center())
            
            self.arrow_points.append((start_point, end_point))
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.arrow_points:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up pen for arrows
        pen = QPen(QColor(0, 0, 255))  # Blue color
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Draw arrows
        for start_point, end_point in self.arrow_points:
            # Draw line
            painter.drawLine(start_point, end_point)
            
            # Calculate arrow head
            angle = np.arctan2(end_point.y() - start_point.y(), 
                             end_point.x() - start_point.x())
            arrow_size = 10
            
            # Calculate arrow head points
            arrow_p1 = end_point - QPoint(
                int(arrow_size * np.cos(angle - np.pi/6)),
                int(arrow_size * np.sin(angle - np.pi/6))
            )
            arrow_p2 = end_point - QPoint(
                int(arrow_size * np.cos(angle + np.pi/6)),
                int(arrow_size * np.sin(angle + np.pi/6))
            )
            
            # Draw arrow head
            painter.drawLine(end_point, arrow_p1)
            painter.drawLine(end_point, arrow_p2) 