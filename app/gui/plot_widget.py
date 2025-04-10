import sys
import os
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal, QTimer
import pyqtgraph as pg

class RealTimePlotWidget(QWidget):
    """
    A high-performance real-time plotting widget using PyQtGraph for displaying measurement data.
    Optimized for efficiency and minimal overhead.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize main layout
        self.layout = QVBoxLayout(self)
        
        # Create plot control layout
        control_layout = QHBoxLayout()
        
        # Channel selector for plot
        self.channel_selector = QComboBox()
        self.channel_selector.addItem("All Channels")
        control_layout.addWidget(QLabel("Channel:"))
        control_layout.addWidget(self.channel_selector)
        
        # Plot type selector
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Voltage vs Time", "Current vs Time", "IV Curve", "Resistance vs Time"])
        control_layout.addWidget(QLabel("Plot Type:"))
        control_layout.addWidget(self.plot_type)
        
        # Clear button
        self.clear_btn = QPushButton("Clear Plot")
        self.clear_btn.clicked.connect(self.clear_plot)
        control_layout.addWidget(self.clear_btn)
        
        # Add control layout to main layout
        self.layout.addLayout(control_layout)
        
        # Set dark background (looks more professional for scientific plots)
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        
        # Create the pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        
        # Add plot to layout
        self.layout.addWidget(self.plot_widget)
        
        # Connect signals
        self.plot_type.currentTextChanged.connect(self.update_plot_type)
        self.channel_selector.currentTextChanged.connect(self.update_channel)
        
        # Initialize data storage
        self.time_data = []
        self.voltage_data = {}
        self.current_data = {}
        self.resistance_data = {}
        
        # Plot items dictionary to track curves
        self.plot_items = {}
        
        # Current plot settings
        self.current_plot_type = "Voltage vs Time"
        self.current_channel = "All Channels"
        
        # Colors for different channels (will cycle)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 128, 255)
        ]
    
    def update_plot_type(self, plot_type):
        """Update the plot type and redraw"""
        self.current_plot_type = plot_type
        
        # Update axis labels
        if plot_type == "Voltage vs Time":
            self.plot_widget.setLabel('left', 'Voltage', units='V')
            self.plot_widget.setLabel('bottom', 'Time', units='s')
        elif plot_type == "Current vs Time":
            self.plot_widget.setLabel('left', 'Current', units='A')
            self.plot_widget.setLabel('bottom', 'Time', units='s')
        elif plot_type == "IV Curve":
            self.plot_widget.setLabel('left', 'Current', units='A')
            self.plot_widget.setLabel('bottom', 'Voltage', units='V')
        elif plot_type == "Resistance vs Time":
            self.plot_widget.setLabel('left', 'Resistance', units='Ω')
            self.plot_widget.setLabel('bottom', 'Time', units='s')
        
        # Redraw plot
        self.redraw_plot()
    
    def update_channel(self, channel):
        """Update the selected channel and redraw"""
        self.current_channel = channel
        self.redraw_plot()
    
    def clear_plot(self):
        """Clear all plot data"""
        self.plot_widget.clear()
        self.plot_items = {}
        # Don't clear data storage - we might want to redraw later
    
    def update_channel_list(self, channels):
        """Update the channel selector with available channels"""
        # Save current selection
        current_channel = self.channel_selector.currentText()
        
        # Clear and rebuild the list
        self.channel_selector.clear()
        self.channel_selector.addItem("All Channels")
        
        for channel in sorted(channels):
            self.channel_selector.addItem(f"Channel {channel}")
        
        # Restore selection if possible
        index = self.channel_selector.findText(current_channel)
        if index >= 0:
            self.channel_selector.setCurrentIndex(index)
        else:
            self.channel_selector.setCurrentIndex(0)  # Default to "All Channels"
    
    def redraw_plot(self):
        """Redraw the plot with current settings"""
        # Clear existing plots
        self.plot_widget.clear()
        self.plot_items = {}
        
        # Check if we have data
        if not self.time_data:
            return
        
        # Get channels to plot
        if self.current_channel == "All Channels":
            if self.current_plot_type == "Voltage vs Time":
                channels = list(self.voltage_data.keys())
            elif self.current_plot_type == "Current vs Time":
                channels = list(self.current_data.keys())
            elif self.current_plot_type == "Resistance vs Time":
                channels = list(self.resistance_data.keys())
            else:  # IV Curve
                channels = list(set(self.voltage_data.keys()) & set(self.current_data.keys()))
        else:
            # Extract channel number from the text
            channel_text = self.current_channel
            if channel_text.startswith("Channel "):
                try:
                    channel_num = int(channel_text.replace("Channel ", ""))
                    channels = [channel_num]
                except ValueError:
                    channels = []
            else:
                channels = []
        
        # Create a plot for each channel
        for i, channel in enumerate(channels):
            color = self.colors[i % len(self.colors)]
            
            if self.current_plot_type == "Voltage vs Time" and channel in self.voltage_data:
                self.plot_items[channel] = self.plot_widget.plot(
                    self.time_data, 
                    self.voltage_data[channel], 
                    pen=color, 
                    name=f"Ch {channel}"
                )
            
            elif self.current_plot_type == "Current vs Time" and channel in self.current_data:
                self.plot_items[channel] = self.plot_widget.plot(
                    self.time_data, 
                    self.current_data[channel], 
                    pen=color, 
                    name=f"Ch {channel}"
                )
            
            elif self.current_plot_type == "Resistance vs Time" and channel in self.resistance_data:
                self.plot_items[channel] = self.plot_widget.plot(
                    self.time_data, 
                    self.resistance_data[channel], 
                    pen=color, 
                    name=f"Ch {channel}"
                )
            
            elif self.current_plot_type == "IV Curve" and channel in self.voltage_data and channel in self.current_data:
                self.plot_items[channel] = self.plot_widget.plot(
                    self.voltage_data[channel], 
                    self.current_data[channel], 
                    pen=color, 
                    name=f"Ch {channel}"
                )
    
    def update_data(self, new_data):
        """
        Update the plot with new experiment data.
        
        Args:
            new_data: Dictionary with experiment data
                {
                    "timestamp": [],
                    "voltage": { channel: [] },
                    "current": { channel: [] },
                    "resistance": { channel: [] }
                }
        """
        # Update time data if present
        if "timestamp" in new_data and new_data["timestamp"]:
            # Convert timestamps to relative time from start if needed
            if not self.time_data:
                # First data point - use as reference
                time_offset = new_data["timestamp"][0]
                self.time_data = [t - time_offset for t in new_data["timestamp"]]
            else:
                # Append new timestamps
                time_offset = self.time_data[0] + new_data["timestamp"][0]
                self.time_data.extend([t - time_offset for t in new_data["timestamp"]])
        
        # Update voltage data
        if "voltage" in new_data:
            for channel, values in new_data["voltage"].items():
                if channel not in self.voltage_data:
                    self.voltage_data[channel] = []
                self.voltage_data[channel].extend(values)
        
        # Update current data
        if "current" in new_data:
            for channel, values in new_data["current"].items():
                if channel not in self.current_data:
                    self.current_data[channel] = []
                self.current_data[channel].extend(values)
                
                # Update channel list if needed
                if not any(f"Channel {channel}" == self.channel_selector.itemText(i) 
                           for i in range(self.channel_selector.count())):
                    self.update_channel_list(self.current_data.keys())
        
        # Update resistance data
        if "resistance" in new_data:
            for channel, values in new_data["resistance"].items():
                if channel not in self.resistance_data:
                    self.resistance_data[channel] = []
                self.resistance_data[channel].extend(values)
        
        # Calculate resistance if not provided
        if "voltage" in new_data and "current" in new_data and "resistance" not in new_data:
            for channel in set(self.voltage_data.keys()) & set(self.current_data.keys()):
                if channel not in self.resistance_data:
                    self.resistance_data[channel] = []
                
                # Calculate resistance for new data points
                start_idx = len(self.resistance_data[channel])
                end_idx = min(len(self.voltage_data[channel]), len(self.current_data[channel]))
                
                for i in range(start_idx, end_idx):
                    voltage = self.voltage_data[channel][i]
                    current = self.current_data[channel][i]
                    
                    if abs(current) > 1e-12:  # Avoid division by zero
                        resistance = voltage / current
                    else:
                        resistance = float('inf')
                    
                    self.resistance_data[channel].append(resistance)
        
        # Redraw the plot
        self.redraw_plot()
    
    def set_title(self, title):
        """Set the plot title"""
        self.plot_widget.setTitle(title)
    
    def reset(self):
        """Reset all plot data."""
        self.time_data = []
        self.voltage_data = {}
        self.current_data = {}
        self.resistance_data = {}
        self.plot_items = {}
        self.plot_widget.clear()

    def add_data_series(self, channel_name, data_type, timestamps, values):
        """
        Add or update a data series for a specific channel and data type.
        
        Args:
            channel_name: Name of the channel (e.g., "Ch 0")
            data_type: Type of data ("Voltage", "Current", "Resistance")
            timestamps: List of timestamp values
            values: List of measurement values
        """
        # Both arrays must be the same length
        if len(timestamps) != len(values):
            print(f"Warning: timestamps and values arrays have different lengths: {len(timestamps)} vs {len(values)}")
            # Use the minimum length to ensure they match
            min_length = min(len(timestamps), len(values))
            timestamps = timestamps[:min_length]
            values = values[:min_length]
            if min_length == 0:
                return
        
        # Convert channel name to a number if it follows the pattern "Ch X"
        channel = None
        if channel_name.startswith("Ch "):
            try:
                channel = int(channel_name.replace("Ch ", ""))
            except ValueError:
                # Use the channel name as a string key if it doesn't parse as a number
                channel = channel_name
        else:
            channel = channel_name
            
        # If this is the first data point, initialize time reference
        if not self.time_data and timestamps:
            time_offset = timestamps[0]
            self.time_data = [t - time_offset for t in timestamps]
            
            # Add the data values to the appropriate data structure
            if data_type == "Voltage":
                self.voltage_data[channel] = values.copy()
            elif data_type == "Current":
                self.current_data[channel] = values.copy()
            elif data_type == "Resistance":
                self.resistance_data[channel] = values.copy()
            
        else:
            # Handle appending to existing data - make sure we preserve the sync between arrays
            new_time_values = []
            new_data_values = []
            
            # Find where to start appending - only append new timestamps
            if timestamps and self.time_data:
                last_time = self.time_data[-1] + timestamps[0]  # Convert to absolute time
                
                # Find the starting point for new data (avoid duplicates)
                start_idx = 0
                for i, t in enumerate(timestamps):
                    if t > last_time - timestamps[0]:  # Convert back to relative time
                        start_idx = i
                        break
                        
                # Append new data starting from that point
                if start_idx < len(timestamps):
                    time_offset = timestamps[0] 
                    for i in range(start_idx, len(timestamps)):
                        new_time_values.append(timestamps[i] - time_offset + self.time_data[0])
                        new_data_values.append(values[i])
                        
                    # Now extend the data arrays
                    if new_time_values:
                        self.time_data.extend(new_time_values)
                        
                        if data_type == "Voltage":
                            if channel not in self.voltage_data:
                                self.voltage_data[channel] = new_data_values
                            else:
                                self.voltage_data[channel].extend(new_data_values)
                        elif data_type == "Current":
                            if channel not in self.current_data:
                                self.current_data[channel] = new_data_values
                            else:
                                self.current_data[channel].extend(new_data_values)
                        elif data_type == "Resistance":
                            if channel not in self.resistance_data:
                                self.resistance_data[channel] = new_data_values
                            else:
                                self.resistance_data[channel].extend(new_data_values)
            
        # Ensure all data arrays have the same length as time_data for this channel
        # This is critical to prevent plotting errors
        if data_type == "Voltage" and channel in self.voltage_data:
            # Extend or trim to match time_data length
            if len(self.voltage_data[channel]) < len(self.time_data):
                # Pad with the last value if needed
                last_value = self.voltage_data[channel][-1] if self.voltage_data[channel] else 0
                self.voltage_data[channel].extend([last_value] * (len(self.time_data) - len(self.voltage_data[channel])))
            elif len(self.voltage_data[channel]) > len(self.time_data):
                self.voltage_data[channel] = self.voltage_data[channel][:len(self.time_data)]
                
        elif data_type == "Current" and channel in self.current_data:
            # Extend or trim to match time_data length
            if len(self.current_data[channel]) < len(self.time_data):
                # Pad with the last value if needed
                last_value = self.current_data[channel][-1] if self.current_data[channel] else 0
                self.current_data[channel].extend([last_value] * (len(self.time_data) - len(self.current_data[channel])))
            elif len(self.current_data[channel]) > len(self.time_data):
                self.current_data[channel] = self.current_data[channel][:len(self.time_data)]
                
        elif data_type == "Resistance" and channel in self.resistance_data:
            # Extend or trim to match time_data length
            if len(self.resistance_data[channel]) < len(self.time_data):
                # Pad with the last value if needed
                last_value = self.resistance_data[channel][-1] if self.resistance_data[channel] else 0
                self.resistance_data[channel].extend([last_value] * (len(self.time_data) - len(self.resistance_data[channel])))
            elif len(self.resistance_data[channel]) > len(self.time_data):
                self.resistance_data[channel] = self.resistance_data[channel][:len(self.time_data)]
        
        # Update channel list if needed
        if isinstance(channel, int) and not any(f"Channel {channel}" == self.channel_selector.itemText(i) 
                        for i in range(self.channel_selector.count())):
            self.update_channel_list(set(ch for ch in list(self.voltage_data.keys()) + 
                                     list(self.current_data.keys()) + 
                                     list(self.resistance_data.keys()) 
                                     if isinstance(ch, int)))
    
    def update_plot(self):
        """Update the plot with current data."""
        self.redraw_plot() 