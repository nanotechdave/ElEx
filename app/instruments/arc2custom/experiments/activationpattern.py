import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarc2

from PyQt6.QtWidgets import QApplication

from arc2custom import dparclib as dparc
from arc2custom import dplib as dp
from arc2custom import measurementsettings, sessionmod
from .experiment import Experiment
from app.gui.measurement_settings_window import MeasurementSettingsWindow

class ActivationPattern(Experiment):
    """
    Defines an Activation Pattern Measurement routine.

    This experiment:
    1. Uses a GUI to get an electrode sequence from the user
    2. Sequentially runs IV measurements between pairs of electrodes in the sequence
    3. Stops each IV when conductance reaches g_stop (1G by default)
    4. Advances to the next pair until the entire sequence is processed
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)

        # Default IV measurement parameters
        self.sample_time = 0.5
        self.start_voltage = 0
        self.end_voltage = 20
        self.voltage_step = 0.2
        self.script = "ActivationPattern"
        self.g_stop = 1.0  # Default to 1G
        self.g_interval = 0.5
        self.g_points = 10
        self.float_at_end = True
        
        # Activation sequence
        self.electrode_sequence = []
        
        # Set default mapping
        self.mapping_file = "grid_fix16.toml"
        # Load the mapping if not already loaded in session
        self.ensure_mapping_loaded()

    def ensure_mapping_loaded(self):
        """Ensure the correct mapping is loaded in the session"""
        if not self.session.useDbMap or self.session.mapping_file != self.mapping_file:
            # Get the path to the mappings directory
            mapping_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "mappings" / self.mapping_file
            print(f"Loading electrode mapping from: {mapping_path}")
            
            # Load the mapping
            self.session.mapping = dp.load_channel_map(str(mapping_path))
            self.session.mapping_file = self.mapping_file
            self.session.useDbMap = True
            print(f"Mapping loaded successfully: {self.mapping_file}")

    def setMeasurement(
        self,
        sample_time: float = 0.5,
        start_voltage: float = 0,
        end_voltage: float = 20,
        voltage_step: float = 0.2,
        g_stop: float = 1.0,
        g_interval: float = 0.5,
        g_points: int = 10,
        float_at_end: bool = True,
        mapping_file: str = "grid_fix16.toml"
    ):
        """
        Set measurement parameters and open the GUI to get the electrode sequence
        """
        # Store the measurement parameters
        self.sample_time = sample_time
        self.start_voltage = start_voltage
        self.end_voltage = end_voltage
        self.voltage_step = voltage_step
        self.g_stop = g_stop
        self.g_interval = g_interval
        self.g_points = g_points
        self.float_at_end = float_at_end
        
        # Set mapping file if changed
        if mapping_file != self.mapping_file:
            self.mapping_file = mapping_file
            self.ensure_mapping_loaded()

        # Get electrode sequence from GUI
        self.electrode_sequence = self.get_electrode_sequence()
        
        if not self.electrode_sequence or len(self.electrode_sequence) < 2:
            print("Error: Need at least 2 electrodes in the sequence to run the experiment.")
            return False
            
        print(f"Electrode sequence set: {self.electrode_sequence}")
        
        # Print the mapping for debugging
        print("Electrode to ARC channel mapping:")
        for electrode in self.electrode_sequence:
            arc_channel = self.session.dbToArc(electrode)
            print(f"  Electrode {electrode} → ARC channel {arc_channel}")
            
        return True
        
    def get_electrode_sequence(self):
        """
        Opens the GUI to let the user select an electrode sequence
        """
        # Create a QApplication instance if one doesn't already exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            
        # Create and show the electrode sequencer window
        window = MeasurementSettingsWindow()
        
        # Define a callback for when sequence is confirmed
        sequence = []
        def on_sequence_confirmed(selected_sequence):
            nonlocal sequence
            sequence = selected_sequence
            
        # Connect the signal
        window.sequenceConfirmed.connect(on_sequence_confirmed)
        
        # Show the window and run the event loop
        window.show()
        app.exec()
        
        return sequence

    def writeLogFile(self, f):
        """Writes Log file for Activation Pattern measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"TIME: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")
        f.write(f"MAPPING: {self.mapping_file}\n")
        if self.session.useDbMap:
            f.write(f"MAPPING of output file: Daughterboard\n")
        else:
            f.write(f"MAPPING of output file: Arc Internal Channels\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Start voltage: {self.start_voltage} \n")
        f.write(f"End voltage: {self.end_voltage} \n")
        f.write(f"Voltage step: {self.voltage_step} \n")
        f.write(f"Conductance stop: {self.g_stop} \n")
        f.write(f"Electrode sequence: {self.electrode_sequence} \n")

        f.write("USED CHANNELS (arc internal numbering): \n\n")
        f.write(f"All channels: {self.settings.mask} \n")
        f.write(f"Channels set to reference voltage: {self.settings.mask_to_gnd}\n")
        f.write(f"Channels set to bias voltage: {self.settings.mask_to_bias}\n")
        f.write(f"Voltage is read from channels: {self.settings.mask_to_read_v}\n")
        f.write(f"Current is read from channels: {self.settings.mask_to_read_i}\n\n")
        f.write("All other channels are set to floating.\n")

        f.flush()
        os.fsync(f.fileno())
        f.close()
        return

    def initializeFiles(self):
        self.session.num = (
            dp.findMaxNum(f"{self.session.savepath}/{self.session.sample}") + 1
        )
        self.session.dateUpdate()
        data_filename = f"{str(self.session.num).zfill(3)}_{self.session.lab}_{self.session.sample}_{self.session.cell}_{self.script}_{self.session.date}"
        data_file = dp.fileInit(
            savepath=f"{self.session.savepath}/{self.session.sample}",
            filename=data_filename,
            header=self.headerInit(self.settings.mask),
        )

        log_file = dp.fileInit(
            savepath=f"{self.session.savepath}/{self.session.sample}",
            filename=f"{data_filename}_log",
            header="",
        )
        self.writeLogFile(log_file)
        return data_file, data_filename

    def run(self, plot: bool = True):
        """
        Performs the Activation Pattern Measurement routine.
        
        For each pair of electrodes in the sequence:
        1. First electrode is set as bias
        2. Second electrode is set as ground
        3. Runs IV measurement until g_stop is reached
        4. Moves to the next pair
        """
        if not self.electrode_sequence or len(self.electrode_sequence) < 2:
            print("Error: Need at least 2 electrodes in the sequence to run the experiment.")
            return False
            
        print(f"{self.name} has started with electrode sequence: {self.electrode_sequence}")
        
        # Files initialization for the full experiment
        data_file, data_filename = self.initializeFiles()
        
        # Prepare for plotting
        if plot:
            T_vec = np.array([])
            I_vec = np.array([])
            Vdiff_vec = np.array([])
            G_vec = np.array([])
            fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4 = dp.plotGen(
                T_vec, I_vec, Vdiff_vec, G_vec
            )
            
        # Process each pair of electrodes
        for i in range(len(self.electrode_sequence) - 1):
            # Get the current pair
            bias_electrode = self.electrode_sequence[i]
            gnd_electrode = self.electrode_sequence[i+1]
            
            print(f"Processing electrode pair: {bias_electrode} (bias) -> {gnd_electrode} (ground)")
            
            # Convert from electrode number to ARC channels using the mapping
            bias_channel = self.session.dbToArc(bias_electrode)
            gnd_channel = self.session.dbToArc(gnd_electrode)
            
            print(f"  Mapped to ARC channels: {bias_channel} (bias) -> {gnd_channel} (ground)")
            
            # Set up the measurement settings for this pair
            self.setMaskSettings(
                mask_to_bias=[bias_channel],
                mask_to_gnd=[gnd_channel],
                mask_to_read_v=[bias_channel, gnd_channel],
                mask_to_read_i=[bias_channel, gnd_channel],
                db_mapping=False  # We've already converted to ARC channels
            )
            
            # Generate the voltage ramp for this measurement
            self.settings.v_tuple = np.array(
                dp.rampGenerator(self.start_voltage, self.end_voltage, self.voltage_step, self.sample_time)
            )
            
            self.settings.vBiasVec, self.settings.vTimes = np.array(
                dp.tuplesToVec(self.settings.v_tuple)
            )
            
            # Run the IV measurement for this pair
            self.run_single_pair(data_file, plot, fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4 if plot else None)
        
        # Save the figure and close the file
        if plot:
            fig.savefig(
                f"{self.session.savepath}/{self.session.sample}/"
                + data_filename
                + ".png"
            )
        
        data_file.close()
        
        # Set all channels to float at the end if requested
        if self.float_at_end:
            dparc.setAllChannelsToFloat(self.arc)
            
        print(f"{self.name} completed successfully.")
        return True
        
    def run_single_pair(self, data_file, plot, fig=None, ax1=None, plot1=None, ax2=None, plot2=None, ax3=None, plot3=None, ax4=None, plot4=None):
        """
        Run a single IV measurement between a pair of electrodes until g_stop is reached
        """
        # Start all useful timers
        start_prog = time.time()
        progress_timer = time.time()
        T_vec = np.array([])
        I_vec = np.array([])
        Vdiff_vec = np.array([])
        V_end_vec = np.array([])
        V0_vec = np.array([])
        G_vec = np.array([])
        
        try:
            # Float all channels at the start
            if self.float_at_end:
                dparc.setAllChannelsToFloat(self.arc)
                
            # Cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                # Set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.01
                    ) / self.settings.meas_iterations
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)
                
                # Bias the electrodes and measure
                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp, voltage, current = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                
                # Write measurement to file
                data_row = self.measureToStr(
                    timestamp, voltage, current, self.settings.mask
                )
                dp.fileUpdate(data_file, data_row)
                
                # Calculate and store values for plotting
                i_0 = dp.first_non_nan(current)
                v_0 = voltage[0]
                v_end = 0  # Ground is at 0V
                I_vec = np.append(I_vec, i_0)
                V_end_vec = np.append(V_end_vec, v_end)
                V0_vec = np.append(V0_vec, v_0)
                Vdiff_vec = np.append(Vdiff_vec, v_0 - v_end)
                T_vec = np.append(T_vec, timestamp)
                
                # Calculate conductance (G = I/V in units of G0 = 7.748e-5 S)
                lastG = abs(i_0 / (v_0 - v_end)) / 7.748e-5
                G_vec = np.append(G_vec, lastG)
                
                # Update plot if requested
                if plot:
                    dp.plotUpdate(
                        T_vec,
                        I_vec,
                        Vdiff_vec,
                        G_vec,
                        fig,
                        ax1,
                        plot1,
                        ax2,
                        plot2,
                        ax3,
                        plot3,
                        ax4,
                        plot4,
                    )
                    print(f"Current conductance: {lastG} G0")
                
                # Check if we've reached the stop condition
                if dp.isStable(G_vec, self.g_stop, self.g_interval, self.g_points):
                    print(f"Reached target conductance of {self.g_stop} G0. Moving to next pair.")
                    return
                
                # Print progress
                if time.time() - progress_timer > 10 or sample_step + 1 == len(self.settings.vBiasVec):
                    progress = round(
                        (time.time() - start_prog) / self.settings.vTimes[-1] * 100
                    )
                    if progress > 100:
                        progress = 100
                    print(
                        "Measurement progress: "
                        + str(progress)
                        + "%. Time from start: "
                        + str(round(time.time() - start_prog))
                        + " seconds."
                    )
                    progress_timer = time.time()
                    
            print("Completed voltage ramp without reaching target conductance.")
            
        except Exception as e:
            print(f"Error during measurement: {str(e)}")
            if self.float_at_end:
                dparc.setAllChannelsToFloat(self.arc)
            raise e 