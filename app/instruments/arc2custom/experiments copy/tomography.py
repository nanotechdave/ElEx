import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pyarc2

from arc2custom import dparclib as dparc
from arc2custom import dplib as dp
from arc2custom import measurementsettings, sessionmod
from .experiment import Experiment

class Tomography(Experiment):
    """
    Defines tomography acquisition routine.

    This experiment performs a voltage reading on successive pairs of electrodes,
    changing bias couple each cycle.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)
        self.sample_time = 0.01
        self.v_read = 0.05
        self.n_reps_avg = 1
        self.script = "Tomography"

    def setMeasurement(
        self,
        sample_time: float,
        v_read: float,
        n_reps_avg: float,
    ):
        self.sample_time = sample_time
        self.v_read = v_read
        self.n_reps_avg = n_reps_avg

    def writeLogFile(self, f):
        """Writes Log file for Tomography measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"TIME: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Read voltage: {self.v_read} \n")
        f.write(f"Number of measurements per configuration: {self.n_reps_avg} \n")

        f.write("USED CHANNELS: \n\n")
        f.write(f"All channels: {self.settings.mask} \n")
        f.write("All two-channels combinations have been performed.")
        f.write("All other channels are set to floating each time.\n")

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
            f"{self.session.savepath}/{self.session.sample}",
            data_filename,
            self.headerInit(self.settings.mask),
        )

        log_file = dp.fileInit(
            f"{self.session.savepath}/{self.session.sample}",
            f"{data_filename}_log",
            "",
        )
        self.writeLogFile(log_file)
        return data_file, data_filename

    def run(self):
        """
        Performs the routine of Tomography.
        """
        print(str(self.name) + " has started.")

        # ----------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()
        # --------------------- routine ----------------------------------------

        # start all useful timers
        start_prog = time.time()
        # loop over all possible channel pairs
        #bias_calibration done for channels 8-23
        bias_calibration=[0.0019,0.0006,0.0001,0.0039,-0.0001,-0.0011, 0.0015, 0.0017, 0.0023, -0.0013, 0.0009, -0.001, 0.0053, -0.0024, 0.0012,-0.0017]
        #bias_calibration done for channels 40-55 (meas.40 on FTO)
        #bias_calibration=[-0.0021, -0.0005, 0.0002, -0.0006, 0.0007, 0.0000, 0.0012, -0.0049, -0.0024, -0.0002, -0.0001, -0.0016, 0.0005, -0.0005, 0.0001, -0.0021]
        for idx in range(len(self.settings.mask)):
            if idx == len(self.settings.mask) - 1:
                couple = (self.settings.mask[idx], self.settings.mask[0])
            else:
                couple = (self.settings.mask[idx], self.settings.mask[idx + 1])
            
            dparc.setAllChannelsToFloat(self.arc)
            data = {}

            # set masks for current channel pair
            self.settings.mask_to_bias = [couple[0]]
            self.settings.mask_to_gnd = [couple[1]]
            print(f'mask:{self.settings.mask_to_bias}')
            self.settings.mask_to_read_i = [couple[0], couple[1]]

            biasedMask = dparc.biasMask(self.v_read-bias_calibration[idx], 0, self.settings)
            """timestamp_sample, voltage_sample, current_sample = dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )"""
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            timestamp_sample, voltage_sample, current_sample = dparc.measureAVG(
                self.arc, start_prog, biasedMask, self.settings,100
            )

            if idx == 0:
                timestamp = np.array([timestamp_sample])
                voltage = np.array(voltage_sample)
                current = np.array(current_sample)
            else:
                timestamp = np.append(timestamp, timestamp_sample)
                voltage = np.vstack((voltage, voltage_sample))
                current = np.vstack((current, current_sample))
        """ couple = (self.settings.mask[-1], self.settings.mask[0])

        dparc.setAllChannelsToFloat(self.arc)
        data = {}

        # set masks for current channel pair
        self.settings.mask_to_bias = [couple[0]]
        self.settings.mask_to_gnd = [couple[1]]
        self.settings.mask_to_read_i = [couple[0], couple[1]]
        biasedMask = dparc.biasMask(self.v_read, 0, self.settings)

        timestamp_sample, voltage_sample, current_sample = dparc.measureFastAVG(
            500, self.arc, start_prog, biasedMask, self.settings
        )
        timestamp = np.append(timestamp, timestamp_sample)
        voltage = np.vstack((voltage, voltage_sample))
        current = np.vstack((current, current_sample)) """
        dparc.setAllChannelsToFloat(self.arc)

        for sample in range(np.shape(voltage)[0]):
            data_row = dp.measureToStr(
                timestamp[sample],
                voltage[sample],
                current[sample],
                self.settings.mask,
            )
            dp.fileUpdate(data_file, data_row)

        data_file.close()

        # connects all channels to GND once the routine is finished
        self.arc.connect_to_gnd(self.settings.mask).execute()
        dp.plot_volt_diff(
            f"{self.session.savepath}/{self.session.sample}/{data_filename}.txt",
            self.v_read,
        )

        return
    def runFastAVG(self):
        """
        Performs the routine of Tomography.
        """
        print(str(self.name) + " has started.")

        # ----------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()
        # --------------------- routine ----------------------------------------

        # start all useful timers
        start_prog = time.time()
        # loop over all possible channel pairs

        for idx in range(len(self.settings.mask)):
            if idx == len(self.settings.mask) - 1:
                couple = (self.settings.mask[idx], self.settings.mask[0])
            else:
                couple = (self.settings.mask[idx], self.settings.mask[idx + 1])

            dparc.setAllChannelsToFloat(self.arc)
            data = {}

            # set masks for current channel pair
            self.settings.mask_to_bias = [couple[0]]
            self.settings.mask_to_gnd = [couple[1]]
            self.settings.mask_to_read_i = [couple[0], couple[1]]
            biasedMask = dparc.biasMask(self.v_read, 0, self.settings)
            """timestamp_sample, voltage_sample, current_sample = dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )"""
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            timestamp_sample, voltage_sample, current_sample = dparc.measureFastAVG(
            1, self.arc, start_prog, biasedMask, self.settings
            )

            if idx == 0:
                timestamp = np.array([timestamp_sample])
                voltage = np.array(voltage_sample)
                current = np.array(current_sample)
            else:
                timestamp = np.append(timestamp, timestamp_sample)
                voltage = np.vstack((voltage, voltage_sample))
                current = np.vstack((current, current_sample))
        """ couple = (self.settings.mask[-1], self.settings.mask[0])

        dparc.setAllChannelsToFloat(self.arc)
        data = {}

        # set masks for current channel pair
        self.settings.mask_to_bias = [couple[0]]
        self.settings.mask_to_gnd = [couple[1]]
        self.settings.mask_to_read_i = [couple[0], couple[1]]
        biasedMask = dparc.biasMask(self.v_read, 0, self.settings)

        timestamp_sample, voltage_sample, current_sample = dparc.measureFastAVG(
            500, self.arc, start_prog, biasedMask, self.settings
        )
        timestamp = np.append(timestamp, timestamp_sample)
        voltage = np.vstack((voltage, voltage_sample))
        current = np.vstack((current, current_sample)) """
        dparc.setAllChannelsToFloat(self.arc)

        for sample in range(np.shape(voltage)[0]):
            data_row = dp.measureToStrFastAVG(
                timestamp[sample],
                voltage[sample],
                current[sample],
                self.settings.mask,
            )
            dp.fileUpdate(data_file, data_row)

        data_file.close()

        # connects all channels to GND once the routine is finished
        self.arc.connect_to_gnd(self.settings.mask).execute()
        dp.plot_volt_diff(
            f"{self.session.savepath}/{self.session.sample}/{data_filename}.txt",
            self.v_read,
        )

        return