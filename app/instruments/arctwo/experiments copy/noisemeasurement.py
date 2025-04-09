import os
import sys
import time
from datetime import date

import numpy as np
import pyarc2

from arc2custom import dparclib as dparc
from arc2custom import dplib as dp
from arc2custom import measurementsettings, sessionmod
from .experiment import Experiment

class NoiseMeasurement(Experiment):
    """
    Defines IV Measurement routine.

    This experiment stimulates with a ramp on the bias channels, and reads
    continuously with a defined sample interval.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)

        self.sample_time = 0.5
        self.duration = 10
        self.bias_voltage = 0
        self.script = "NoiseMeasurement"

    def setMeasurement(
        self,
        duration: float,
        bias_voltage: float,
    ):
        self.bias_voltage = bias_voltage
        self.duration = duration

        vVec = np.repeat(bias_voltage, round(duration / 0.015, 2))
        tVec = np.linspace(0, duration, len(vVec))

        self.settings.v_tuple = dp.vecToTuples(vVec, tVec)
        self.settings.vBiasVec, self.settings.vTimes = dp.tuplesToVec(
            self.settings.v_tuple
        )

    def writeLogFile(self, f):
        """Writes Log file for noise measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Bias voltage: {self.bias_voltage} \n")
        f.write(f"Duration: {self.duration} \n")

        f.write("USED CHANNELS: \n\n")
        f.write(f"All channels: {self.settings.mask} \n")
        f.write(f"Channels set to reference voltage: {self.settings.mask_to_gnd}\n")
        f.write(f"Channels set to bias voltage: {self.settings.mask_to_bias}\n")
        f.write(f"Voltage is read from channels: {self.settings.mask_to_read_v}\n")
        f.write("Current is not read.\n")
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
        data_filename_start = f"{str(self.session.num).zfill(3)}_{self.session.lab}_{self.session.sample}_{self.session.cell}_"
        data_filename_end = f"{self.script}_{self.session.date}"
        data_filename = data_filename_start + data_filename_end
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
        return data_file

    def runFast(self):
        """
        Performs the routine of PulseMeasurement.
        Saves on file ONLY at the end of the routine.
        Expected sample time: 0.015s
        """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        biasedMask = dparc.biasMask(self.bias_voltage, 0, self.settings)
        self.arc.config_channels(biasedMask, base=None).execute()

        # cycle over the samples
        for sample_step, v_bias in enumerate(self.settings.vBiasVec):
            data[sample_step] = dparc.measureVoltage(
                self.arc, start_prog, self.settings
            )

        for value in data.values():
            data_file.write(str(value) + "\r")

        data_file.close()
        # connects all channels to GND once the routine is finished
        self.arc.connect_to_gnd(self.settings.mask_to_bias)
        self.arc.execute()
        return