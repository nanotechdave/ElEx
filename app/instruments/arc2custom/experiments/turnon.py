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

class TurnOn(Experiment):
    """
    Defines TurnOn acquisition routine.

    This experiment Turns on the sample, and launches a fast train of voltage readings.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)
        self.sample_time = 0.01
        self.v_read = 0.05
        self.n_reps_avg = 1
        self.script = "TurnOn"

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
        """Writes Log file for TurnOn measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
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
        """ """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------

        progress_timer = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        try:
            biasedMask = dparc.biasMask(self.v_read, 0, self.settings)
            self.arc.config_channels(biasedMask, base=None)  # set channels
            start_prog = time.time()
            self.arc.generate_vread_train(
                self.settings.mask_to_read_v, True, 1000, 1000
            ).execute()
            voltages = []
            voltages = self.arc.get_iter(pyarc2.DataMode.All, pyarc2.ReadType.Voltage)
            voltarr = np.array([np.array(lst, dtype=float) for lst in voltages])

            currents = np.full(64, np.nan)
            timestamp = time.time() - start_prog  # get the sample timestamp

            dparc.setAllChannelsToFloat(self.arc)

            for idx, sample in enumerate(voltarr):
                timestamp = timestamp + 0.00032
                data_row = dp.measureToStrFastAVG(
                    timestamp,
                    sample[0],
                    currents,
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)
            data_file.close()
            return
        except KeyboardInterrupt:
            dparc.setAllChannelsToFloat(self.arc)

            for idx, sample in enumerate(voltarr):
                timestamp = timestamp + 0.00032
                data_row = dp.measureToStrFastAVG(
                    timestamp,
                    sample[0],
                    currents,
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)
            data_file.close()