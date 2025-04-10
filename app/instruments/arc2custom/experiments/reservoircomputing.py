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

class ReservoirComputing(Experiment):
    """
    Reservoir Computing routine.


    ex:
    mem=MemoryCapacity()
    mem.run(v_bias_vector = dp.generate_random_array(100,0,1))

    This experiment feeds a random generated waveform in input
    and evaluates the memory capacity of the network.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)
        self.sample_time = 0.5
        self.v_read = 0.05
        self.n_reps_avg = 1
        self.script = "ReservoirComputing"

    def setMeasurement(
        self,
        sample_time: float,
        v_read: float,
        n_reps_avg: float,
    ):
        self.sample_time = sample_time
        self.v_read = v_read
        self.n_reps_avg = n_reps_avg

    def writeLogFile(self, f, v_bias_matrix):
        """Writes Log file for Reservoir Computing routine"""
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
        f.write(f"The matrix in input is the following: /n {str(v_bias_matrix)}")
        f.write("All other channels are set to floating each time.\n")

        f.flush()
        os.fsync(f.fileno())
        f.close()
        return

    def initializeFiles(self, v_bias_matrix, mat_class):
        self.session.num = (
            dp.findMaxNum(f"{self.session.savepath}/{self.session.sample}") + 1
        )
        self.session.dateUpdate()
        data_filename = f"{str(self.session.num).zfill(3)}_{self.session.lab}_{self.session.sample}_{self.session.cell}_{self.script}{mat_class}_{self.session.date}"
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
        self.writeLogFile(log_file, v_bias_matrix)
        return data_file, data_filename

    def run(self, v_bias_matrix, mat_class):
        """
        Performs reservoir routine.
        """

        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles(v_bias_matrix, mat_class)

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        try:
            # cycle over the samples

            for sample_step, v_bias in enumerate(v_bias_matrix):
                # set sleep until next sample for all iterations but the first

                biasedMask = dp.vecToTuples(self.settings.mask_to_bias, v_bias * 3)

                timestamp_sample, voltage_sample, current_sample = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                self.arc.delay(int(self.sample_time * (10**9)))
                if sample_step == 0:
                    timestamp = np.array([timestamp_sample])
                    voltage = np.array(voltage_sample)
                    current = np.array(current_sample)
                else:
                    timestamp = np.append(timestamp, timestamp_sample)
                    voltage = np.vstack((voltage, voltage_sample))
                    current = np.vstack((current, current_sample))
                # print the measurement progress at the end of the iteration
                # if at least 10 seconds have passed or if the measurement is completed

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
            return
        except KeyboardInterrupt:
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
        return