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

class ConductivityMatrix(Experiment):
    """
    Defines conductivity matrix analysis routine.

    This experiment performs a current reading on all pairs of electrodes.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)

        self.sample_time = 0.5
        self.v_read = 0.05
        self.n_reps_avg = 10
        self.script = "ConductivityMatrix"

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
        """Writes Log file for Conductivity Matrix measurement routine"""
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
        Performs the routine of Conductivity Matrix.
        """
        print(str(self.name) + " has started.")

        # ----------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()
        # --------------------- routine ----------------------------------------
        ch_pairs = dp.combinations(self.settings.mask)
        """ print(f"channel pairs: {ch_pairs}") """
        # start all useful timers
        start_prog = time.time()
        # loop over all possible channel pairs

        for couple in ch_pairs:
            dparc.setAllChannelsToFloat(self.arc)
            data = {}

            # set masks for current channel pair
            self.settings.mask_to_bias = [couple[0]]
            self.settings.mask_to_gnd = [couple[1]]
            self.settings.mask_to_read_i = [couple[0], couple[1]]
            self.settings.mask_to_read_v = [couple[0], couple[1]]
            """ print(f"masktov:{self.settings.mask_to_read_v}")
            print(f"masktoi:{self.settings.mask_to_read_i}") """
            i_1 = np.array([])
            i_2 = np.array([])
            v_1 = np.array([])
            v_2 = np.array([])
            t = np.array([])

            # retrieve indexes of selected channels to write correctly on file
            index_ch1_temp = np.where(self.settings.mask == couple[0])
            index_ch2_temp = np.where(self.settings.mask == couple[1])
            index_ch1 = index_ch1_temp[0][0]
            index_ch2 = index_ch2_temp[0][0]
            """ print(f"index cha 1 and 2: {index_ch1}, {index_ch2}") """
            biasedMask = dparc.biasMask(self.v_read, 0.00, self.settings)

            # loop for measurements
            c_,a_,b_ = dparc.measure(
                        arc= self.arc,
                        start_prog= start_prog,
                        biasedMask= biasedMask,
                        settings= self.settings,
                        
            )
            timestamp, voltage, current = dparc.measure(
                        arc= self.arc,
                        start_prog= start_prog,
                        biasedMask= biasedMask,
                        settings= self.settings,
                        
            )
            """ print(f"voltage: {voltage}")
            print(f"current: {current}") """
            t = np.append(t, timestamp)
            i_1 = np.append(i_1, current[self.settings.mask_to_read_i[0]])
            v_1 = np.append(v_1, voltage[0])
            i_2 = np.append(i_2, current[self.settings.mask_to_read_i[1]])
            v_2 = np.append(v_2, voltage[1])

            # compute and write on file median values for selected channels,
            # nan for non selected channels
            # t_avg = t
            # i_1_avg = np.median(i_1)
            # v_1_avg = np.median(v_1)
            # i_2_avg = np.median(i_2)
            # v_2_avg = np.median(v_2)

            data_to_print = np.empty(len(self.settings.mask) * 2 + 1)
            data_to_print[:] = np.nan  # fill with nans
            data_to_print[0] = t  # substitute desired values

            data_to_print[index_ch1 * 2 + 2] = i_1
            data_to_print[index_ch1 * 2 + 1] = v_1
            data_to_print[index_ch2 * 2 + 2] = i_2
            data_to_print[index_ch2 * 2 + 1] = v_2
            data_to_write = ""

            for element in data_to_print:
                data_to_write += str(element) + " "
            """ print(f"data to write: {data_to_write}") """
            dp.fileUpdate(data_file, data_to_write)  # write on file

        data_file.close()

        # reopen file to elaborate data
        dataMat = dp.txtToMatrix(
            f"{self.session.savepath}/{self.session.sample}/" + data_filename + ".txt"
        )
        GMat = np.full(
            (len(self.settings.mask), len(self.settings.mask)), np.nan
        )  # filled with nans

        # assign values to correct matrix indexes
        for row in dataMat:
            non_nan_indexes = []
            for i in range(len(row)):
                if not np.isnan(row[i]):
                    non_nan_indexes.append((row[i], i))
            non_nan_indexes = non_nan_indexes[1:]

            v1 = non_nan_indexes[0][0]
            v2 = non_nan_indexes[2][0]
            i2 = non_nan_indexes[3][0]

            # fill matrix
            GMat[int(non_nan_indexes[1][1] / 2) - 1][
                int(non_nan_indexes[3][1] / 2) - 1
            ] = (abs(i2 / (v2 - v1)) / 7.7e-5)

        dp.plotMatrix(
            matrix=GMat,
            rowIdx=self.settings.mask,
            colIdx=self.settings.mask,
            title=f"Conductivity Map, {self.session.sample}",
            xlabel="channel",
            ylabel="channel",
            colorlabel="GNORM [G0]",
            save=True,
            filename=f"{self.session.savepath}/{self.session.sample}/"
            + data_filename
            + ".png",
        )

        # connects all channels to GND once the routine is finished
        self.arc.connect_to_gnd(self.settings.mask)
        self.arc.execute()
        return