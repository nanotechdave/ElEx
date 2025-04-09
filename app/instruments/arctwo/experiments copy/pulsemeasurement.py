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

class PulseMeasurement(Experiment):
    """
    Defines Pulse Measurement routine.

    This experiment stimulates with a pulse on the bias channels, and reads
    continuously with a defined sample interval.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)

        self.sample_time = 0.5
        self.pre_pulse_time = 10
        self.pulse_time = 10
        self.post_pulse_time = 60
        self.pulse_voltage = [1, 2, 3, 4, 5]
        self.interpulse_voltage = 0.01
        self.script = "PulseMeasurement"

    def setMeasurement(
        self,
        sample_time: float,
        pre_pulse_time: float,
        pulse_time: float,
        post_pulse_time: float,
        pulse_voltage: list[float],
        interpulse_voltage: float,
    ):
        """Updates object measurement settings based on arguments"""
        self.sample_time = sample_time
        self.pre_pulse_time = pre_pulse_time
        self.pulse_time = pulse_time
        self.post_pulse_time = post_pulse_time
        self.pulse_voltage = pulse_voltage
        self.interpulse_voltage = interpulse_voltage

        self.settings.v_tuple = dp.pulseGenerator(
            self.sample_time,
            self.pre_pulse_time,
            self.pulse_time,
            self.post_pulse_time,
            self.pulse_voltage,
            self.interpulse_voltage,
        )
        self.settings.vBiasVec, self.settings.vTimes = dp.tuplesToVec(
            self.settings.v_tuple
        )
        return

    def writeLogFile(self, f):
        """Writes a log in file f relative to pulse experiment"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"self: {self.name}, {self.script} script. \n\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Pre pulse time: {self.pre_pulse_time} \n")
        f.write(f"Pulse time: {self.pulse_time} \n")
        f.write(f"Post pulse time: {self.post_pulse_time} \n")
        f.write(f"Pulse voltage: {self.pulse_voltage} \n")
        f.write(f"Interpulse voltage: {self.interpulse_voltage} \n\n")

        f.write("USED CHANNELS: \n\n")
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
        return data_file, data_filename

    def run(self, plot: bool = True):
        """
        Performs the routine of PulseMeasurement.
        Saves on file every iteration.
        Minimum sample time:0.2s
        """
        if self.sample_time < 0.2:
            sys.exit(
                "SYSTEM EXIT: Sample time too short, use runFast() or increase Sample time."
            )

        print(str(self.name) + " has started.")

        # ----------------- files initialization ------------------------------
        data_file, data_filename = self.initializeFiles()
        # ---------------------- routine --------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        T_vec = np.array([])
        I_vec = np.array([])
        Vdiff_vec = np.array([])
        V_end_vec = np.array([])
        V0_vec = np.array([])
        T_vec = np.array([])
        G_vec = np.array([])

        try:
            if plot:
                fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4 = dp.plotGen(
                    T_vec, I_vec, Vdiff_vec, G_vec
                )

            dparc.setAllChannelsToFloat(self.arc)
            
            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.01
                    )
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)

                # bias the channels and perform measurement
                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp, voltage, current = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                data_row = dp.measureToStr(
                    timestamp, voltage, current, self.settings.mask
                )
                dp.fileUpdate(data_file, data_row)

                if plot:
                    i_0 = current[self.settings.mask_to_read_i[-1]]
                    v_0 = voltage[0]
                    v_end = voltage[-1]
                    I_vec = np.append(I_vec, i_0)
                    V_end_vec = np.append(V_end_vec, v_end)
                    V0_vec = np.append(V0_vec, v_0)
                    Vdiff_vec = np.append(Vdiff_vec, v_0 - v_end)
                    T_vec = np.append(T_vec, timestamp)
                    lastG = abs(i_0 / (v_0 - v_end)) / 7.748e-5
                    G_vec = np.append(G_vec, lastG)
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

                # print the mearusement progress at the end of the interation
                # if at least 10 seconds have passed or if the measurement is completed
                if time.time() - progress_timer > 10 or sample_step + 1 == len(
                    self.settings.vBiasVec
                ):
                    progress = round(
                        (time.time() - start_prog) / self.settings.vTimes[-1] * 100
                    )
                    if progress > 100:
                        progress = 100
                    print(
                        f"Measurement progress: {progress}%. Time from start: "
                        + str(round(time.time() - start_prog))
                        + " seconds."
                    )
                    progress_timer = time.time()

            if plot:
                fig.savefig(
                    f"{self.session.savepath}/{self.session.sample}/"
                    + data_filename
                    + ".png"
                )
            data_file.close()
            # float all once the routine is finished
            dparc.setAllChannelsToFloat(self.arc)
            return

        except KeyboardInterrupt:
            if plot:
                fig.savefig(
                    f"{self.session.savepath}/{self.session.sample}/"
                    + data_filename
                    + ".png"
                )
            data_file.close()
            dparc.setAllChannelsToFloat(self.arc)

    def runAVG(self, nAVG: int = 32, plot: bool = True):
        """
        Performs the routine of PulseMeasurement.
        Saves on file every iteration.
        Minimum sample time:0.2s
        """
        if self.sample_time < 0.2:
            sys.exit(
                "SYSTEM EXIT: Sample time too short, use runFast() or increase Sample time."
            )

        print(str(self.name) + " has started.")

        # ----------------- files initialization ------------------------------
        data_file, data_filename = self.initializeFiles()

        # ---------------------- routine --------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        T_vec = np.array([])
        I_vec = np.array([])
        Vdiff_vec = np.array([])
        V_end_vec = np.array([])
        V0_vec = np.array([])
        T_vec = np.array([])
        G_vec = np.array([])

        try:
            if plot:
                fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4 = dp.plotGen(
                    T_vec, I_vec, Vdiff_vec, G_vec
                )

            dparc.setAllChannelsToFloat(self.arc)

            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.04
                    )
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)

                # bias the channels and perform measurement
                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp, voltage, current = dparc.measureFastAVG(
                    nAVG, self.arc, start_prog, biasedMask, self.settings
                )
                data_row = dp.measureToStrFastAVG(
                    timestamp, voltage, current, self.settings.mask
                )
                dp.fileUpdate(data_file, data_row)

                if plot:
                    i_0 = current[self.settings.mask_to_read_i[-1]]
                    v_0 = voltage[self.settings.mask_to_read_i[0]]
                    v_end = voltage[self.settings.mask_to_read_i[-1]]
                    I_vec = np.append(I_vec, i_0)
                    V_end_vec = np.append(V_end_vec, v_end)
                    V0_vec = np.append(V0_vec, v_0)
                    Vdiff_vec = np.append(Vdiff_vec, v_0 - v_end)
                    T_vec = np.append(T_vec, timestamp)
                    lastG = abs(i_0 / (v_0 - v_end)) / 7.748e-5
                    G_vec = np.append(G_vec, lastG)

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

                # print the mearusement progress at the end of the interation
                # if at least 10 seconds have passed or if the measurement is completed
                if time.time() - progress_timer > 10 or sample_step + 1 == len(
                    self.settings.vBiasVec
                ):
                    progress = round(
                        (time.time() - start_prog) / self.settings.vTimes[-1] * 100
                    )
                    if progress > 100:
                        progress = 100
                    print(
                        f"Measurement progress: {progress}%. Time from start: "
                        + str(round(time.time() - start_prog))
                        + " seconds."
                    )
                    progress_timer = time.time()

            if plot:
                fig.savefig(
                    f"{self.session.savepath}/{self.session.sample}/"
                    + data_filename
                    + ".png"
                )
            data_file.close()
            # float all once the routine is finished
            dparc.setAllChannelsToFloat(self.arc)
            return

        except KeyboardInterrupt:
            if plot:
                fig.savefig(
                    f"{self.session.savepath}/{self.session.sample}/"
                    + data_filename
                    + ".png"
                )
            data_file.close()
            dparc.setAllChannelsToFloat(self.arc)

    def runFast(self):
        """
        Performs the routine of PulseMeasurement.
        Saves on file ONLY at the end of the routine.
        Minimum sample time:0.04s
        """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        try:
            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    self.arc.wait()
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.01
                    )
                    if waitfor < 0:
                        waitfor = 0
                    self.arc.delay(int(waitfor * (10**9))).execute()

                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp_sample, voltage_sample, current_sample = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
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

    def runFastAVG(self, nAVG: int = 1):
        """
        Performs the routine of PulseMeasurement.
        Saves on file ONLY at the end of the routine.
        Minimum sample time:0.04s
        """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        progress_timer = time.time()
        data = {}
        dparc.setAllChannelsToFloat(self.arc)
        try:
            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    self.arc.wait()
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.02
                    )
                    if waitfor < 0:
                        waitfor = 0
                    self.arc.delay(int(waitfor * (10**9))).execute()

                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp_sample, voltage_sample, current_sample = dparc.measureFastAVG(
                    nAVG, self.arc, start_prog, biasedMask, self.settings
                )
                if sample_step == 0:
                    timestamp = np.array([timestamp_sample])
                    voltage = np.array(voltage_sample)
                    current = np.array(current_sample)
                else:
                    timestamp = np.append(timestamp, timestamp_sample)
                    voltage = np.vstack((voltage, voltage_sample))
                    current = np.vstack((current, current_sample))

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