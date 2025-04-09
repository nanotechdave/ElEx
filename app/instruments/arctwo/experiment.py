# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:16:42 2023

@author: Davide Pilati
"""

import os
import sys
import time
from datetime import date

import numpy as np
import pyarc2

import arc2custom.dparclib as dparc
import arc2custom.dplib as dp
from arc2custom import measurementsettings, sessionmod


class Experiment:
    """
    Contains variables and functions for arc connection, measurement and saving settings.

    """

    def __init__(
        self, arc: pyarc2.Instrument, experiment_name: str, session: sessionmod.Session
    ):
        self.arc = arc
        self.name = experiment_name
        self.settings = measurementsettings.MeasurementSettings()
        self.session = session

    def headerInit(self, channel_vec: list) -> str:
        """
        Creates a header string in the following format:
        Time[s] 1V[V] 1I[A] 2V[V] 2I[A] ...
        """
        header = "Time[s] "
        for channel in channel_vec:
            header += str(channel) + "_V[V] " + str(channel) + "_I[A] "
        return header

    def testConnections(self):
        """
        Performs a routine to verify that all desired channels are connected to the sample.

        Bias channels are set to a 0.01V bias, all other channels
        are set to 0V independently of their purpose during the experiment.
        Then a measurement is performed, reading both voltage and current
        from each channel, in order to verify the connections.

        Obtained data gets printed on terminal, so that the user can choose
        to continue with the actual measurement routine or not.
        """
        start_prog = time.time()
        biasedMask = dparc.biasMaskTest(
            v_high=0.01,
            v_low=0,
            settings=self.settings,
        )
        dparc.setAllChannelsToFloat(self.arc)
        timestamp, voltage, current = dparc.measureTest(
            self.arc,
            start_prog,
            biasedMask,
            self.settings,
        )
        data_row = dp.measureToStr(timestamp, voltage, current, self.settings.mask)
        header = self.headerInit(self.settings.mask)
        dparc.setAllChannelsToFloat(self.arc)
        print(header)
        print(data_row)

        while True:
            user_input = input(
                "Do you want to proceed with the measurement? [Y/N]: "
            ).upper()
            if user_input == "Y":
                print("Proceding with the measurement.\n")
                return True
            elif user_input == "N":
                print("Exiting program")
                return False
            else:
                print("Invalid input, please enter either 'Y' or 'N'\n")

    def setMaskSettings(
        self,
        mask_to_gnd: list,
        mask_to_bias: list,
        mask_to_read_v: list,
        mask_to_read_i: list,
    ):
        """Updates object mask values based on arguments"""
        self.settings.mask_to_gnd = np.array(mask_to_gnd)
        self.settings.mask_to_bias = np.array(mask_to_bias)
        self.settings.mask_to_read_v = np.array(mask_to_read_v)
        self.settings.mask_to_read_i = np.array(mask_to_read_i)
        self.settings.mask = np.array(
            dp.concat_vectors(
                [
                    self.settings.mask_to_gnd,
                    self.settings.mask_to_bias,
                    self.settings.mask_to_read_i,
                    self.settings.mask_to_read_v,
                ]
            )
        )
        return


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


class IVMeasurement(Experiment):
    """
    Defines IV Measurement routine.

    This experiment stimulates with a ramp on the bias channels, and reads
    continuously with a defined sample interval.
    """

    def __init__(self, arc, experiment_name: str, session: sessionmod.Session):
        Experiment.__init__(self, arc, experiment_name, session)

        self.sample_time = 0.5
        self.start_voltage = 0
        self.end_voltage = 20
        self.voltage_step = 0.2
        self.script = "IVMeasurement"
        self.g_stop = 1.25
        self.g_interval = 0.5
        self.g_points = 10
        self.float_at_end = True

    def setMeasurement(
        self,
        sample_time: float,
        start_voltage: float,
        end_voltage: float,
        voltage_step: float,
        g_stop: float,
        g_interval: float,
        g_points: int,
        float_at_end: bool,
    ):
        self.sample_time = sample_time
        self.start_voltage = start_voltage
        self.end_voltage = end_voltage
        self.voltage_step = voltage_step
        self.g_stop = g_stop
        self.g_interval = g_interval
        self.g_points = g_points
        self.float_at_end = float_at_end

        self.settings.v_tuple = np.array(
            dp.rampGenerator(start_voltage, end_voltage, voltage_step, sample_time)
        )
        self.settings.vBiasVec, self.settings.vTimes = np.array(
            dp.tuplesToVec(self.settings.v_tuple)
        )

    def customWave(
        self,
        vbias_vector: np.array,
        timesteps_vector: np.array,
    ):
        self.settings.vBiasVec = vbias_vector
        self.settings.vTimes = timesteps_vector
        self.script = "Custom Wave Measurement"

    def customWave(
        self,
        voltage_times_tuple_array,
    ):
        self.settings.v_tuple = np.array(voltage_times_tuple_array)
        self.settings.vBiasVec, self.settings.vTimes = np.array(
            dp.tuplesToVec(self.settings.v_tuple)
        )
        self.script = "Custom Wave Measurement"

    def writeLogFile(self, f):
        """Writes Log file for IV measurement routine"""
        f.write(f"DATE: {self.session.date}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Start voltage: {self.start_voltage} \n")
        f.write(f"End voltage: {self.end_voltage} \n")
        f.write(f"Voltage step: {self.voltage_step} \n")

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
        Performs the routine of IVMeasurement.
        """
        print(str(self.name) + " has started.")

        # files initialization
        data_file, data_filename = self.initializeFiles()

        # start all useful timers
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

            if self.float_at_end:
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
                    ) / self.settings.meas_iterations
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)

                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                a_,b_,c_ = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                timestamp, voltage, current = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                data_row = dp.measureToStr(
                    timestamp, voltage, current, self.settings.mask
                )
                dp.fileUpdate(data_file, data_row)
                i_0 = dp.first_non_nan(current)
                v_0 = voltage[0]
                v_end = 0
                I_vec = np.append(I_vec, i_0)
                V_end_vec = np.append(V_end_vec, v_end)
                V0_vec = np.append(V0_vec, v_0)
                Vdiff_vec = np.append(Vdiff_vec, v_0 - v_end)
                T_vec = np.append(T_vec, timestamp)
                lastG = abs(i_0 / (v_0 - v_end)) / 7.748e-5
                G_vec = np.append(G_vec, lastG)
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
                    print(lastG)
                if (
                dp.isStable(G_vec, self.g_stop, self.g_interval, self.g_points)
                ):
                    if plot:
                        fig.savefig(
                            f"{self.session.savepath}/{self.session.sample}/"
                            + data_filename
                            + ".png"
                        )
                    data_file.close()

                    # connects all channels to GND once the routine is finished
                    #self.arc.connect_to_gnd(self.settings.mask_to_bias)
                    #self.arc.execute()
                    return

                # print the measurement progress at the end of the iteration
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
                        "Measurement progress: "
                        + str(progress)
                        + "%. Time from start: "
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

            # connects all channels to GND once the routine is finished
            if self.float_at_end:
                self.arc.connect_to_gnd(self.settings.mask_to_bias)
                self.arc.execute()
            return

        except KeyboardInterrupt:
            if plot:
                fig.savefig(
                    f"{self.session.savepath}/{self.session.sample}/"
                    + data_filename
                    + ".png"
                )
            data_file.close()

            # connects all channels to GND once the routine is finished
            self.arc.connect_to_gnd(self.settings.mask_to_bias)
            self.arc.execute()

    def runFast(self):
        """
        Performs the routine of IVMeasurement.
        Saves on file ONLY at the end of the routine.
        Expected sample time: 0.015s
        """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        if self.float_at_end:
            dparc.setAllChannelsToFloat(self.arc)
        timestamp = np.array([])

        lastG = 0
        G = np.array([])
        try:
            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
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

                lastG = (
                    abs(
                        current_sample[self.settings.mask_to_read_i[-1]]
                        / (voltage_sample[0])
                    )
                    / 7.748e-5
                )
                G = np.append(G, lastG)
                # print(lastG)
                if (
                    dp.isStable(G, self.g_stop, self.g_interval, self.g_points)
                    and time.time() - start_prog > 10
                ):
                    # float all once the routine is finished
                    if self.float_at_end:
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
            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStr(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)
            data_file.close()
            # float all once the routine is finished
            if self.float_at_end:
                dparc.setAllChannelsToFloat(self.arc)
            return
        except KeyboardInterrupt:
            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStr(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )

                dp.fileUpdate(data_file, data_row)
            data_file.close()
            # float all once the routine is finished
            dparc.setAllChannelsToFloat(self.arc)

    def runFastAVG(self):
        """
        Performs the routine of IVMeasurement.
        Saves on file ONLY at the end of the routine.
        Expected sample time: 0.015s
        """
        print(str(self.name) + " has started.")

        # ---------------------- files initialization -------------------------
        data_file, data_filename = self.initializeFiles()

        # ----------------------- routine -------------------------------------
        start_prog = time.time()
        if self.float_at_end: 
            dparc.setAllChannelsToFloat(self.arc)
        timestamp = np.array([])

        lastG = 0
        G = np.array([])
        try:
            # cycle over the samples
            for sample_step, v_bias in enumerate(self.settings.vBiasVec):
                
                # set sleep until next sample for all iterations but the first
                if sample_step != 0:
                    waitfor = (
                        self.settings.vTimes[sample_step]
                        - time.time()
                        + start_prog
                        - 0.02
                    ) / self.settings.meas_iterations
                    if waitfor < 0:
                        waitfor = 0
                    time.sleep(waitfor)
                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                timestamp_sample, voltage_sample, current_sample = dparc.measureFastAVG(
                    1, self.arc, start_prog, biasedMask, self.settings
                )

                if sample_step == 0:
                    timestamp = np.array([timestamp_sample])
                    voltage = np.array(voltage_sample)
                    current = np.array(current_sample)
                else:
                    timestamp = np.append(timestamp, timestamp_sample)
                    voltage = np.vstack((voltage, voltage_sample))
                    current = np.vstack((current, current_sample))

                lastG = (
                    abs(
                        current_sample[self.settings.mask_to_read_i[-1]]
                        / (
                            voltage_sample[self.settings.mask_to_read_i[0]]
                            - voltage_sample[self.settings.mask_to_read_i[-1]]
                        )
                    )
                    / 7.748e-5
                )
                G = np.append(G, lastG)
                # print(lastG)
                if (
                    dp.isStable(G, self.g_stop, self.g_interval, self.g_points)
                    and time.time() - start_prog > 10
                ):
                    # float all once the routine is finished
                    if self.float_at_end:
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
            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStrFastAVG(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )
                dp.fileUpdate(data_file, data_row)
            data_file.close()
            # float all once the routine is finished
            if self.float_at_end:
                dparc.setAllChannelsToFloat(self.arc)
            return
        except KeyboardInterrupt:
            for sample in range(np.shape(voltage)[0]):
                data_row = dp.measureToStrFastAVG(
                    timestamp[sample],
                    voltage[sample],
                    current[sample],
                    self.settings.mask,
                )

                dp.fileUpdate(data_file, data_row)
            data_file.close()
            # float all once the routine is finished
            dparc.setAllChannelsToFloat(self.arc)


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
            biasedMask = dparc.biasMask(self.v_read, 0, self.settings)

            # loop for measurements
            c_,a_,b_ = dparc.measureFastAVG(
                1, self.arc, start_prog, biasedMask, self.settings
            )
            timestamp, voltage, current = dparc.measureFastAVG(
                1, self.arc, start_prog, biasedMask, self.settings
            )

            t = np.append(t, timestamp)
            i_1 = np.append(i_1, current[self.settings.mask_to_read_i[0]])
            v_1 = np.append(v_1, voltage[self.settings.mask_to_read_i[0]])
            i_2 = np.append(i_2, current[self.settings.mask_to_read_i[1]])
            v_2 = np.append(v_2, voltage[self.settings.mask_to_read_i[1]])

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
            timestamp_sample, voltage_sample, current_sample = dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            tr1,tr2,tr3= dparc.measure(
                self.arc, start_prog, biasedMask, self.settings
            )
            timestamp_sample, voltage_sample, current_sample = dparc.measureAVG(
                self.arc, start_prog, biasedMask, self.settings
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


class MemoryCapacity(Experiment):
    """
    Memory capacity evaluation routine.
    DOES NOT NEED setMeasurement() to work.
    DOES NEED v_bias_vector passed to run().

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
        self.script = "MemoryCapacity"

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
        """Writes Log file for Memory Capacity measurement routine"""
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

    def run(self, v_bias_vector):
        """
        Performs Memory Capacity routine.
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
            for sample_step, v_bias in enumerate(v_bias_vector):
                # set sleep until next sample for all iterations but the first

                biasedMask = dparc.biasMask(v_bias, 0, self.settings)
                self.arc.config_channels(
                    biasedMask, base=None
                ).execute()  # set channels
                #time.sleep(0.1)
                timestamp_sample, voltage_sample, current_sample = dparc.measure(
                    self.arc, start_prog, biasedMask, self.settings
                )
                # time.sleep(self.sample_time)
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


# ------------------------------ MAYBE USEFUL CODE FOR FAST PULSING --------------
# def runPulse(self, settings: MeasurementSettings):

#     print(str(self.name)+' has started.')

#     cell_data = dp.create_cell_data_structure(settings)
#     self.arc.connect_to_gnd(settings.mask_to_gnd)

#     # start all useful timers
#     start_prog = time.time()
#     save_timer = time.time()
#     progress_timer = time.time()

#     # cycle over the samples

#     for k in range(10000):
#         #voltage={}
#         for i in range(100):

#             self.arc.pulse_slice_fast_open([(13,4,0)],[40000,40000,40000,40000,40000,40000,40000,40000],False)
#             #delay is to be added as time between consecutive rising edges,
#             #atm a 4k ns is to be taken away from delay instruction,
#             #as the instrument takes approx. 4us to perform the instruction
#             #voltage[1]=self.arc.vread_channels(settings.mask_to_read, False).
#             #DELAY MUST BE AT LEAST EQUAL TO TIMING IN PULSE()
#             self.arc.delay(40000)

#         self.arc.execute()
#         #print(voltage)

#         #cell_data = measure(self.arc, sample_step, start_prog, biasedMask, cell_data, settings)


#     # # save every 5 minutes or when the measurement is finished
#     # if (time.time()-save_timer)>300 or sample_step>=len(settings.vBiasVec)-1:
#     #     saveCellOnTxt(cell_data, settings.mask_to_read, v_bias, 0)

#     # # print the mearusement progress at the end of the interation
#     # # if at least 10 seconds have passed or if the measurement is completed
#     # if time.time()-progress_timer>10 or sample_step+1==len(settings.vBiasVec):
#     #     progress=round((time.time()-start_prog)/settings.vTimes[-1]*100)
#     #     print('Measurement progress: '+str(progress)+'%. Time from start: '+str(round(time.time()-start_prog))+' seconds.')
#     #     progress_timer=time.time()
