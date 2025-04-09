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
        f.write(f"TIME: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"LAB: {self.session.lab}\n")
        f.write(f"SAMPLE: {self.session.sample}\n")
        f.write(f"CELL: {self.session.cell}\n")
        f.write(f"EXPERIMENT: {self.name}, {self.script} script. \n\n")
        if self.session.useDbMap:
            f.write(f"MAPPING of output file: Daughterboard\n")
        else:
            f.write(f"MAPPING of output file: Arc Internal Channels\n")

        f.write("Experiment parameters:\n\n")
        f.write(f"Sample time: {self.sample_time} \n")
        f.write(f"Start voltage: {self.start_voltage} \n")
        f.write(f"End voltage: {self.end_voltage} \n")
        f.write(f"Voltage step: {self.voltage_step} \n")

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
                data_row = self.measureToStr(
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