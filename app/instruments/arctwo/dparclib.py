# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:17:18 2023

@author: Davide Pilati
"""

import pyarc2
from arc2custom import fwutils
import numpy as np
import time
import sys
from .experiments.experiment import Experiment
from . import measurementsettings
from . import dplib
import math
import os


ALL_CHANNELS = list(range(64))


def initialize_instrument(firmware_release: str = "efm03_20220905.bin"):
    """
    Initializes the ARC2 instrument by finding the ids and creating an Instrument object.
    """
    ids = pyarc2.find_ids()  # find the ids of the connected ARC2 instruments

    # raise an error if no instrument is found
    if not ids:
        sys.exit("Error: No ARC2 instrument found.")
    dirpath = os.path.dirname(os.path.realpath(__file__))
    firmware_pointer = f"{dirpath}./{firmware_release}"
    # create an Instrument object using the first id found and the firmware path
    arc = pyarc2.Instrument(
        ids[0],
        firmware_pointer,
    )
    arc.finalise_operation(pyarc2.IdleMode.SoftGnd)
    return arc


def biasMask(
    v_high: float,
    v_low: float,
    settings: measurementsettings.MeasurementSettings,
) -> list[tuple]:
    """
    Compiles biasedMask based on the masks of the settings. Channel is assigned
    to v_high if it is contained in mask_to_bias, connects it to v_low if it is
    contained in mask_to_gnd
    """
    biasedMask = []
    temp_mask = np.concatenate((settings.mask_to_gnd, settings.mask_to_bias), axis=0)
    temp_mask.sort()

    for channel in temp_mask:
        if channel in settings.mask_to_bias:
            biasedMask.append((channel, v_high))
        else:
            biasedMask.append((channel, v_low))

    return biasedMask


def biasMaskTest(
    v_high: float,
    v_low: float,
    settings: measurementsettings.MeasurementSettings,
) -> list[tuple]:
    """
    Compiles biasedMask based on the masks of the settings for testing operations.
    Channel is assigned to v_high if it is contained in mask_to_bias,
    connects it to v_low if it is contained in mask_to_gnd.
    """
    biasedMask = []
    temp_mask = settings.mask
    temp_mask.sort()

    for channel in temp_mask:
        if channel in settings.mask_to_bias:
            biasedMask.append((channel, v_high))
        else:
            biasedMask.append((channel, v_low))
    return biasedMask


def setAllChannelsToFloat(arc: pyarc2.Instrument) -> None:
    arc.connect_to_gnd([]).execute()
    arc.open_channels(ALL_CHANNELS).execute()

    return


def connectMask(arc: pyarc2.Instrument, biasedMask: list[tuple]) -> None:
    """
    Connects biases contained in biasedMask to the masked channels
    """
    arc.config_channels(biasedMask, base=None).execute()

    return


def measure(
    arc: pyarc2.Instrument,
    start_prog: float,
    biasedMask: list[tuple],
    settings: measurementsettings.MeasurementSettings,
) -> float:
    """
    Performs measurements and stores the data in three arrays: timestamp, voltages, currents.
    """

    arc.config_channels(biasedMask, base=None).execute()  # set channels

    voltage = arc.vread_channels(settings.mask_to_read_v, False)
    sampleTime = time.time()  # get the sample timestamp

    # read the current of the masked channels
    currentSample = arc.read_slice_open(settings.mask_to_read_i, False)

    # initialize and compile the string containing the acquired data
    timestamp = sampleTime - start_prog

    # for idx, channel in enumerate(settings.mask_to_read_v):
    #     data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[channel])])
    return timestamp, voltage, currentSample


def measureFastAVG(
    nAVG: int,
    arc: pyarc2.Instrument,
    start_prog: float,
    biasedMask: list[tuple],
    settings: measurementsettings.MeasurementSettings,
) -> float:
    """
    Performs a series of voltage and current using train reads,
    with each read spaced by 1uS. Returns the average of the reads.
    """
    
    arc.config_channels(biasedMask, base=None).execute()  # set channels
    
    arc.generate_vread_train(settings.mask_to_read_v, True, nAVG, 100).execute()
    voltages = []
    voltages = arc.get_iter(pyarc2.DataMode.All, pyarc2.ReadType.Voltage)
    voltage = dplib.average_of_elements(voltages)
    sampleTime = time.time()  # get the sample timestamp

    # read the current of the masked channels
    arc.generate_read_train(
        None, settings.mask_to_read_i, 0, nAVG, 100, False
    ).execute()
    currentSamples = []
    currentSamples = arc.get_iter(pyarc2.DataMode.All, pyarc2.ReadType.Current)
    currentSample = dplib.average_of_elements(currentSamples)

    # initialize and compile the string containing the acquired data
    timestamp = sampleTime - start_prog

    # for idx, channel in enumerate(settings.mask_to_read_v):
    #     data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[channel])])
    return timestamp, voltage[0], currentSample[0]




def measureAVG(
    arc: pyarc2.Instrument,
    start_prog: float,
    biasedMask: list[tuple],
    settings: measurementsettings.MeasurementSettings,
    nRep: int,
) -> str:
    """
    Perform measurements and store the data in the cell_data structure.
    """
    voltage_accumulator = []
    current_accumulator = []
    setAllChannelsToFloat(arc)
    arc.config_channels(biasedMask, base=None).execute()  # set channels
    v_trash= arc.vread_channels(settings.mask_to_read_v, False)
    current_reading_trash = arc.read_slice_open(settings.mask_to_read_i, False)
    for _ in range(nRep):
        voltage_reading = arc.vread_channels(settings.mask_to_read_v, False)
        voltage_accumulator.append(voltage_reading)
        
        current_reading = arc.read_slice_open(settings.mask_to_read_i, False)
        current_accumulator.append(current_reading)
        
    voltage_accumulator=np.array(voltage_accumulator)
    current_accumulator=np.array(current_accumulator)
    
    voltage = np.mean(voltage_accumulator, axis=0)
    sampleTime = time.time()  # get the sample timestamp

        # read the current of the masked channels
       
       

    currentSample = np.mean(
        current_accumulator,
        axis=0,
    )

    # initialize and compile the string containing the acquired data
    timestamp = sampleTime - start_prog

    # for idx, channel in enumerate(settings.mask_to_read_v):
    #     data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[channel])])
    return timestamp, voltage, currentSample


def measureVoltage(
    arc: pyarc2.Instrument,
    start_prog: float,
    settings: measurementsettings.MeasurementSettings,
) -> str:
    """
    Perform measurements and store the data in the cell_data structure.
    """

    voltage = arc.vread_channels(settings.mask_to_read_v, False)
    sampleTime = time.time()  # get the sample timestamp

    # initialize and compile the string containing the acquired data
    timestamp = sampleTime - start_prog

    # for idx, channel in enumerate(settings.mask):
    #     data_row = " ".join([data_row, str(voltage[idx]), "nan"])
    return timestamp, voltage


def measureTest(
    arc: pyarc2.Instrument,
    start_prog: float,
    biasedMask: list[tuple],
    settings: measurementsettings.MeasurementSettings,
) -> str:
    """
    Perform measurements and store the data in the cell_data structure.
    """
    setAllChannelsToFloat(arc)
    arc.config_channels(biasedMask, base=None).execute()  # set channels

    voltage = arc.vread_channels(settings.mask, False)
    sampleTime = time.time()  # get the sample timestamp

    # read the current of the masked channels
    currentSample = arc.read_slice_open(settings.mask, False)

    # initialize and compile the string containing the acquired data
    timestamp = sampleTime - start_prog

    # for idx, channel in enumerate(settings.mask):
    #     data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[channel])])

    setAllChannelsToFloat(arc)  # <- the program may end here

    return timestamp, voltage, currentSample


def calculate_resistance(
    voltage: list[float],
    currentSample: list[float],
    mask: list[int],
) -> list[float]:
    """
    Calculates the resistance from the data stored in the cell_data structure and returns it.
    """
    resistance = [0 for _ in range(len(mask))]
    for idx, channel in enumerate(mask):
        # calculate the resistance for each channel
        resistance[idx] = (voltage[0] - voltage[idx]) / currentSample[channel]
        return resistance


# ----------------------------UNUSED FUNCTIONS --------------------------------
def saveCellOnTxt(cell_data, mask, v_bias, sample_step):
    """
    Saves cell_data on .txt files.
    This function creates one file per channel with the following format (one line for each sample):

        [(channel, voltage, current, resistance, time),
         (channel, voltage, current, resistance, time),
         (channel, voltage, current, resistance, time),
         ...
         (channel, voltage, current, resistance, time)
         ]
    """
    for idx, channel in enumerate(mask):
        directory = "C:/Users/mcfab/Desktop/Measurements/"
        voltstring = str(v_bias)
        chstring = str(channel)
        iteration_string = str(sample_step)
        filename = iteration_string + "_" + voltstring + "_" + chstring

        with open(r"" + directory + filename + ".txt", "w") as file:
            file.write(np.array2string(cell_data[idx]))
    return


def maskFromCells(self):
    """
    Creates a mask based on the cells object of the QModule BaseModule in case the GUI is used
    """
    mask = []
    for cell in self.cells:
        (w, b) = (cell.w, cell.b)
        (high, low) = self.mapper.wb2ch[w][b]

        if high not in mask:
            mask.append(high)

        if low not in mask:
            mask.append(low)

    mask.sort()
    return mask


def vBiasVecCompiler(
    v_p: float, v_read: float, t_p: float, t_read: float, sample_interval: float
):
    """
    Compiles vBiasVec based on v_p, v_read, t_p, t_read, and sample_interval.
    vBiasVec will be a list of float representing the voltages to assign to the
    channels in mask_to_bias at regular time intervals i.e. sample_interval.
    """
    iterations_p = math.ceil((t_p) / sample_interval)
    iterations_read = math.ceil(t_read / sample_interval)
    vBiasVec = []

    for i in range(iterations_p):
        vBiasVec.append(v_p)
    for i in range(iterations_read):
        vBiasVec.append(v_read)

    return vBiasVec


def create_cell_data_structure(settings: measurementsettings.MeasurementSettings):
    """
    Create the structure for storing the cell data
    """
    _RET_DTYPE = [
        ("channel", "<u8"),
        ("bias", "<f8"),
        ("current", "<f4"),
        ("resistance", "<f4"),
        ("tstamp", "<f8"),
    ]  # define the data type of the numpy array
    cell_data = {}  # initialize an empty dictionary
    for idx in range(len(settings.mask)):
        cell_data[idx] = np.empty(
            shape=(settings.tot_iterations,), dtype=_RET_DTYPE
        )  # create an empty numpy array with the specified shape and data type for each channel
    return cell_data


def plotResistance(v1, v2, I, time, ax3):
    (plot3,) = ax3.plot(time, (v1 - v2) / I)
    ax3.set_ylabel("Resistance (Ohm)")
    ax3.set_title("Resistance")
    ax3.set_xlabel("Time (s)")
    return ax3, plot3


def plotIV(time, I, V, ax4):
    (plot4,) = ax4.plot(V, I)
    ax4.set_ylabel("Current (A)")
    ax4.set_title("IV")
    ax4.set_xlabel("Voltage (V)")
    return ax4, plot4


def plotUpdate(
    Time, I, V, R, fig, ax1, plot1, ax2, plot2, ax3, plot3
):  # , ax4, plot4):
    plot1.set_xdata(Time)
    plot1.set_ydata(V)
    ax1.relim()
    ax1.autoscale_view()
    plot2.set_xdata(Time)
    plot2.set_ydata(I)
    ax2.relim()
    ax2.autoscale_view()
    plot3.set_xdata(V)
    plot3.set_ydata(I)
    ax3.relim()
    ax3.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return
