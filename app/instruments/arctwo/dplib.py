# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:15:57 2023

@author: Davide Pilati
"""

import os
import random
#import winsound

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import signal
import tomli



def first_non_nan(vector: np.array):
    for value in vector:
        if not math.isnan(value):
            return value
    return None

def generate_random_array(x, vmin, vmax):
    return np.round(np.random.uniform(vmin, vmax, x), 3)

def loadToml(filename) -> dict:
        """Load toml data from file"""

        with open(filename, "rb") as f:
            toml_data: dict = tomli.load(f)
            return toml_data
        
def generate_step(timeup, timedown, time_pre_step, vUp, vDwn, sampling_time):
    num_samples_pre = int(time_pre_step / sampling_time)
    num_samples_up = int(timeup / sampling_time)
    num_samples_down = int(timedown / sampling_time)
    voltage_pre_values = np.full(num_samples_pre, vDwn)
    voltage_up_values = np.full(num_samples_up, vUp)
    voltage_down_values = np.full(num_samples_down, vDwn)
    num_samples = num_samples_pre + num_samples_up + num_samples_down 
    voltages = np.append(voltage_pre_values, voltage_up_values)
    voltages = np.append(voltages, voltage_down_values)
    timestamps = np.linspace(0, timeup+timedown+ time_pre_step, num_samples, endpoint=False)
    return vecToTuples(voltages, timestamps)

def generate_constant_voltage(total_time, sampling_time, voltage):
    """
    Generates two vectors: one with timestamps and the other with a constant voltage value.

    :param total_time: The total time duration.
    :param sampling_time: The time interval between samples.
    :param voltage: The voltage value for each sample.
    :return: A tuple of two vectors, one for timestamps and one for voltage values.
    """
    # Calculate the number of samples
    num_samples = int(total_time / sampling_time)

    # Generate timestamp vector
    timestamps = np.linspace(0, total_time, num_samples, endpoint=False)

    # Generate voltage vector
    voltage_values = np.full(num_samples, voltage)

    return vecToTuples(voltage_values, timestamps)


def isStable(
    vector: np.array,
    avg: float,
    tolerance: float,
    last_n_points: int,
):
    if last_n_points > len(vector):
        return False

    last_elements = vector[-last_n_points:]
    lower_bound = avg - tolerance
    upper_bound = avg + tolerance
    return np.all((last_elements >= lower_bound) & (last_elements <= upper_bound))


def sineGenerator(amplitude, periods, frequency, dc_bias, sample_rate):
    """
    Generates a sine wave based on the specified parameters.

    :param amplitude: Amplitude of the sine wave.
    :param periods: Number of periods of the sine wave to generate.
    :param frequency: Frequency of the sine wave.
    :param sample_rate: Sample rate (samples per second).
    :return: NumPy array of tuples containing the sine wave samples and the timestamps.
    """
    # Calculate the total number of samples
    total_samples = int(periods * sample_rate / frequency)

    # Create an array of time values
    t = np.linspace(0, periods / frequency, total_samples, endpoint=False)

    # Generate the sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t) + dc_bias

    return vecToTuples(sine_wave, t)

def triangGenerator(amplitude, periods, frequency, dc_bias, sample_rate):
    """
    Generates a sine wave based on the specified parameters.

    :param amplitude: Amplitude of the sine wave.
    :param periods: Number of periods of the sine wave to generate.
    :param frequency: Frequency of the sine wave.
    :param sample_rate: Sample rate (samples per second).
    :return: NumPy array of tuples containing the sine wave samples and the timestamps.
    """
    # Calculate the total number of samples
    total_samples = int(periods * sample_rate / frequency)

    # Create an array of time values
    t = np.linspace(0, periods / frequency, total_samples, endpoint=False)

    # Generate the sine wave
    triang_wave = amplitude * signal.sawtooth(2 * np.pi * frequency * t,0.5) + dc_bias

    return vecToTuples(triang_wave, t)


def combinations(lst: list) -> list[tuple]:
    """
    Returns a list of all the possible unordered pairs of lst elements.

    ex. return -> [(v1,t1),(v2,t2)...]
    """
    result = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            result.append([lst[i], lst[j]])
    return result


def generate_random_matrix():
    # Initialize a 4x4 matrix with low bias (zeros)
    matrix = np.zeros((4, 4))

    # Generate a random number between 1 and 4
    matrix_class = random.randint(1, 4)

    if matrix_class == 1:
        # Horizontal line
        row = random.randint(0, 3)
        matrix[row, :] = 1
    elif matrix_class == 2:
        # Vertical line
        col = random.randint(0, 3)
        matrix[:, col] = 1
    elif matrix_class == 3:
        # Diagonal line from top left to bottom right
        matrix[range(4), range(4)] = 1
    else:
        # Diagonal line from top right to bottom left
        matrix[range(4), range(3, -1, -1)] = 1

    return matrix, matrix_class


def rampGenerator(
    start: float = 0,
    end: float = 20,
    voltage_step: float = 0.2,
    time_step: float = 0.5,
) -> list[tuple]:
    """
    Returns a list of tuples for the experiment settings based on input parameters.

    ex. return -> [(v1,t1),(v2,t2)...]
    """
    vBiasVec = np.round(np.arange(start, end + voltage_step, voltage_step), 4)
    timeVec = np.arange(0, len(vBiasVec) * time_step, time_step)
    timeVec = timeVec[: len(vBiasVec)]

    return vecToTuples(vBiasVec, timeVec)


def average_of_elements(list_of_lists):
    # Convert the input list of lists into a numpy array, filling with NaN values for lists of different lengths
    arr = np.array([np.array(lst, dtype=float) for lst in list_of_lists])

    # Use numpy's nanmean function to calculate the average for each column, ignoring NaN values
    avg_list = np.nanmean(arr, axis=0)

    return avg_list


def measureToStr(timestamp, voltage, currentSample, mask):
    data_row = str(timestamp)
    for idx, channel in enumerate(mask):
        data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[channel])])
    return data_row


def measureToStrFastAVG(timestamp, voltage, currentSample, mask):
    data_row = str(timestamp)
    for idx, channel in enumerate(mask):
        data_row = " ".join(
            [data_row, str(voltage[channel]), str(currentSample[channel])]
        )
    return data_row


def pulseGenerator(
    sample_time: float = 0.5,
    pre_pulse_time: float = 10,
    pulse_time: float = 10,
    post_pulse_time: float = 60,
    pulse_voltage: list[float] = [1, 2, 3, 4, 5],
    interpulse_voltage: float = 0.01,
) -> list[tuple]:
    """
    Takes in input the pulse settings, returns a list of tuples for the experiment settings.

    ex. return -> [(v1,t1),(v2,t2)...]
    """
    vBiasVec_constr = []

    for step in range(len(pulse_voltage)):
        vBiasVec_prepulse = np.repeat(
            interpulse_voltage, round(pre_pulse_time / sample_time, 2)
        )
        vBiasVec_pulse = np.repeat(
            pulse_voltage[step], round(pulse_time / sample_time, 2)
        )
        vBiasVec_postpulse = np.repeat(
            interpulse_voltage, round(post_pulse_time / sample_time, 2)
        )
        vBiasVec_constr_toapp = np.round(
            np.concatenate((vBiasVec_prepulse, vBiasVec_pulse, vBiasVec_postpulse)), 2
        )
        vBiasVec_constr = np.append(vBiasVec_constr, vBiasVec_constr_toapp)

    time_tot = (pre_pulse_time + pulse_time + post_pulse_time) * len(pulse_voltage)
    vTimes_constr = np.round(
        np.linspace(0, time_tot - sample_time, len(vBiasVec_constr)), 2
    )

    return vecToTuples(vBiasVec_constr, vTimes_constr)


def load_channel_map(toml_path="dendrites_fix32.toml"):
    """
    Loads the channel map from a TOML file and returns a dictionary
    mapping integer daughterboard pins to integer ARC channels.
    """
    config = tomli.load(f"mappings/{toml_path}")

    # The "db_pin_to_arc_channel" table is itself a dict with string keys in TOML.
    raw_map = config["db_pin_to_arc_channel"]

    # Convert string keys to integers, preserving values as is.
    db_pin_to_arc_channel = {int(k): v for k, v in raw_map.items()}

    return db_pin_to_arc_channel


def vecToTuples(vec1: list, timesvec: list) -> list[tuple]:
    """
    Takes in input two vectors and returns a list of tuples:

    tulpelist=[(vec1[0],timesvec[0]), (vec1[1],timesvec[1]), ...]
    """
    tuplelist = []
    for times in range(len(timesvec)):
        tuplelist.append((vec1[times], timesvec[times]))
    return tuplelist


def tuplesToVec(list_of_tuples: list[tuple]):
    """
    Takes in input a list of tuples and returns two vectors:
        vec1 with the first elements of the tuples
        vec2 with the second elements of the tuples
    """
    vec1 = []
    vec2 = []

    for tup in list_of_tuples:
        vec1.append(tup[0])
        vec2.append(tup[1])

    return vec1, vec2


""" def beepFinished(mario: bool = False):
    
    Emits a Super Mario sound if mario is true, a default sound otherwise.
    
    # SUPER MARIO BEEP
    if mario:
        winsound.Beep(1047, 300)
        winsound.Beep(784, 300)
        winsound.Beep(660, 300)
        winsound.Beep(880, 200)
        winsound.Beep(988, 200)
        winsound.Beep(932, 100)
        winsound.Beep(880, 200)
        winsound.Beep(784, 100)

    # DEFAULT SOUND
    else:
        winsound.PlaySound("*", winsound.SND_ALIAS)
    return """


def ensureDirectoryExists(path: str):
    """Ensures the path directory exists.
    If it does not exists it creates one."""
    if not os.path.exists(path):
        os.makedirs(path)
    return


def fileInit(savepath, filename, header):
    """Initializes a file with a header.
    Header can be empty, in this case no line gets printed."""
    try:
        f = open(f"{savepath}/{filename}.txt", "w")
    except:
        os.makedirs(f"{savepath}")
        f = open(f"{savepath}/{filename}.txt", "w")
        f.close()
        f = open(f"{savepath}/{filename}.txt", "w")

    if header:
        f.write(f"{header}\n")
    return f


def fileUpdate(f, data: str):
    """Updates a file appending 'data'"""
    dataRow = "".join(str(i[-1]) for i in data)
    f.write(f"{dataRow}\n")
    f.flush()
    os.fsync(f.fileno())
    return


def concat_vectors(vectors: list[list]) -> list:
    """concatenates a list of arrays in one array without repetitions"""
    unique_elements = set()

    for vector in vectors:
        for element in vector:
            unique_elements.add(element)

    unique_elements = sorted(list(unique_elements))

    return unique_elements


def findMaxNum(path: str) -> int:
    """Returns the highest number which a filename begins with in the path directory."""
    max_number = 0

    for file in os.listdir(path):
        int_str = file.split("_")[0]
        if int(int_str) > max_number:
            max_number = int(int_str)

    return max_number


def txtToMatrix(filename):
    """Converts a .txt measurement formatted file into a matrix."""
    with open(filename, "r") as f:
        lines = f.readlines()
        matrix = np.zeros((len(lines) - 1, len(lines[0].split())))

        for i in range(1, len(lines)):
            matrix[i - 1] = [float(x) for x in lines[i].split()]

    return matrix


def plotMatrix(
    matrix: list[list[float]],
    rowIdx: list[int],
    colIdx: list[int],
    title: str,
    xlabel: str,
    ylabel: str,
    colorlabel: str,
    save: bool = False,
    filename: str = "matrix.png",
):
    """Plots a matrix following the conductivity matrix representation."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = mpl.cm.hot
    cmap.set_bad(color="grey")
    cax = ax.matshow(matrix, cmap=cmap, vmin=1e-3, vmax=7)
    # if a log plot is needed, substitute vmin, vmax with the following argument:
    # norm=mpl.colors.LogNorm(vmin=1e-9, vmax=10)

    # Set tick locations for x and y axes
    ax.set_xticks(np.arange(len(rowIdx)))
    ax.set_yticks(np.arange(len(colIdx)))

    # Set tick labels for x and y axes
    ax.set_xticklabels(rowIdx)
    ax.set_yticklabels(colIdx)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.colorbar(cax, orientation="vertical", label=colorlabel)

    if not save:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
    return


def plotResistance(v1, v2, I, time, ax3):
    """Sets the resistance plot"""
    (plot3,) = ax3.plot(time, (v1 - v2) / I)
    ax3.set_ylabel("Resistance (Ohm)")
    ax3.set_title("Resistance", loc="right")
    ax3.set_xlabel("Time (s)")
    return ax3, plot3


def plotG(v1, v2, I, time, ax3):
    """Sets the G plot"""
    (plot3,) = ax3.plot(time, I / (v1 - v2))
    ax3.set_ylabel("G/G0 [G0]")
    ax3.set_title("GNORM", loc="right")
    ax3.set_xlabel("Time (s)")
    return ax3, plot3


def plotIV(time, I, V, ax4):
    """Sets the IV plot"""
    (plot4,) = ax4.plot(V, I)
    ax4.set_ylabel("Current (A)")
    ax4.set_title("IV", loc="right")
    ax4.set_xlabel("Voltage (V)")
    return ax4, plot4


def plotGen(Time, I, V, G):
    """Generates a figure with 4 sublplots for real time IV measurement analysis"""
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=False, sharey=False, figsize=(13, 9)
    )
    (plot1,) = ax1.plot(Time, V)
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Voltage", loc="right")
    ax1.set_xlabel("Time (s)")
    (plot2,) = ax2.plot(Time, I)
    ax2.set_ylabel("Current (A)")
    ax2.set_title("Current", loc="right")
    ax2.set_xlabel("Time (s)")
    (plot3,) = ax3.plot(Time, G)
    ax3.set_ylabel("GNorm (G0)")
    ax3.set_title("GNORM", loc="right")
    ax3.set_xlabel("Time (s)")
    (plot4,) = ax4.plot(V, I)
    ax4.set_ylabel("Current (A)")
    ax4.set_title("IV", loc="right")
    ax4.set_xlabel("Voltage (V)")
    return fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4


def plotUpdate(Time, I, V, R, fig, ax1, plot1, ax2, plot2, ax3, plot3, ax4, plot4):
    """Updates a figure with 4 sublplots for real time IV measurement analysis"""
    plot1.set_xdata(Time)
    plot1.set_ydata(V)
    ax1.relim()
    ax1.autoscale_view()
    plot2.set_xdata(Time)
    plot2.set_ydata(I)
    ax2.relim()
    ax2.autoscale_view()
    plot3.set_xdata(Time)
    plot3.set_ydata(R)
    ax3.relim()
    ax3.autoscale_view()
    plot4.set_xdata(V)
    plot4.set_ydata(I)
    ax4.relim()
    ax4.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return


def plot_volt_diff(file_path, vread):
    # Read file
    df = pd.read_csv(file_path, sep=" ", skipinitialspace=True)

    # Get columns where current is NaN
    current_cols = df.columns[df.columns.str.contains("_I")]
    voltage_cols = df.columns[df.columns.str.contains("_V")]

    voltage_diff_vector = []
    transresistance_vector = []
    channel_pairs = []
    vbias_vector = []
    current_vector = []
    # Calculate voltage differences row by row
    for idx, row in df.iterrows():
        nan_current_cols = [col for col in current_cols if np.isnan(row[col])]
        not_nan_current_cols = [col for col in current_cols if not np.isnan(row[col])]
        
        voltage_cols_for_nan_current = [
            col.replace("_I[A]", "_V[V]") for col in nan_current_cols
        ]
        voltage_cols_for_not_nan_current = [
            col.replace("_I[A]", "_V[V]") for col in not_nan_current_cols
        ]
        current_cols_for_not_nan_current = [col for col in not_nan_current_cols]
        voltage_shifted = np.append(voltage_cols[idx:],voltage_cols[:idx])
        
        voltagerow= np.append(row[voltage_shifted], [row[voltage_shifted[0]]])
        
        
        voltage_diff_row = np.diff(voltagerow)
        voltage_diff_vector.extend(voltage_diff_row)
        
        current = row[current_cols_for_not_nan_current[0]]
        for _ in range(16):
            vbias_vector = np.append(vbias_vector, -voltage_diff_row[0])
            current_vector = np.append(current_vector, np.abs(current))
        
        transresistance_row = (
            voltage_diff_row / current if current != 0 else np.nan
        )
        
        transresistance_vector.extend(transresistance_row)
        

    # Plot
    path = file_path.rsplit(".", 1)[0]
    titleText = path.split("/")[-1].rsplit(".", 1)[0]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))  # 3 rows, 1 column

    fig.suptitle(f"{titleText}, {vread}V")
    # Plotting on the first subplot
    axs[0].plot(transresistance_vector)
    axs[0].set_xlabel("Measurement #")
    axs[0].set_ylabel("Transresistance / Ohm")
    axs[0].set_title(f"Transresistance")
    axs[0].grid(visible=True)
    
    axs[1].plot(voltage_diff_vector)
    axs[1].set_xlabel("Measurement #")
    axs[1].set_ylabel("Voltage / V")
    axs[1].set_title(f"Vsense")
    axs[1].grid(visible=True)
    
    axs[2].plot(current_vector)
    axs[2].set_xlabel("Measurement #")
    axs[2].set_ylabel("Current / A")
    axs[2].set_title(f"Currents")
    axs[2].grid(visible=True)

    plt.tight_layout()
    plt.savefig(f"{path}_Vdiff.png", dpi=300)
    plt.show()
    savepathtxt = "C:/Users/g.milano/Desktop/Davide/ARC/FTOTESTFIGURES"
    """np.savetxt(f'{savepathtxt}/trasres.txt', transresistance_vector)
    np.savetxt(f'{savepathtxt}/voltage_diff.txt', voltage_diff_vector)
    np.savetxt(f'{savepathtxt}/current.txt', current_vector)"""
