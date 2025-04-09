import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_test_split_time_series(data, target, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))

    data_train = data[:split_index, :]
    data_test = data[split_index:, :]

    target_train = target[:split_index]
    target_test = target[split_index:]

    return data_train, data_test, target_train, target_test


def kDelayAccuracy(input: np.array, k_delay: int, states: np.array):
    # Assumption: first 100 points of input are CUT and serve as a buffer

    lr = LinearRegression()

    states_train, states_test, target_train, target_test = train_test_split_time_series(
        states,
        input,
        test_size=0.2,
    )

    # Reshape your states matrix into a 2D array if it's not already
    """ states_train = states_train.reshape(-1, 1)
    states_test = states_test.reshape(-1, 1) """
    # Fit the model to your data
    lr.fit(states_train, target_train)

    # Now you can predict using the linear model:
    prediction_test = lr.predict(states_test)
    r2 = r2_score(target_test, prediction_test)
    mae = mean_absolute_error(target_test, prediction_test)
    mse = mean_squared_error(target_test, prediction_test)
    return mae, target_test, prediction_test


def calculate_memory_capacity(estimated_waveforms, target_waveforms):
    """
    Calculate the memory capacity of a system given the estimated and target waveforms for each delay.

    Parameters:
    estimated_waveforms (list of np.array): The estimated waveforms from the system for each delay.
    target_waveforms (list of np.array): The target waveforms for each delay.

    Returns:
    float: The memory capacity of the system.
    """
    assert len(estimated_waveforms) == len(
        target_waveforms
    ), "Input waveforms must be the same length"

    MemC = 0
    for estimated_waveform, target_waveform in zip(
        estimated_waveforms, target_waveforms
    ):
        # Calculate the covariance and variances
        covariance = np.cov(estimated_waveform, target_waveform)[0, 1]
        variance_estimate = np.var(estimated_waveform)
        variance_target = np.var(target_waveform)

        # Calculate the MC for this delay
        MC_k = covariance**2 / (variance_estimate * variance_target)

        # Add to the total MC
        MemC += MC_k

    return MemC


def generate_forgetting_curve(estimated_waveforms, target_waveforms):
    """
    Generate the forgetting curve of a system given the estimated and target waveforms for each delay.

    Parameters:
    estimated_waveforms (list of np.array): The estimated waveforms from the system for each delay.
    target_waveforms (list of np.array): The target waveforms for each delay.
    """
    assert len(estimated_waveforms) == len(
        target_waveforms
    ), "Input waveforms must be the same length"

    MC_values = []
    for estimated_waveform, target_waveform in zip(
        estimated_waveforms, target_waveforms
    ):
        # Calculate the covariance and variances
        covariance = np.cov(target_waveform, estimated_waveform)[0, 1]

        # plt.plot(target_waveform)
        # plt.plot(estimated_waveform)
        # plt.xlabel("Samples")
        # plt.ylabel("Voltage  [V]")
        # plt.legend("target", "predicted")
        # plt.show()
        variance_estimate = np.var(estimated_waveform)
        variance_target = np.var(target_waveform)
        # print(variance_estimate, variance_target)
        # Calculate the MC for this delay
        MC_k = covariance**2 / (variance_estimate * variance_target)

        # Add to the list of MC values
        MC_values.append(MC_k)

    # Plot the forgetting curve

    plt.plot(range(1, len(MC_values) + 1), MC_values)

    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    plt.title("Forgetting Curve")

    return MC_values


def read_and_parse_voltages(filename):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(filename, delim_whitespace=True)

    # Get voltage column names by filtering out current (I) columns
    voltage_columns = [col for col in df.columns if "V" in col]

    # Create a matrix with the voltage values
    voltage_matrix = df[voltage_columns].values

    return voltage_matrix

def getMcMeasurement(path:str) -> float:
    
    voltage_mat = read_and_parse_voltages(path)
    voltage_mat = voltage_mat[10:]
    estimated_vec = []
    target_vec = []
    n_MC = [0, 0, 0]
    MC_vec = []
    for n in np.arange(15, 16):
        estimated_vec = []
        target_vec = []
        for k in np.arange(1, 30):
            target = np.roll(voltage_mat[:, 0], k)
            if k == 0:
                target = target[:]
                #print(target.shape)
                data = voltage_mat[:, 1:n]
                #print(data.shape)
            else:
                target = target[:-k]
                #print(target.shape)
                data = voltage_mat[:-k, 1:n]
                #plt.plot(data[:, 2])
            target = target[10:-10]
            data = data[10:-10, :]
            mse, target_test, prediction_test = kDelayAccuracy(target, 1, data)
            target_vec.append(target_test)
            estimated_vec.append(prediction_test)
            # print(k, mse)
        # plt.show()

        MemC = generate_forgetting_curve(estimated_vec, target_vec)
        MC = calculate_memory_capacity(estimated_vec, target_vec)
        #print(MemC)
        #print(MC)
        n_MC.append(MC)
        MC_vec.append(MemC)


    #plt.set_cmap(cmap="jet")
    #plt.show()
    #np.savetxt("C:/Users/David/OneDrive/Desktop/mcvec.txt", MC_vec)
    return MC