# functions_pulsing.py
import os
import pandas as pd

def numSearch(data, target):
    # Example search function implementation
    for index, value in enumerate(data):
        if value == target:
            return index
    return -1

def fileInit(savepath, measurement_name):
    # Initialize the data file
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filepath = os.path.join(savepath, f"{measurement_name}.csv")
    with open(filepath, "w") as file:
        file.write("Voltage,Current\n")  # Header for CSV

def fileUpdate(savepath, measurement_name, voltage, current):
    # Update the file with new data
    filepath = os.path.join(savepath, f"{measurement_name}.csv")
    with open(filepath, "a") as file:
        file.write(f"{voltage},{current}\n")
