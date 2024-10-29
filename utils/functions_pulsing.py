# functions_pulsing.py
import os
import glob
import pandas as pd

def numSearch(savepath):
    """Search for the latest file in savepath and return the next file number."""
    path_search = f"{savepath}/*txt"
    try:
        file_list = glob.glob(path_search)
        num_list = [int(file.split("\\")[-1].split('_')[0]) for file in file_list]
        start_num = max(num_list) + 1
    except ValueError:
        start_num = 0
    return start_num

def dataInit(genNum):
    """Initialize a data structure for pulse data collection."""
    box = []
    for i in range(genNum):
        box.append([])
    return box

def fileInit(savepath, measurement_name):
    """Initialize a CSV file to store voltage and current readings."""
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filepath = os.path.join(savepath, f"{measurement_name}.csv")
    with open(filepath, "w") as file:
        file.write("Voltage,Current\n")

def fileUpdate(savepath, measurement_name, voltage, current):
    """Update the CSV file with new voltage and current data."""
    filepath = os.path.join(savepath, f"{measurement_name}.csv")
    with open(filepath, "a") as file:
        file.write(f"{voltage},{current}\n")
