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

from arc2custom import dparclib as dparc
from arc2custom import dplib as dp
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
        for arc_ch in channel_vec:
            if self.session.useDbMap:
                # Convert ARC channel to DB pin
                db_label = self.session.arcToDb[arc_ch]
                label_str = f"{db_label}"
            else:
                # Use ARC channel as label
                label_str = f"{arc_ch}"
            header += f"{label_str}_V[V] {label_str}_I[A] "
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
        db_mapping: bool = True, 
    ):
        """
        set db_mapping to True if channels are passed as appear on the daughterboard, 
        set to False to pass internal channels
        """
        if db_mapping:
            arc_mask_to_gnd   = self.session.dbListToArc(mask_to_gnd)
            arc_mask_to_bias  = self.session.dbListToArc(mask_to_bias)
            arc_mask_to_read_v = self.session.dbListToArc(mask_to_read_v)
            arc_mask_to_read_i = self.session.dbListToArc(mask_to_read_i)
            """Updates object mask values based on arguments"""
            self.settings.mask_to_gnd = np.array(arc_mask_to_gnd)
            self.settings.mask_to_bias = np.array(arc_mask_to_bias)
            self.settings.mask_to_read_v = np.array(arc_mask_to_read_v)
            self.settings.mask_to_read_i = np.array(arc_mask_to_read_i)
        else:
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
    
    def measureToStr(self, timestamp, voltage, currentSample, mask, label_as_db=False):
        """
        Builds one line of text: timestamp + pairs of (voltage, current).
        
        :param timestamp: float, the measurement time
        :param voltage: 1D array/list of voltages, in the same order as 'mask'
        :param currentSample: array/list (or dict) indexed by ARC channel
        :param mask: list of ARC channels in the order used for 'voltage'
        :param label_as_db: if True, reorder columns by ascending DB pin
        
        Example output format (if label_as_db=False):
        "123.456 0.01 1e-9 0.02 2e-9 ..."
        """
        if not label_as_db:
            # Keep the ARC ordering (as in your original function)
            data_row = str(timestamp)
            for idx, arc_ch in enumerate(mask):
                data_row = " ".join([data_row, str(voltage[idx]), str(currentSample[arc_ch])])
            return data_row
        else:
            # Sort by DB pin: build a new channel list sorted by ascending DB
            db_sorted_mask = sorted(mask, key=lambda ch: self.session.dict_arcToDb[ch])
            
            # Reorder 'voltage' to match the new sorted channel list
            # (For each ARC channel in db_sorted_mask, find its index in the old mask)
            voltage_db_sorted = [voltage[mask.index(ch)] for ch in db_sorted_mask]
            
            # Build the data row in DB-sorted order
            data_row = str(timestamp)
            for idx, arc_ch in enumerate(db_sorted_mask):
                data_row = " ".join([data_row, str(voltage_db_sorted[idx]), str(currentSample[arc_ch])])
            return data_row



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
