# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:17:44 2023

@author: Davide Pilati
"""
from datetime import date
from arc2custom import dplib as dp
import numpy as np


class MeasurementSettings:
    """
    Defines the measurement settings
    """

    def __init__(self):
        # SET DEFAULT GENERIC PARAMETERS FOR LIST OF TUPLES BASED EXPERIMENT

        self.mask_to_gnd = np.array([16])
        self.mask_to_bias = np.array([13])
        self.mask_to_read_v = np.array([13, 16])
        self.mask_to_read_i = np.array([13, 16])
        self.mask = np.array(
            dp.concat_vectors(
                [
                    self.mask_to_gnd,
                    self.mask_to_bias,
                    self.mask_to_read_i,
                    self.mask_to_read_v,
                ]
            )
        )

        self.v_tuple = dp.pulseGenerator()  # [(v, time)]
        self.vBiasVec, self.vTimes = dp.tuplesToVec(self.v_tuple)
        self.meas_iterations = 1
        self.tot_iterations = len(self.vBiasVec)
        self.readafter = 0

    def set_attr(self, name, value):
        setattr(self, name, value)

    def get_setting(self, name):
        getattr(self, name)

    def reset_settings(self):
        MeasurementSettings.__init__()
