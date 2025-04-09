
import time

import numpy as np
import pyarc2

from arc2custom import dparclib, dplib, fwutils, measurementsettings, sessionmod
from sessionmod import db_to_arc
from .experiments.experiment import Experiment
from .experiments.conductivitymatrix import ConductivityMatrix
from .experiments.ivmeasurement import IVMeasurement
from .experiments.memorycapacity import MemoryCapacity
from .experiments.noisemeasurement import NoiseMeasurement
from .experiments.pulsemeasurement import PulseMeasurement
from .experiments.reservoircomputing import ReservoirComputing
from .experiments.tomography import Tomography
from .experiments.turnon import TurnOn

from arc2custom import memorycapacity


def main(args=None):
    arc = dparclib.initialize_instrument(firmware_release="efm03_20220905.bin")

    # ----------------------- SAVING SETTINGS ----------------------------------
    
    session = sessionmod.Session()

    # ---------------------- DAUGHTERBOARD MAPPING -----------------------------

    session.mapping = dplib.load_channel_map("dendrites_fix32.toml")
    session.useDbMap = True
    
    # ------------------------ ROUTINES INITIALIZATION -------------------------
    mat = ConductivityMatrix(arc, "Conductivity Matrix", session)
    mat.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    mat.setMeasurement(
        sample_time=0.01,
        v_read=0.05,
        n_reps_avg=10,
    )

    pulse = PulseMeasurement(arc, "relaxation", session)
    pulse.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    pulse.setMeasurement(
        sample_time=0.2,
        pre_pulse_time=10,
        pulse_time=10,
        post_pulse_time=300,
        pulse_voltage=[1],
        interpulse_voltage=0.05,
    )

    noise = NoiseMeasurement(arc, "Noise Measurement", session)
    noise.setMaskSettings(
        mask_to_bias=[13],
        mask_to_gnd=[18],
        mask_to_read_v=[13, 14, 15, 16, 17],
        mask_to_read_i=[13, 17],
    )
    noise.setMeasurement(
        duration=360,
        bias_voltage=0.05,
    )

    iv = IVMeasurement(arc, "IV char", session)
    iv.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8,24),
        mask_to_read_i=[8, 17],
    )
    iv.setMeasurement(
        sample_time=0.05,
        start_voltage=0.01,
        end_voltage=10,
        voltage_step=0.01,
        g_stop=200,
        g_interval=0.2,
        g_points=2,
        float_at_end=True,
    )

    constant = IVMeasurement(arc, "constant", session)
    constant.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8,24),
        mask_to_read_i=[8, 17],
    )
    constant.setMeasurement(
        sample_time=0.01,
        start_voltage=0.01,
        end_voltage=4,
        voltage_step=0.005,
        g_stop=1.25,
        g_interval=0.05,
        g_points=10,
        float_at_end=True,
    )
    """ constant.customWave(
        voltage_times_tuple_array=dplib.generate_constant_voltage(
            total_time=100, sampling_time=0.01, voltage=0.5
        )
    ) """
    constant.customWave(
        voltage_times_tuple_array = dplib.generate_step(timeup=2, timedown=60, time_pre_step=10, vUp=3, vDwn=0.05, sampling_time=0.1)
    )

    relaxation = IVMeasurement(arc, "relaxation", session)
    relaxation.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8,24),
        mask_to_read_i=[8, 17],
    )
    relaxation.setMeasurement(
        sample_time=0.01,
        start_voltage=0.01,
        end_voltage=4,
        voltage_step=0.005,
        g_stop=1.25,
        g_interval=0.05,
        g_points=10,
        float_at_end=True,
    )
    relaxation.customWave(
        voltage_times_tuple_array=dplib.generate_step(20, 200,10, 5, 0.05, 0.2)
    )

    sine = IVMeasurement(arc, "sinewave", session)
    sine.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8,24),
        mask_to_read_i=[8, 17],
    )
    sine.setMeasurement(
        sample_time=0.01,
        start_voltage=0.01,
        end_voltage=4,
        voltage_step=0.005,
        g_stop=1.25,
        g_interval=0.05,
        g_points=10,
        float_at_end=True,
    )
    sine.customWave(
        voltage_times_tuple_array=dplib.sineGenerator(
            amplitude=0.3, periods=100, frequency=1, dc_bias=0.5, sample_rate=100
        )
    )

    tri = IVMeasurement(arc, "Triangular", session)
    tri.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8,24),
        mask_to_read_i=[8, 17],
    )
    tri.setMeasurement(
        sample_time=0.01,
        start_voltage=0.01,
        end_voltage=4,
        voltage_step=0.005,
        g_stop=1.25,
        g_interval=0.05,
        g_points=10,
        float_at_end=True,
    )
    tri.customWave(
        voltage_times_tuple_array=dplib.triangGenerator(
            amplitude=0.3, periods=50, frequency=1, dc_bias=0.5, sample_rate=100
        )
    )

    tomography = Tomography(arc, "Tomography", session)
    tomography.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[23],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 23],
    )
    tomography.setMeasurement(
        sample_time=0.1,
        v_read=0.050,
        n_reps_avg=100,
    )
    turnon = TurnOn(arc, "TurnOn", session)
    turnon.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    turnon.setMeasurement(
        sample_time=0.01,
        v_read=1,
        n_reps_avg=1,
    )

    mem = MemoryCapacity(arc, "MemoryCapacity", session)
    mem.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    mem.setMeasurement(
        sample_time=0.1,
        n_samples = 3000,
        v_read=0.05,
        n_reps_avg=1,
    )

    res = ReservoirComputing(arc, "Reservoir", session)
    res.setMaskSettings(
        mask_to_bias=[8, 13, 17, 21],
        mask_to_gnd=[],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 13, 17, 21],
    )
    res.setMeasurement(
        sample_time=3,
        v_read=0.1,
        n_reps_avg=1,
    )

    # -----------------------RUN SEQUENCE---------------------------------------
    
    crit_bias = 0.8
    crit_amp = 0.3 
    MCsampletime = 0.2
    bestMCValue = 0
    bestIdx = 0
    
    
    
    '''if tomography.testConnections():
        mat.run()
        iv.run(plot=True)
        mat.run() '''
        
     
    

    if tomography.testConnections():
        for i in range(10):

            mat.run()
            mcfilename = mem.run(v_bias_vector = dplib.generate_random_array(3000,crit_bias-crit_amp,crit_bias+crit_amp))
            MCvalue = memorycapacity.getMcMeasurement(mcfilename)
            print(f"MC = {MCvalue}")
            sine.customWave(
                voltage_times_tuple_array=dplib.sineGenerator(
                    amplitude=0.3, periods=200, frequency=1, dc_bias=crit_bias, sample_rate=50
                )
            )
            mat.run()
            sine.runFastAVG()
            mat.run()
            
        
        for i in range(10):

            mat.run()
           
            sine.customWave(
                voltage_times_tuple_array=dplib.sineGenerator(
                    amplitude=0.3, periods=200, frequency=1, dc_bias=crit_bias, sample_rate=50
                )
            )
            sine.runFastAVG()
            mat.run()
            
        
    
        

        
    """ for i in range(30):
        mat.run()
        tomography.run()
        constant.customWave(
                voltage_times_tuple_array=dplib.generate_constant_voltage(
                    total_time=60, sampling_time=0.01, voltage=crit_bias
                )
            )
        sine.customWave(
            voltage_times_tuple_array=dplib.sineGenerator(
                amplitude=0.3, periods=10, frequency=0.2, dc_bias=crit_bias, sample_rate=100
            )
        )
        constant.runFastAVG()
        mcfilename = mem.run(v_bias_vector = dplib.generate_random_array(3000,crit_bias-crit_amp,crit_bias+crit_amp))
        sine.runFastAVG()
        mat.run()
        tomography.run()
        print(f"filename is: {mcfilename}")
        MCvalue = memorycapacity.getMcMeasurement(mcfilename)
        print(f"MCValue is:{MCvalue}, best was {bestMCValue}")
        if MCvalue>bestMCValue:
            bestMCValue = MCvalue
            bestIdx = i
        #MCsampletime += 0.1
        mem.setMeasurement(
            sample_time=MCsampletime,
            n_samples = 3000,
            v_read=0.1,
            n_reps_avg=1,
        )
        crit_bias += 0.2
        time.sleep(1800)  """  
   
    """ # ----------------- CRITICALITY PROTOCOL - PAPER REVISION 1 - 13/01/25 ---------------------
    if tomography.testConnections():            

            crit_bias = 1
            crit_amp = 0.5
                
            mat.run()

            iv.setMeasurement(                
                start_voltage=0.01,
                end_voltage=10,                
                voltage_step=0.01,
                sample_time=0.5,
                #voltage_step=0.0009,
                #sample_time=0.015,
                
                g_stop=80,
                #g_stop=0.1,
                g_interval=50.40,
                g_points=5,
                float_at_end=True,
            )  
            iv.run(plot=True)

            

            
            constant.customWave(
                voltage_times_tuple_array=dplib.generate_constant_voltage(
                    total_time=100, sampling_time=0.02, voltage=crit_bias
                )
            )
            constant.runFastAVG()
            
            
            sine.customWave(
                voltage_times_tuple_array=dplib.sineGenerator(
                    amplitude=crit_amp, periods=30, frequency=1, dc_bias=crit_bias, sample_rate=50
                )
            )
            sine.runFastAVG()
            constant.runFastAVG()
            
            mat.run()
        

            dparclib.setAllChannelsToFloat(arc) """