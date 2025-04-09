
import time

import numpy as np
import pyarc2

import dparclib, dplib, fwutils, measurementsettings, sessionmod
from experiments.experiment import (
    ConductivityMatrix,
    Experiment,
    IVMeasurement,
    MemoryCapacity,
    NoiseMeasurement,
    PulseMeasurement,
    ReservoirComputing,
    Tomography,
    TurnOn,
)


def main(args=None):
    arc = dparclib.initialize_instrument(firmware_release="efm03_20220905.bin")

    # ----------------------- SAVING SETTINGS - --------------------------------
    '''session = sessionmod.Session(
        savepath="D:\Desktop\Test",
        lab="INRiMJanis",
        sample="NWN_Pad78M",
        cell="grid_SO",
    )'''
    
    session = sessionmod.Session(
        savepath= "C:/Users/g.milano/Desktop/Davide/ARC",
        lab="INRiMJanis",
        sample="test",
        cell="grid_NO",
    )
    
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
        v_read=0.01,
        n_reps_avg=2,
    )

    pulse = PulseMeasurement(arc, "Long steady pulse", session)
    pulse.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    pulse.setMeasurement(
        sample_time=0.5,
        pre_pulse_time=1,
        pulse_time=200,
        post_pulse_time=1,
        pulse_voltage=[10],
        interpulse_voltage=0.01,
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
        sample_time=0.01,
        start_voltage=0.01,
        end_voltage=4,
        voltage_step=0.005,
        g_stop=0.42,
        g_interval=0.05,
        g_points=10,
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
    constant.customWave(
        voltage_times_tuple_array=dplib.generate_constant_voltage(
            total_time=7200, sampling_time=0.1, voltage=1
        )
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
            amplitude=0.05, periods=1, frequency=0.1, dc_bias=0, sample_rate=100
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
            amplitude=0.05, periods=1, frequency=0.1, dc_bias=0, sample_rate=100
        )
    )

    tomography = Tomography(arc, "Tomography", session)
    tomography.setMaskSettings(
        mask_to_bias=[8],
        mask_to_gnd=[17],
        mask_to_read_v=np.arange(8, 24),
        mask_to_read_i=[8, 17],
    )
    tomography.setMeasurement(
        sample_time=0.1,
        v_read=0.01,
        n_reps_avg=10,
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
        sample_time=0.01,
        v_read=0.1,
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
    
    mem.run(v_bias_vector = dplib.generate_random_array(1000,1,2))
    """  #crit_bias = 0.8
    crit_amp = 0.6
    #crit_bias = 1.9
    if tomography.testConnections(): 
        #time.sleep(7200) 
          

        for j in range(1):
            for i in range(48):
                
                #crit_amp = 0.1*(i+1)
                crit_bias = 0.6+0.1*(i+1)
                
                mat.run()

                iv.setMeasurement(                
                    start_voltage=0.01,
                    end_voltage=3,                
                    voltage_step=0.01,
                    sample_time=0.5,
                    #voltage_step=0.0009,
                    #sample_time=0.015,
                    
                    g_stop=51,
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
                
                '''tri.customWave(
                    voltage_times_tuple_array=dplib.triangGenerator(
                        amplitude=crit_amp, periods=30, frequency=0.1, dc_bias=crit_bias, sample_rate=20
                        #amplitude=0.3, periods=50, frequency=0.5, dc_bias=0.5, sample_rate=10
                    )
                )
                tri.runFastAVG()''' 

                sine.customWave(
                    voltage_times_tuple_array=dplib.sineGenerator(
                        amplitude=crit_amp, periods=30, frequency=1, dc_bias=crit_bias, sample_rate=50
                    )
                )
                sine.runFastAVG()

                mem.run(v_bias_vector = dplib.generate_random_array(1000,crit_bias-crit_amp,crit_bias+crit_amp))
                mat.run()
                tomography.run()

                dparclib.setAllChannelsToFloat(arc)
                time.sleep(3600) 
                
                
                
                # criticality check
                
                '''sine.customWave(
                    voltage_times_tuple_array=dplib.sineGenerator(
                        amplitude=0.1*(i+1), periods=100, frequency=1, dc_bias=1, sample_rate=10
                    )
                )'''

                

                
                
                '''
                mem.run(v_bias_vector = dplib.generate_random_array(1000,crit_bias-(crit_amp*(i+1)),crit_bias+(crit_amp*(i+1))))
                mat.run()
                tomography.run()
                dparclib.setAllChannelsToFloat(arc)
                #time.sleep(5400)
                '''
        #dparclib.setAllChannelsToFloat(arc)
   
    """