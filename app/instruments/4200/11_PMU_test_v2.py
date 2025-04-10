"""
Author: Fabio Michieletti

e-mail: fabio.michieletti@polito.it

This work is licensed under CC BY-NC-SA 4.0 
"""
import matplotlib.pyplot as plt
from Instruments import Keithley4200
import os


def main(args=None): 

# =============================================================================
# #                            PARAMETERS SETTING
# =============================================================================  
  
    sample = "TestPMU"         # Sample name
    cell = 'r10k'           # Measured cell
    savepath = "TestPMU"       # Folder name (automatically created if not existing)
    lab = 'INRiM4200'       # Lab and instrument of measurement execution
    
    pulses_number = 1
    v_bias_PMU = 0
    v_amp_PMU = 1
    period = 10e-6
    rising_time = period/50
    pulse_width = (period-2*rising_time)/2
    

    cycles = 1              # Number of measuring cycles

# =============================================================================
# #                       EXECUTION - OBJECT CREATION
# =============================================================================  

    plt.close('all')
    KT4200 = Keithley4200(address = "TCPIP0::10.60.5.103::1225::SOCKET")
    KT4200.InstInit()

    KT4200.sample = sample         # Sample name
    KT4200.cell = cell           # Measured cell
    KT4200.savepath = savepath       # Folder name (automatically created if not existing)
    KT4200.lab = lab       # Lab and instrument of measurement execution
    KT4200.script = os.path.basename(__file__).split('.')[0]

    KT4200.pulses_number = pulses_number
    KT4200.v_bias_PMU = v_bias_PMU
    KT4200.v_amp_PMU = v_amp_PMU
    KT4200.pulse_width = pulse_width
    KT4200.rising_time = rising_time

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):              
        KT4200.PMUSquareWaveRun()    
                
    KT4200.closeSession()
    


if __name__ == "__main__":
    main()