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
    
    number_of_periods = 1     # Number of square wave periods
    v_bias_PMU = 0      # Average voltage of square wave [V]
    v_amp_PMU = 1       # Amplitude of the square wave centered in v_bias_PMU [V]
    period = 20e-3      # Time period of the square wave [s]
    rising_time = period/50     # Time required for the rising edge [s]. Modify it only if drops below the minimum limit of 40 nS
    
    

    cycles = 1              # Number of measuring cycles

# =============================================================================
# #                       EXECUTION - OBJECT CREATION
# =============================================================================  

    plt.close('all')
    KT4200 = Keithley4200(address = "TCPIP0::10.60.5.103::1225::SOCKET")
    KT4200.InstInit()

    pulse_width = (period-2*rising_time)/2

    KT4200.sample = sample         # Sample name
    KT4200.cell = cell           # Measured cell
    KT4200.savepath = savepath       # Folder name (automatically created if not existing)
    KT4200.lab = lab       # Lab and instrument of measurement execution
    KT4200.script = os.path.basename(__file__).split('.')[0]

    KT4200.pulses_number = number_of_periods
    KT4200.v_bias_PMU = v_bias_PMU
    KT4200.v_amp_PMU = v_amp_PMU
    KT4200.pulse_width = pulse_width
    KT4200.rising_time = rising_time

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        # try:              
            KT4200.PMUSquareWaveRun() 
        #except:
         #   print("Invalid parameters. Try reducing the number of periods.")           
                
    KT4200.closeSession()
    


if __name__ == "__main__":
    main()