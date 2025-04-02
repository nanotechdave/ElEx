"""
Author: Fabio Michieletti

e-mail: fabio.michieletti@polito.it

This work is licensed under CC BY-NC-SA 4.0 
"""
import matplotlib.pyplot as plt
from Instruments import Keithley6430
import os
import time


def main(args=None): 

# =============================================================================
# #                            PARAMETERS SETTING
# =============================================================================  
  
    sample = "NP_Pad001"         # Sample name
    cell = 'W2E2'           # Measured cell
    savepath = "C:/Users/user/Desktop/Tan/20241014"       # Folder name (automatically created if not existing)
    lab = 'INRiM6430'       # Lab and instrument of measurement execution
    
    v_set = 1             # Maximum positive sweep voltage [V]
    v_reset = 1           # Maximum negative sweep voltage [V]
    step_set = 0.05         # Voltage step between set branch points [V]
    step_reset = 0.05       # Voltage step between reset branch points [V]
    i_range_set = 1e-3      # SMU current range for the whole sweep [A] (0 = autorange)
    i_cc_set = 1e-3         # compliance current for the whole sweep [A] 
    i_range_reset = 1e-3      # SMU current range for the reset branch [A] (0 = autorange)
    i_cc_reset = 1e-3         # compliance current for the reset branch [A] 

    plotFlag = True         # True for plotting in real time (msampling rate is strongly dumped)
    
    cycles = 1              # Number of measuring cycles

# =============================================================================
# #                       EXECUTION - OBJECT CREATION
# =============================================================================  
    
    plt.close('all')
    KT6430 = Keithley6430(address = "GPIB0::3::INSTR")
    KT6430.InstInit()

    KT6430.sample = sample
    KT6430.cell = cell
    KT6430.savepath = savepath
    KT6430.lab = lab 
    KT6430.script = os.path.basename(__file__).split('.')[0]
    
    KT6430.v_set = v_set
    KT6430.v_reset = v_reset
    KT6430.step_set = step_set
    KT6430.step_reset = step_reset    
    KT6430.i_range_set = i_range_set         
    KT6430.i_cc_set = i_cc_set       
    KT6430.i_range_reset = i_range_reset         
    KT6430.i_cc_reset = i_cc_reset       
    
    KT6430.ptNumRead = 1

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    time.sleep(3)
    for cNum in range(cycles):
        try:        
            KT6430.sweepFullVideoRun(Plot = plotFlag)               
            KT6430.sourceOFF()            
        except:        
            KT6430.exitSave()
            KT6430.sourceOFF()
            KT6430.videoRelease()
        
    KT6430.closeSession()
    


if __name__ == "__main__":
    main()