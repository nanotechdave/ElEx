"""
Author: Fabio Michieletti

e-mail: fabio.michieletti@polito.it

This work is licensed under CC BY-NC-SA 4.0 
"""
import matplotlib.pyplot as plt
from Instruments import Keithley6430
import os


def main(args=None): 

# =============================================================================
# #                            PARAMETERS SETTING
# =============================================================================  
  
    sample = "NP_Pad002"         # Sample name
    cell = 'W2E2'           # Measured cell
    savepath = "C:/Users/user/Desktop/Tan/20241015"       # Folder name (automatically created if not existing)
    lab = 'INRiM6430'       # Lab and instrument of measurement execution
    
    v_read = 0.5             # Applied voltage [V]
    t_read = 200              # Application time [s]
    t_step_read = 0.2       # Time step between read points [s] (min 0.2)
    i_range_read = 1e-5      # SMU current range [A] (0 = autorange)
    i_cc_read = 1e-5         # compliance current [A] 

    plotFlag = True         # True for plotting in real time (msampling rate is strongly dumped)
    

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
    
    KT6430.v_read = v_read
    KT6430.i_range_read = i_range_read         # SMU current range for reading [A]
    KT6430.i_cc_read = i_cc_read           # compliance current for reading [A]        
    KT6430.t_read = t_read          # length of constant voltage stimulation [s]
    KT6430.t_step_read = t_step_read             # time spacing between readings [s]        
    
    KT6430.ptNumRead = 1

    cycles = 1              # Number of measuring cycles

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        try:        
            KT6430.constRun(Plot = plotFlag)    
            KT6430.sourceOFF()            
        except:        
            KT6430.exitSave()
            KT6430.sourceOFF()
        
    KT6430.closeSession()
    


if __name__ == "__main__":
    main()