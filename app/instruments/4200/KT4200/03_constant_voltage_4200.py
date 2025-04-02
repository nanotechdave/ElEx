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
  
    sample = "Test"         # Sample name
    cell = 'Test'           # Measured cell
    savepath = "Test"       # Folder name (automatically created if not existing)
    lab = 'INRiM4200'       # Lab and instrument of measurement execution
 
    v_read = 0.5             # Applied voltage [V]
    t_read = 1800              # Application time [s]
    t_step_read = 0.001       # Time step between read points [s] (min 0.2)
    i_range_read = 1e-6      # SMU current range [A] (0 = autorange)
    i_cc_read = 1e-6         # compliance current [A] 

    plotFlag = True         # True for plotting in real time (msampling rate is strongly dumped)
    

# =============================================================================
# #                       EXECUTION - OBJECT CREATION
# =============================================================================  
    
    plt.close('all')
    KT4200 = Keithley4200(address = "TCPIP0::10.60.5.103::1225::SOCKET")
    KT4200.InstInit()

    KT4200.sample = sample
    KT4200.cell = cell
    KT4200.savepath = savepath
    KT4200.lab = lab 
    KT4200.script = os.path.basename(__file__).split('.')[0]
    
    KT4200.v_read = v_read
    KT4200.i_range_read = i_range_read         # SMU current range for reading [A]
    KT4200.i_cc_read = i_cc_read           # compliance current for reading [A]        
    KT4200.t_read = t_read          # length of constant voltage stimulation [s]
    KT4200.t_step_read = t_step_read             # time spacing between readings [s]        
    
    KT4200.ptNumRead = 1

    KT4200.VRead = True

    cycles = 1              # Number of measuring cycles

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        try:        
            KT4200.constRun(Plot = plotFlag)    
            KT4200.sourceOFF()            
        except:        
            KT4200.exitSave()
            KT4200.sourceOFF()
        
    KT4200.closeSession()
    


if __name__ == "__main__":
    main()