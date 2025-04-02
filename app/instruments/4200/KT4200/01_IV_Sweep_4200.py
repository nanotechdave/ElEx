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
  
    sample = "TestSweep"         # Sample name
    cell = 'Test'           # Measured cell
    savepath = "Test"       # Folder name (automatically created if not existing)
    lab = 'INRiM4200'       # Lab and instrument of measurement execution
      
    v_set = 2             # Maximum positive sweep voltage [V]
    v_reset = 2           # Maximum negative sweep voltage [V]
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
    KT4200 = Keithley4200(address = "TCPIP0::10.60.5.103::1225::SOCKET")
    KT4200.InstInit()

    KT4200.sample = sample
    KT4200.cell = cell
    KT4200.savepath = savepath
    KT4200.lab = lab 
    KT4200.script = os.path.basename(__file__).split('.')[0]
    
    KT4200.v_set = v_set
    KT4200.v_reset = v_reset
    KT4200.step_set = step_set
    KT4200.step_reset = step_reset    
    KT4200.i_range_set = i_range_set         
    KT4200.i_cc_set = i_cc_set 
    KT4200.i_range_reset = i_range_reset         
    KT4200.i_cc_reset = i_cc_reset          
    
    KT4200.ptNumRead = 1

    KT4200.VRead = True
# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        try:        
            KT4200.sweepFullRun(Plot = plotFlag)
            KT4200.sourceOFF()            
        except:        
            KT4200.exitSave()
            KT4200.sourceOFF()
        
    KT4200.closeSession()
    


if __name__ == "__main__":
    main()