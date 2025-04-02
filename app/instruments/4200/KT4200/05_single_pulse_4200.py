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
  
    sample = "TestSinglePulse"         # Sample name
    cell = 'Test'           # Measured cell
    savepath = "C:/Users/franc/Desktop/Fabio"       # Folder name (automatically created if not existing)
    lab = 'POLITO4200'       # Lab and instrument of measurement execution
    
    v_pulse_read  = 0.01       # Reading voltage [V]
    v_pulse = 2.5               # Pulse voltage [V]
    pulse_width = [3, 1, 300]       # Vector of time widhts in the form [prepulse read time, pulse width, postpulse reading time] [s]
    t_step_pulse = 0.2          # time spacing between readings [s] 
    i_range_pulse = 1e-3         # SMU current range [A] (0 = autorange)   
    i_cc_pulse = 1e-3           # compliance current [A] 
    
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
                  
    KT4200.v_pulse_read  = v_pulse_read
    KT4200.v_pulse = v_pulse
    KT4200.t_read_pre = pulse_width[0]      
    KT4200.t_pulse = pulse_width[1]
    KT4200.t_read_post = pulse_width[2]
    KT4200.t_step_pulse = t_step_pulse       
    KT4200.i_cc_pulse = i_cc_pulse           
    KT4200.i_range_pulse = i_range_pulse         

    KT4200.ptNumRead = 1

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        try:        
            KT4200.pulseRun(Plot = plotFlag)    
            KT4200.sourceOFF()            
        except:        
            KT4200.exitSave()
            KT4200.sourceOFF()
        
    KT4200.closeSession()
    


if __name__ == "__main__":
    main()