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
  
    sample = "Test"         # Sample name
    cell = 'Test'           # Measured cell
    savepath = "Test"       # Folder name (automatically created if not existing)
    lab = 'INRiM6430'       # Lab and instrument of measurement execution
    
    v_pulse_read  = 0.01       # Reading voltage [V]
    v_pulse_set = 2    
    v_pulse_reset = 1           # Pulse voltage [V]
    pulse_width = [5, 5, 5, 5, 5, 0]       # Vector of time widhts in the form [prepulse read time, set pulse width, interpulse read time, reset pulse width, postpulse reading time, post-train reading] [s]
    t_step_pulse = 0.3          # time spacing between readings [s] 
    i_range_pulse = 1e-5         # SMU current range [A] (0 = autorange)   
    i_cc_pulse = 1e-2           # compliance current [A] 
    pulses_number = 2           # Number of pulses in the train
    
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
                  
    KT6430.v_pulse_read  = v_pulse_read
    KT6430.v_pulse = v_pulse_set
    KT6430.v_pulse_reset = v_pulse_reset
    KT6430.t_read_pre = pulse_width[0]      
    KT6430.t_pulse = pulse_width[1]
    KT6430.t_read_inter = pulse_width[2]
    KT6430.t_pulse_reset = pulse_width[3]
    KT6430.t_read_post = pulse_width[4]
    KT6430.t_read_end = pulse_width[5]
    KT6430.t_step_pulse = t_step_pulse       
    KT6430.i_cc_pulse = i_cc_pulse           
    KT6430.i_range_pulse = i_range_pulse  
    KT6430.perNumPulse = pulses_number       

    KT6430.ptNumRead = 1

# =============================================================================
# #                       EXECUTION - MEASUREMENT
# =============================================================================  
    
    for cNum in range(cycles):
        try:        
            KT6430.pulseTrainSetResetRun(Plot = plotFlag)    
            KT6430.sourceOFF()            
        except:        
            KT6430.exitSave()
            KT6430.sourceOFF()
        
    KT6430.closeSession()
    


if __name__ == "__main__":
    main()