"""
Author: Fabio Michieletti

e-mail: fabio.michieletti@polito.it

This work is licensed under CC BY-NC-SA 4.0 
"""


import pyvisa as visa
import pandas as pd
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
from pymeasure.instruments.lakeshore import LakeShore331
import os
import cv2
from moviepy import VideoFileClip
import mss
from typing import Tuple, Dict, Any, List, Optional, Union


class Keithley6430:
    
    G0 = 7.74809173e-5
    
    def __init__(self, address = "GPIB0::3::INSTR"):
        self.address = address
        
        self.sample = "NWN_Pad68C"
        self.cell = 'N3S3'
        self.savepath = "Test"
        self.lab = 'INRiMJanis' 
        self.script = '11_Vsin_and_verify'
        
        self.numSearch()
        self.date = datetime.today().strftime('%Y_%m_%d')
        self.temperature = 303
        
        self.v_set = 1                 # maximum positive voltage [V]
        self.i_range_set = 1e-3          # SMU current range in positive voltage branch [A]
        self.i_cc_set = 500e-6           # compliance current in positive voltage branch [A]
        self.step_set = 5e-3            # voltage sweep step in positive voltage branch [V]
        
        self.v_reset = -5              # minimum negative voltage [V]
        self.i_range_reset = 1e-1        # SMU current range in negative voltage branch [A]
        self.i_cc_reset = 1e-1          # compliance current in negative voltage branch [A]
        self.step_reset = 1e-2           # voltage sweep step in negative voltage branch [V]        
        
        self.v_min_perc = 0              # minimum negative voltage [V]
        self.v_max_perc = 1              # minimum negative voltage [V]
        self.i_range_perc = 1e-1        # SMU current range in negative voltage branch [A]
        self.i_cc_perc = 1e-1          # compliance current in negative voltage branch [A]
        self.step_perc = 1e-2           # voltage sweep step in negative voltage branch [V]        
                
        
        self.v_read = 0.5    #0.19          # reading voltage after plateau detection [V]
        self.i_range_read = 0         # SMU current range for reading [A]
        self.i_cc_read = 1e-1           # compliance current for reading [A]        
        self.AvLen = 10                   # number of consecutive equal conductances for reading triggering 
        self.G0Target = [0.12, 0.2, 0.3]            # list of G0 multiples to be searched
        self.edges = [0.01, 0.1]          # inferior and superior interval aroung G0Target values in which the last AvLen conductances must lie 
        self.t_read = 3600*2           # length of constant voltage stimulation [s]
        self.t_step_read = 0.1             # time spacing between readings [s]
        
        self.v_amp = -0.05            # sin voltage amplitude [V]
        self.v_bias = self.v_read           # sin voltage baseline [V]
        self.sin_f = 0.1             # sin frequency [Hz]
        self.i_range_sin = 0         # SMU current range for reading [A]
        self.i_cc_sin = 1e-1           # compliance current for reading [A]        
        self.perNumSin = 100            # length of sin stimulation [number of periods]
        self.t_step_sin = 0.1  # time spacing between readings [s]
        
        self.v_pulse = 1
        self.t_pulse = 10
        self.v_pulse_read = 0.01
        self.i_range_pulse = 0         # SMU current range for reading [A]
        self.i_cc_pulse = 1e-1           # compliance current for reading [A]
        self.t_read_pre = 10
        self.t_read_post = 10
        self.v_pulse_up = 0
        self.perNumPulse = 5
        self.t_step_pulse = 0.1
        self.t_read_inter = 5
        self.t_pulse_reset = 10
        self.v_pulse_reset = 1
        self.t_read_end = 0
        
        self.i_range_custom = 0         # SMU current range for reading [A]
        self.i_cc_custom = 1e-1           # compliance current for reading [A]
        
        self.currNPLC = 1
        self.voltNPLC = 1
        self.voltSourRange = 0
        self.voltSensRange = 0
        
        self.region = (477,132,1165,875)
        
        self.cycles = 1                 # number of I-V sweeps cycles 

        self.saveDelay = 10*60*60
        
        self.plotLabels = {'Title':['Voltage','Current','I-V','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','Temperature[K]']}
        
        self.temperatureMeas = False
        self.VRead = True
        self.Plot = False
        self.G0Found = False

        
        
# =============================================================================
#                       INSTRUMENT SINGLE FUNCTIONS
# =============================================================================

    def InstInit(self):
        resourceManager = visa.ResourceManager()
        resourceManager.list_resources()
        self.session = resourceManager.open_resource(self.address)
        """ self.session.timeout = None """
        print(f"Keithley6430 session opened at address {self.address}") 
        return
    
        
    def KT6430Init(self,i_range,i_cc,vIn):      
        """ if not self.VRead: """
        self.reset()
        self.session.write(":SYST:TIME:RES")
        self.session.write(":ROUT:TERM REAR")
        self.session.write(':DISP:ENAB OFF')
        self.session.write(":SOUR:FUNC VOLT")
        self.session.write(":SOUR:VOLT:MODE FIX")
        if self.voltSourRange == 0:
            self.session.write(":SOUR:VOLT:RANG:AUTO ON")
        else:
            self.session.write(":SOUR:VOLT:RANG:AUTO OFF")
            self.session.write(f":SOUR:VOLT:RANG {self.voltSourRange}")        
        self.session.write(":SENS:FUNC \"CURR\" ")
        if self.VRead:
            self.session.write(":SENS:FUNC \"VOLT\" ")
        self.session.write(":TRACe:TSTamp:FORMat ABS")
        self.session.write(":SENS:AVER:AUTO OFF")
        self.session.write(f":SENS:CURR:NPLC {self.currNPLC}")
        self.session.write(f":SENS:VOLT:NPLC {self.voltNPLC}")
        self.session.write(f":SOUR:VOLT:LEV {vIn}")
        self.session.write(f":SENS:CURR:PROT {i_cc}")
        if i_range == 0:
            self.session.write(":SENS:CURR:RANG:AUTO ON")
            print("Autorange ON")
        else:
            self.session.write(":SENS:CURR:RANG:AUTO OFF")
            self.session.write(f":SENS:CURR:RANG {i_range}")
            print(f"Autorange OFF \n i_range = {i_range}")
        if self.voltSensRange == 0 and self.VRead:
            self.session.write(":SENS:VOLT:RANG:AUTO ON")
        elif self.VRead:
            self.session.write(":SENS:VOLT:RANG:AUTO OFF")
            self.session.write(f":SENS:VOLT:RANG {self.voltSensRange}")
        return
    
    def rangeSet(self,i_range,i_cc):
        self.session.write(f":SENS:CURR:PROT {i_cc}")
        if i_range == 0:
            self.session.write(":SENS:CURR:RANG:AUTO ON")
            print("Autorange ON")
        else:
            self.session.write(":SENS:CURR:RANG:AUTO OFF")
            self.session.write(f":SENS:CURR:RANG {i_range}")
            print(f"Autorange OFF \n i_range = {i_range}")
        return
    
    def KT6430Start(self): 
        self.session.write(":OUTP ON")
        _, _, _, t0,_ = self.session.query(":READ?").split(",")
        self.t0 = float(t0)    
        return
    
        
    def KT6430Meas(self,v_point,const = False):        
        if not const:
            self.session.write(f":SOUR:VOLT:LEV {v_point}")
        v, i, _, t,_ = self.session.query(":READ?").split(",")
        v = float(v)
        i = float(i)
        r = v/i
        t = float(t)-self.t0
        if r == 0:
            g = float("nan")
        else:
            g = i/v
        gnorm = g/self.G0
        self.data.loc[len(self.data)]=([t, v_point, v, i, r, 0, self.temperature, gnorm])
        return
    
    
    
        
    def sourceOFF(self):
        self.session.write(":OUTP OFF")  
        return
        
    
    def reset(self):        
        self.session.write("*RST")
        return
    
        
    def sendCommand(self,command):
        self.session.write(f"{command}") 
        return
    
    
    def closeSession(self):        
        self.session.close()  
        return
    
        
# =============================================================================
#                       TEMPERATURE CONTROL FUNCTIONS
# =============================================================================
         
    def heatStart(self,wait = True,address = "GPIB0::11::INSTR"):
        self.LS331 = LakeShore331(address)
        
        self.LS331.output_1.setpoint = self.temperature
        self.LS331.output_1.heater_range = "low"        
        self.temperatureMeas = True
        if wait:
            self.LS331.input_A.wait_for_temperature()
        return
   
    
    def heatStop(self):
        self.LS331.output_1.heater_range = 'off'
        self.temperatureMeas = False
        return
        
# =============================================================================
#                         DATA HANDLING FUNCTIONS
# =============================================================================
    def checkPath(self):
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)
        return
            
        
    def numSearch(self):
        pathSearch = f"{self.savepath}/*txt"
        try:
            filelst = glob.glob(pathSearch)
            numlst = [int(filelst[i].split("\\")[-1].split('_')[0]) for i in range(filelst.__len__())]
            self.startNum = max(numlst)+1
        except:
            self.startNum = 1
        return
            
            
    def plotGen(self):
        plt.ion()    
        self.fig,((self.ax1,self.ax2),(self.ax3,self.ax4)) = plt.subplots(2,2,sharex = False, sharey = False, figsize=(13,8))
        self.plot1, = self.ax1.plot(0,0)
        self.ax1.set_ylabel(self.plotLabels['ylabel'][0])
        self.ax1.set_title(self.plotLabels['Title'][0])
        self.ax1.set_xlabel(self.plotLabels['xlabel'][0])    
        self.plot2, = self.ax2.plot(0,0)
        self.ax2.set_ylabel(self.plotLabels['ylabel'][1])
        self.ax2.set_title(self.plotLabels['Title'][1])
        self.ax2.set_xlabel(self.plotLabels['xlabel'][1])        
        self.plot3, = self.ax3.plot(0,0)
        self.ax3.set_ylabel(self.plotLabels['ylabel'][2])
        self.ax3.set_title(self.plotLabels['Title'][2])
        self.ax3.set_xlabel(self.plotLabels['xlabel'][2])
        self.plot4, = self.ax4.plot(0,0)
        self.ax4.set_ylabel(self.plotLabels['ylabel'][3])
        self.ax4.set_title(self.plotLabels['Title'][3])
        self.ax4.set_xlabel(self.plotLabels['xlabel'][3])
        plt.show()     
        return
    
        
    def PlotUpdate(self):
        self.plot1.set_xdata(self.data[self.plotLabels['xlabel'][0]])
        self.plot1.set_ydata(self.data[self.plotLabels['ylabel'][0]])      
        self.ax1.relim()
        self.ax1.autoscale_view()    
        self.plot2.set_xdata(self.data[self.plotLabels['xlabel'][1]])
        self.plot2.set_ydata(self.data[self.plotLabels['ylabel'][1]])           
        self.ax2.relim()
        self.ax2.autoscale_view()    
        self.plot3.set_xdata(self.data[self.plotLabels['xlabel'][2]])
        self.plot3.set_ydata(self.data[self.plotLabels['ylabel'][2]])              
        self.ax3.relim()
        self.ax3.autoscale_view() 
        self.plot4.set_xdata(self.data[self.plotLabels['xlabel'][3]])
        self.plot4.set_ydata(self.data[self.plotLabels['ylabel'][3]])                
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  
        
        # self.plot1.set_xdata(self.data['Time[s]'])
        # self.plot1.set_ydata(self.data['Voltage[V]'])      
        # self.ax1.relim()
        # self.ax1.autoscale_view()    
        # self.plot2.set_xdata(self.data['Time[s]'])
        # self.plot2.set_ydata(self.data['Current[A]'])           
        # self.ax2.relim()
        # self.ax2.autoscale_view()    
        # self.plot3.set_xdata(self.data['Voltage[V]'])
        # self.plot3.set_ydata(self.data['Current[A]'])              
        # self.ax3.relim()
        # self.ax3.autoscale_view() 
        # self.plot4.set_xdata(self.data['Time[s]'])
        # self.plot4.set_ydata(self.data['Temperature[K]'])                
        # self.ax4.relim()
        # self.ax4.autoscale_view()
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()  
        return
    
    
    def PlotSave(self):        
        self.fig.savefig(f"{self.fileName}.png", dpi=300) 
        self.Plot = False
        return
    
    
    def PlotClose(self):
        plt.close('all')
        return
    
    
    def dataInit(self):
        self.data = pd.DataFrame(columns=['Time[s]','Voltage_prog[V]','Voltage_read[V]','Current[A]','Resistance[ohm]','Temperature[K]','TargetTemperature[K]','GNorm[G0]'])
        return
    
    
    def printData(self,header = False):  
        if header:
            head = (" ").join(list(self.data.columns.values))
            print(f"{head}")
        lastRow = (" ").join([str(x) for x in self.data.loc[len(self.data)-1].values.tolist()])
        print(f"{lastRow}\n")  
        return
    
    def printG0(self):  
        lastRow = (" ").join([str(x) for x in self.data.loc[len(self.data)-1].values.tolist()])
        print(f"G = {self.data['GNorm[G0]'].loc[len(self.data)-1]} G0\n")  
        return
    
    
    def saveData(self):        
        self.data.to_csv(f"{self.fileName}.txt",sep = ' ',index = False)  
        return
    
        
    def exitSave(self):        
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        self.PlotUpdate()               
        self.PlotSave()
        self.saveData()       
        return
        
    
        
   
        
    def G0Check(self):
        G0TargetLow = [self.G0Target[i]-self.edges[0] for i in range(self.G0Target.__len__())]
        G0TargetHigh = [self.G0Target[i]+self.edges[1] for i in range(self.G0Target.__len__())]
        for tarNum in range(self.G0Target.__len__()):            
            if all(el > G0TargetLow[tarNum] for el in self.data['GNorm[G0]'].loc[-self.AvLen:]) and all(el < G0TargetHigh[tarNum] for el in self.data['GNorm[G0]'].loc[-self.AvLen:]):
                self.G0found = True
                break
        return
            
# =============================================================================
#                           WAVEFORM GENERATORS
# =============================================================================

    def sweepGenSet(self):
        v_sweep1 = np.linspace(0, self.v_set, int((self.v_set/self.step_set)+1))
        v_sweep2 = v_sweep1[-2:0:-1]    
        self.vInSet = np.concatenate((v_sweep1,v_sweep2))
        self.ptNumSet = self.vInSet.__len__()            
        return
    

    def sweepGenReset(self):    
        v_sweep3 = np.linspace(0, -self.v_reset, int(abs(self.v_reset/self.step_reset)+1))
        v_sweep4 = v_sweep3[-2:0:-1]
        self.vInReset = np.concatenate((v_sweep3,v_sweep4,[0]))
        self.ptNumReset = self.vInReset.__len__()   
        return
    
    
    def sweepGenFull(self): 
        self.sweepGenSet()
        self.sweepGenReset()           
        self.vInFull = np.concatenate((self.vInSet,self.vInReset))
        self.ptNumFull = self.vInFull.__len__()
        return


    def sweepGenPerc(self):
        self.vInPerc = np.linspace(self.v_min_perc, self.v_max_perc, int(((self.v_max_perc-self.v_min_perc)/self.step_perc)+1))
        self.ptNumPerc = self.vInPerc.__len__()        
        return
    
    def sweepGenConst(self):    #EDIT
        self.vInPerc = np.linspace(self.v_min_perc, self.v_max_perc, int(((self.v_max_perc-self.v_min_perc)/self.step_perc)+1))
        self.ptNumPerc = self.vInPerc.__len__() 
        self.ptNumRead = int((self.t_read/self.t_step_read)+1)
        self.vInRead = np.full((self.ptNumRead,), self.v_read, dtype=float)    
        self.vInPerc = np.concatenate([self.vInPerc,self.vInRead])   
        self.ptNumPerc = self.ptNumPerc + self.ptNumRead
        return


    # def sweepGenCurr(v_pos, v_neg, step):
    #     v_sweep1 = np.linspace(v_neg, v_pos, int((v_pos/step)+1))
    #     v_sweep2 = v_sweep1[-2:0:-1]
    #     v_sweep3 = np.linspace(0, v_neg, int(abs(v_neg/step)+1))
    #     v_sweep4 = v_sweep3[-2:0:-1]
    #     # v_sweep_np = np.concatenate((v_sweep1,v_sweep2))
    #     v_sweep_np = np.concatenate((v_sweep1,v_sweep2,v_sweep3,v_sweep4))
    #     ptNum = v_sweep1.__len__()
    #     # v_sweep = ', '.join([str(i) for i in v_sweep_np])
    #     return v_sweep1, ptNum
    
    
    def readTableGen(self):
        self.ptNumRead = int((self.t_read/self.t_step_read)+1)
        self.vInRead = np.full((self.ptNumRead,), self.v_read, dtype=float)    
        self.tInRead = np.linspace(0, self.t_read, self.ptNumRead)        
        return
    
    
    def sinGen(self):
        t_end = 1/self.sin_f*self.perNumSin
        self.ptNumSin = int((t_end/self.t_step_sin)+1) 
        self.tInSin = np.linspace(0, t_end, self.ptNumSin)      
        self.vInSin = self.v_amp * np.sin(2 * np.pi * self.sin_f * self.tInSin) + self.v_bias    
        return
    
    
    def pulseTrainGen(self):
        ptNumPulse = int((self.t_pulse/self.t_step_pulse))
        ptNumReadPre = int((self.t_read_pre/self.t_step_pulse))
        ptNumReadPost = int((self.t_read_post/self.t_step_pulse))
        pulseSeg = np.full(ptNumPulse,self.v_pulse)
        readPreSeg = np.full(ptNumReadPre,self.v_pulse_read)
        readPostSeg = np.full(ptNumReadPost,self.v_pulse_read)
        self.vInPulse = np.concatenate([readPreSeg,pulseSeg,readPostSeg])

        nextSegment = np.concatenate([pulseSeg*(self.t_pulse>0),readPostSeg*(self.t_read_post>0)])
        for i in range(self.perNumPulse-1):
            self.vInPulse = np.concatenate([self.vInPulse,nextSegment])

        if self.t_read_end > 0:
            ptNumReadEnd = int(((self.t_read_end)/self.t_step_pulse))
            readEndSeg = np.full(ptNumReadEnd,self.v_pulse_read)
            self.vInPulse = np.concatenate([self.vInPulse,readEndSeg])

        self.ptNumPulse = self.vInPulse.__len__()
        self.tInPulse = np.linspace(0, self.t_step_pulse*(self.ptNumPulse), self.ptNumPulse)  
        return
    
    def pulseTrainSetResetGen(self):
        ptNumReadPre = int((self.t_read_pre/self.t_step_pulse))
        ptNumPulse = int((self.t_pulse/self.t_step_pulse))
        ptNumReadInter = int((self.t_read_inter/self.t_step_pulse))
        ptNumPulseReset = int((self.t_pulse_reset/self.t_step_pulse))
        ptNumReadPost = int((self.t_read_post/self.t_step_pulse))        

        readPreSeg = np.full(ptNumReadPre,self.v_pulse_read)
        pulseSeg = np.full(ptNumPulse,self.v_pulse)
        readInterSeg = np.full(ptNumReadInter,self.v_pulse_read)
        pulseResetSeg = np.full(ptNumPulseReset,-self.v_pulse_reset)
        readPostSeg = np.full(ptNumReadPost,self.v_pulse_read)
        

        self.vInPulse = np.concatenate([readPreSeg*(self.t_read_pre>0),pulseSeg*(self.t_pulse>0),readInterSeg*(self.t_read_inter>0),pulseResetSeg*(self.t_pulse_reset>0),readPostSeg*(self.t_read_post>0)])
        
        nextSegment = np.concatenate([pulseSeg*(self.t_pulse>0),readInterSeg*(self.t_read_inter>0),pulseResetSeg*(self.t_pulse_reset>0),readPostSeg*(self.t_read_post>0)])

        for i in range(self.perNumPulse-1):
            self.vInPulse = np.concatenate([self.vInPulse,nextSegment])
        
        if self.t_read_end > 0:
            ptNumReadEnd = int(((self.t_read_end)/self.t_step_pulse))
            readEndSeg = np.full(ptNumReadEnd,self.v_pulse_read)
            self.vInPulse = np.concatenate([self.vInPulse,readEndSeg])
            
        self.ptNumPulse = self.vInPulse.__len__()
        self.tInPulse = np.linspace(0, self.t_step_pulse*(self.ptNumPulse), self.ptNumPulse)  
        return
    
    
# =============================================================================
# #                             EXPERIMENTS
# =============================================================================
        
    def rampRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenPerc()     
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT6430Start()    
        for rNum in range(self.ptNumPerc):                                            
            self.KT6430Meas(self.vInPerc[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()         
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        self.PlotSave()
        return
        
        
    def rampCheckRun(self,Plot = True,off_at_end = True):  
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenPerc()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT6430Start()
        for rNum in range(self.ptNumPerc):                                            
            self.KT6430Meas(self.vInPerc[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A 
            self.printG0()
            self.G0Check()
            if self.G0Found:
                break
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPerc-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
        if off_at_end:
            self.sourceOFF()
        self.PlotUpdate()
        self.saveData()
        self.PlotSave() 
        return
    
    def rampConstRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenConst()     
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT6430Start()    
        for rNum in range(self.ptNumPerc):                                            
            self.KT6430Meas(self.vInPerc[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()     
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        self.PlotSave()
        return
        
        
    """ def sweepFullRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenFull()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT6430Start()
        for rNum in range(self.ptNumFull):                                            
            self.KT6430Meas(self.vInFull[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumFull-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                self.PlotSave()
        self.saveData()
        self.PlotSave()
        return """
    
    def sweepFullRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenSet()  
        self.sweepGenReset()
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT6430Start()
        for rNum in range(self.ptNumSet):                                            
            self.KT6430Meas(self.vInSet[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0
        self.rangeSet(self.i_range_reset,self.i_cc_reset)
        for rNum in range(self.ptNumReset):  
            self.KT6430Meas(self.vInReset[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0    
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        if off_at_end:
            self.sourceOFF()
        self.PlotUpdate()
        self.PlotSave()
        self.saveData()        
        return
    
    
    def constRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.readTableGen()  
        tDiff = np.diff(self.tInRead)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_const_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_read,self.i_cc_read,self.vInRead[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumRead):                                            
            self.KT6430Meas(self.vInRead[rNum],const = True)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1 or (self.data['Time[s]'].loc[len(self.data)-1]-self.t0) >= self.t_read:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
                break
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        return
    
        
    def sinRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sinGen()  
        tDiff = np.diff(self.tInSin)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sin_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_sin,self.i_cc_sin,self.vInSin[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumSin):                                            
            self.KT6430Meas(self.vInSin[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumSin-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        return
    
        
    def pulseTrainRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT6430Meas(self.vInPulse[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()            
        self.saveData()
        return
    
    def pulseTrainSetResetRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainSetResetGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT6430Meas(self.vInPulse[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        return
    
    def pulseRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.perNumPulse = 1
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT6430Meas(self.vInPulse[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        return
    
        
    def customRun(self,tInCustom,vInCustom,Plot = True,off_at_end = True):
        self.tInCustom = tInCustom
        self.vInCustom = vInCustom
        self.ptNumCustom = tInCustom.__len__()
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0        
        tDiff = np.diff(self.tInCustom)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_custom_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_custom,self.i_cc_custom,self.vInCustom[0])
        self.KT6430Start()
        tPre = time.time() 
        for rNum in range(self.ptNumCustom):                                            
            self.KT6430Meas(self.vInCustom[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumCustom-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.saveData()
        return
    
    
    
    
    def pulseTrainVideoRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):    
            self.KT6430Meas(self.vInPulse[rNum])
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
            time.sleep(5)
        self.videoRelease()
        self.saveData()
        return
    
    def pulseTrainSetResetVideoRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainSetResetGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT6430Meas(self.vInPulse[rNum])
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.videoRelease()
        self.saveData()
        return
    
    def pulseVideoRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.perNumPulse = 1
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT6430Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT6430Start()
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT6430Meas(self.vInPulse[rNum])
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.videoRelease()
        self.saveData()
        return
    
    def constVideoRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.readTableGen()  
        tDiff = np.diff(self.tInRead)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_const_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_read,self.i_cc_read,self.vInRead[0])
        self.KT6430Start()
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumRead):                                            
            self.KT6430Meas(self.vInRead[rNum],const = True)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1 or (self.data['Time[s]'].loc[len(self.data)-1]-self.t0) >= self.t_read:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
                break
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        if off_at_end:
            self.sourceOFF()
        self.videoRelease()
        self.saveData()
        return
    
    def sweepFullVideoRun(self,Plot = True,off_at_end = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenFull()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT6430Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT6430Start()
        self.videoInit()
        for rNum in range(self.ptNumSet):                                            
            self.KT6430Meas(self.vInSet[rNum])
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0
        self.rangeSet(self.i_range_reset,self.i_cc_reset)
        for rNum in range(self.ptNumReset):  
            self.KT6430Meas(self.vInReset[rNum])
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0    
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        if off_at_end:
            self.sourceOFF()
        self.PlotUpdate()
        self.PlotSave()
        self.videoRelease()
        self.saveData()        
        return
    
    def videoInit(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        self.fps = 20.0 
        self.outputVideo = cv2.VideoWriter(f"{self.fileName}.mp4", fourcc, self.fps, (self.region[2], self.region[3]))  
        return

    """ def frameCapture(self):
        img = pyautogui.screenshot(region=self.region)    
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
        self.outputVideo.write(frame)
        return  """
    
    def frameCapture(self):
        with mss.mss() as sct:
            # Capture the screen based on the specified region
            
            mon = sct.monitors[1]  # monitor[2] refers to the second monitor

        # Adjust the region for the second monitor based on your self.region values
            monitor = {
                "top": mon["top"] + self.region[1],
                "left": mon["left"] + self.region[0],
                "width": self.region[2],
                "height": self.region[3],
                "mon": 1
            }
            # Capture the screenshot
            screenshot = sct.grab(monitor)
            
            # Convert the raw data from the screenshot to a NumPy array
            frame = np.array(screenshot)

            # Convert from BGRA to BGR (mss gives BGRA format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Write the frame to the video output
            self.outputVideo.write(frame) 

    def videoRelease(self):        
        self.outputVideo.release()
        self.adjust_video_timescale()

    def screenRecord(self):
        self.videoInit()
        while True:
            self.frameCapture()
        return
    
    def adjust_video_timescale(self):
        """
        Adjusts the video timescale based on frame capture times.
        """
        # Load the video
        video = VideoFileClip(f"{self.fileName}.mp4")  
        # Total duration of the original video
        original_duration = video.duration
        # Calculate the actual total duration based on the frame capture times
        actual_total_duration = self.data['Time[s]'].to_numpy()[-1] - self.data['Time[s]'].to_numpy()[0]
        # Calculate the speed factor (ratio of original duration to actual duration)
        speed_factor = original_duration / actual_total_duration
        # Apply the speed factor to adjust the video's duration
        adjusted_clip = video.fx(lambda clip: clip.speedx(factor=speed_factor))
        # Write the output file with the adjusted timing
        adjusted_clip.write_videofile(f"{self.fileName}_adj.mp4", codec="libx264")
        return 

        





class Keithley4200:
    
    G0 = 7.74809173e-5
    
    def __init__(self, address: str = "GPIB0::3::INSTR"):
        self.address: str = address
        
        self.sample = "NWN_Pad68C"
        self.cell = 'N3S3'
        self.savepath = "Test"
        self.lab = 'INRiM4200' 
        self.script = '11_Vsin_and_verify'
        
        self.numSearch()
        self.date = datetime.today().strftime('%Y_%m_%d')
        self.temperature = 303
        
        self.v_set = 1                 # maximum positive voltage [V]
        self.i_range_set = 1e-3          # SMU current range in positive voltage branch [A]
        self.i_cc_set = 500e-6           # compliance current in positive voltage branch [A]
        self.step_set = 5e-3            # voltage sweep step in positive voltage branch [V]
        
        self.v_reset = -5              # minimum negative voltage [V]
        self.i_range_reset = 1e-1        # SMU current range in negative voltage branch [A]
        self.i_cc_reset = 1e-1          # compliance current in negative voltage branch [A]
        self.step_reset = 1e-2           # voltage sweep step in negative voltage branch [V]        
        
        self.v_min_perc = 0              # minimum negative voltage [V]
        self.v_max_perc = 1              # minimum negative voltage [V]
        self.i_range_perc = 1e-1        # SMU current range in negative voltage branch [A]
        self.i_cc_perc = 1e-1          # compliance current in negative voltage branch [A]
        self.step_perc = 1e-2           # voltage sweep step in negative voltage branch [V]        
                
        
        self.v_read = 0.5    #0.19          # reading voltage after plateau detection [V]
        self.i_range_read = 0         # SMU current range for reading [A]
        self.i_cc_read = 1e-1           # compliance current for reading [A]        
        self.AvLen = 10                   # number of consecutive equal conductances for reading triggering 
        self.G0Target = [0.12, 0.2, 0.3]            # list of G0 multiples to be searched
        self.edges = [0.01, 0.1]          # inferior and superior interval aroung G0Target values in which the last AvLen conductances must lie 
        self.t_read = 3600*2           # length of constant voltage stimulation [s]
        self.t_step_read = 0.1             # time spacing between readings [s]
        
        self.v_amp = -0.05            # sin voltage amplitude [V]
        self.v_bias = self.v_read           # sin voltage baseline [V]
        self.sin_f = 0.1             # sin frequency [Hz]
        self.i_range_sin = 0         # SMU current range for reading [A]
        self.i_cc_sin = 1e-1           # compliance current for reading [A]        
        self.perNumSin = 100            # length of sin stimulation [number of periods]
        self.t_step_sin = 0.1  # time spacing between readings [s]
        
        self.v_pulse = 1
        self.t_pulse = 10
        self.v_pulse_read = 0.01
        self.i_range_pulse = 0         # SMU current range for reading [A]
        self.i_cc_pulse = 1e-1           # compliance current for reading [A]
        self.t_read_pre = 10
        self.t_read_post = 10
        self.v_pulse_up = 0
        self.perNumPulse = 5
        self.t_step_pulse = 0.1
        self.t_read_inter = 5
        self.t_pulse_reset = 10
        self.v_pulse_reset = 1
        self.t_read_end = 0
        
        self.i_range_custom = 0         # SMU current range for reading [A]
        self.i_cc_custom = 1e-1           # compliance current for reading [A]
        
        self.currNPLC = 1
        self.voltNPLC = 1
        self.voltSourRange = 0
        self.voltSensRange = 0
        
        
        self.region = (477,132,1165,875)
        
        self.cycles = 1                 # number of I-V sweeps cycles 

        self.saveDelay = 10*60*60
        
        self.plotLabels = {'Title':['Voltage','Current','I-V','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','Temperature[K]']}
        
        self.temperatureMeas = False
        self.VRead = True
        self.Plot = False
        self.G0Found = False


        self.ch_source = 1
        self.ch_ground = 2
        self.sourceFlag = False
        
        #######################################################################
        #                         PMU ATTRIBUTES
        #######################################################################

        self.pulses_number = 5
        self.seg_number = 4
        self.pulse_width = 10e-8
        self.rising_time = 2e-8
        self.v_amp_PMU = 2
        self.v_bias_PMU = 0
        self.i_range_PMU = 1e-3
        self.last_sequence = 0
        
        
# =============================================================================
#                       INSTRUMENT SINGLE FUNCTIONS
# =============================================================================

    def InstInit(self):
        resourceManager = visa.ResourceManager()
        self.session = resourceManager.open_resource(self.address)
        self.session.write_termination = '\0'
        self.session.read_termination = '\0' 
        self.session.timeout = None   
        #self.session.query('BC DR1')
        self.session.query('DR0')
        self.session.query('BC')
        #self.SPol()        
        print(f"Keithley4200 session opened at address {self.address}") 
        return     
        
    def KT4200Init(self,i_range,i_cc,vIn):      
        """ if not self.VRead: """
        self.reset()        
        # General
        self.session.query(f"SS RP SMU{self.ch_ground}, 2")
        self.session.query(f"SS RP SMU{self.ch_source}, 2")
        # Integration
        self.session.query(f"IT4, 1.3, 3, {self.currNPLC}")        
        # Set current range
        if i_range == 0:
            self.session.query(f'RG{self.ch_source}, 1e-9')  # Lowest current range 
            print("Autorange ON - minimum range = 1e-9")
        else:            
            self.session.query(f"RG{self.ch_source}, {i_range}")
            print(f"Autorange ON - minimum range = {i_range}")
        return
    
    def SPol(self):
        while int(self.session.query('SP')) > 15:
            print(self.session.query('SP'))
            time.sleep(1e-15)        
        return 
    
    def rangeSet(self,i_range,i_cc):
        if i_range == 0:
            self.session.query(f'RG{self.ch_source}, 1e-9')  # Lowest current range 
            print("Autorange ON - minimum range = 1e-9")
        else:            
            self.session.query(f"RG{self.ch_source}, {i_range}")
            print(f"Autorange ON - minimum range = {i_range}")
        return
    
    def KT4200Start(self,i_cc):     
        self.session.query('US')
        self.session.query(f'DV{self.ch_ground}, 1, 1E-6, {i_cc}')      
        self.t0 = t0 = time.time()    
        return
    
        
    def KT4200Meas(self,v_point,i_cc,const = False): 
             
        if not const or not self.sourceFlag:
            self.session.query(f'DV{self.ch_source}, 1, {v_point}, {i_cc}')      
            self.sourceFlag = True
        t = time.time()
        if self.VRead:
            _ , v ,  = self.session.query('TV1').split('V')
        else:
            v = v_point
        _ , i ,  = self.session.query('TI1').split('I')                
        v = float(v)
        i = float(i)
        r = v/i
        t = float(t)-self.t0
        if r == 0:
            g = float("nan")
        else:
            g = i/v
        gnorm = g/self.G0
        self.data.loc[len(self.data)]=([t, v_point, v, i, r, 0, self.temperature, gnorm])
        return
    
    def PMUInit(self):
        self.last_sequence += 1

        self.session.query(":PMU:INIT 1") #0 = Pulse mode, 1 = SegArb mode

        self.session.query(":PMU:RPM:CONFIGURE PMU1-1, 0") # 0 = PMU, 1 = CV_2W, 2 = SMU, 3 = CV_4W
        self.session.query(f":PMU:MEASURE:RANGE {self.ch_source}, 2, {self.i_range_PMU}") # set current range, 0 = Autorange, 1 = Limited Autorange (minimum range), 2 = Fixed range
        self.session.query(f":PMU:SARB:SEQ:TIME {self.ch_source}, {self.last_sequence}, {self.SEGTIME}") # defines time sequence, Channel, sequence number (1 to 512), comma separated array of time segments (20 ns resolution)
        self.session.query(f":PMU:SARB:SEQ:STARTV {self.ch_source}, {self.last_sequence}, {self.STARTV_1}")
        self.session.query(f":PMU:SARB:SEQ:STOPV {self.ch_source}, {self.last_sequence}, {self.STOPV_1}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:TYPE {self.ch_source}, {self.last_sequence}, {self.MEASTYPE}") # 0 = No measurement, 1 = Spot Mean (1 sample for seg), 2 = Waveform disrete (more samples for pulse) 
        self.session.query(f":PMU:SARB:SEQ:MEAS:START {self.ch_source}, {self.last_sequence}, {self.MEASSTART}") # set delay for measurement after segment start (0 for measuring all)
        self.session.query(f":PMU:SARB:SEQ:MEAS:STOP {self.ch_source}, {self.last_sequence}, {self.MEASSTOP}") # set time of end measuring for each segment (set equal to SEGTIME)
        
        self.session.query(":PMU:RPM:CONFIGURE PMU1-2, 0") #Configure the RPM input for channel 2 of pulse card 1 to the PMU
        self.session.query(f":PMU:MEASURE:RANGE {self.ch_ground}, 2, {self.i_range_PMU}") #Set channel 2 for a fixed 1 mA current range
        self.session.query(f":PMU:SARB:SEQ:TIME {self.ch_ground}, {self.last_sequence}, {self.SEGTIME}") #Use the same segment time as sequence 1 of channel 1 for sequence 1 of channel 2
        self.session.query(f":PMU:SARB:SEQ:STARTV {self.ch_ground}, {self.last_sequence}, {self.STARTV_2}")
        self.session.query(f":PMU:SARB:SEQ:STOPV {self.ch_ground}, {self.last_sequence}, {self.STOPV_2}") # set V all to 0?
        self.session.query(f":PMU:SARB:SEQ:MEAS:TYPE {self.ch_ground}, {self.last_sequence}, {self.MEASTYPE}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:START {self.ch_ground}, {self.last_sequence}, {self.MEASSTART}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:STOP {self.ch_ground}, {self.last_sequence}, {self.MEASSTOP}")

        self.session.query(f":PMU:SARB:WFM:SEQ:LIST {self.ch_source}, {self.last_sequence}, {self.pulses_number}") # define for each sequence how many times is in output
        self.session.query(f":PMU:SARB:WFM:SEQ:LIST {self.ch_ground}, {self.last_sequence}, {self.pulses_number}")

    def PMUAddLastSequence(self):
        self.last_sequence += 1

        self.session.query(f":PMU:SARB:SEQ:TIME {self.ch_source}, {self.last_sequence}, {self.SEGTIME_last}") # defines time sequence, Channel, sequence number (1 to 512), comma separated array of time segments (20 ns resolution)
        self.session.query(f":PMU:SARB:SEQ:STARTV {self.ch_source}, {self.last_sequence}, {self.STARTV_1_last}")
        self.session.query(f":PMU:SARB:SEQ:STOPV {self.ch_source}, {self.last_sequence}, {self.STOPV_1_last}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:TYPE {self.ch_source}, {self.last_sequence}, {self.MEASTYPE_last}") # 0 = No measurement, 1 = Spot Mean (1 sample for seg), 2 = Waveform disrete (more samples for pulse) 
        self.session.query(f":PMU:SARB:SEQ:MEAS:START {self.ch_source}, {self.last_sequence}, {self.MEASSTART_last}") # set delay for measurement after segment start (0 for measuring all)
        self.session.query(f":PMU:SARB:SEQ:MEAS:STOP {self.ch_source}, {self.last_sequence}, {self.MEASSTOP_last}") # set time of end measuring for each segment (set equal to SEGTIME)

        self.session.query(f":PMU:SARB:SEQ:TIME {self.ch_ground}, {self.last_sequence}, {self.SEGTIME_last}") #Use the same segment time as sequence 1 of channel 1 for sequence 1 of channel 2
        self.session.query(f":PMU:SARB:SEQ:STARTV {self.ch_ground}, {self.last_sequence}, {self.STARTV_2_last}")
        self.session.query(f":PMU:SARB:SEQ:STOPV {self.ch_ground}, {self.last_sequence}, {self.STOPV_2_last}") # set V all to 0?
        self.session.query(f":PMU:SARB:SEQ:MEAS:TYPE {self.ch_ground}, {self.last_sequence}, {self.MEASTYPE_last}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:START {self.ch_ground}, {self.last_sequence}, {self.MEASSTART_last}")
        self.session.query(f":PMU:SARB:SEQ:MEAS:STOP {self.ch_ground}, {self.last_sequence}, {self.MEASSTOP_last}")

        self.session.query(f":PMU:SARB:WFM:SEQ:LIST:ADD {self.ch_source}, {self.last_sequence}, 1") # define for each sequence how many times is in output
        self.session.query(f":PMU:SARB:WFM:SEQ:LIST:ADD {self.ch_ground}, {self.last_sequence}, 1")







    def PMUExecute(self):
        
        self.session.query(f":PMU:OUTPUT:STATE {self.ch_source}, 1") # set to 1 = ON before EXECUTE (turns on after execute0, set to 0 = OFF at the end of test)
        self.session.query(f":PMU:OUTPUT:STATE {self.ch_ground}, 1")        
        self.session.query(":PMU:EXECUTE") # performs checks and starts test execution 
        while True:
            status = self.session.query(":PMU:TEST:STATUS?")    
            if int(status) == 0:
                print("Measurement Complete.")
                break  
        self.session.query(f":PMU:OUTPUT:STATE {self.ch_source}, 0")
        self.session.query(f":PMU:OUTPUT:STATE {self.ch_ground}, 0")
            
        
    def sourceOFF(self):
        self.session.query('DV1;DV2')    
        self.session.query(f"SS RP SMU{self.ch_ground}, 0")
        self.session.query(f"SS RP SMU{self.ch_source}, 0")   
        self.session.query('SP')
        self.sourceFlag = False
        return
        
    
    def reset(self):        
        self.session.query("*RST")
        return
    
        
    def sendCommand(self,command):
        self.session.write(f"{command}") 
        return
    
    def query(self,command):
        out = self.session.query(f"{command}") 
        return out    
    
    def closeSession(self):        
        self.session.close()  
        return
    
        
# =============================================================================
#                       TEMPERATURE CONTROL FUNCTIONS
# =============================================================================
         
    def heatStart(self,wait = True,address = "GPIB0::11::INSTR"):
        self.LS331 = LakeShore331(address)
        
        self.LS331.output_1.setpoint = self.temperature
        self.LS331.output_1.heater_range = "low"        
        self.temperatureMeas = True
        if wait:
            self.LS331.input_A.wait_for_temperature()
        return
   
    
    def heatStop(self):
        self.LS331.output_1.heater_range = 'off'
        self.temperatureMeas = False
        return
        
# =============================================================================
#                         DATA HANDLING FUNCTIONS
# =============================================================================
    def checkPath(self):
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)
        return
            
        
    def numSearch(self):
        pathSearch = f"{self.savepath}/*txt"
        try:
            filelst = glob.glob(pathSearch)
            numlst = [int(filelst[i].split("\\")[-1].split('_')[0]) for i in range(filelst.__len__())]
            self.startNum = max(numlst)+1
        except:
            self.startNum = 1
        return
            
            
    def plotGen(self):
        plt.ion()    
        self.fig,((self.ax1,self.ax2),(self.ax3,self.ax4)) = plt.subplots(2,2,sharex = False, sharey = False, figsize=(13,8))
        self.plot1, = self.ax1.plot(0,0)
        self.ax1.set_ylabel(self.plotLabels['ylabel'][0])
        self.ax1.set_title(self.plotLabels['Title'][0])
        self.ax1.set_xlabel(self.plotLabels['xlabel'][0])    
        self.plot2, = self.ax2.plot(0,0)
        self.ax2.set_ylabel(self.plotLabels['ylabel'][1])
        self.ax2.set_title(self.plotLabels['Title'][1])
        self.ax2.set_xlabel(self.plotLabels['xlabel'][1])        
        self.plot3, = self.ax3.plot(0,0)
        self.ax3.set_ylabel(self.plotLabels['ylabel'][2])
        self.ax3.set_title(self.plotLabels['Title'][2])
        self.ax3.set_xlabel(self.plotLabels['xlabel'][2])
        self.plot4, = self.ax4.plot(0,0)
        self.ax4.set_ylabel(self.plotLabels['ylabel'][3])
        self.ax4.set_title(self.plotLabels['Title'][3])
        self.ax4.set_xlabel(self.plotLabels['xlabel'][3])
        plt.show()     
        return
    
        
    def PlotUpdate(self):
        self.plot1.set_xdata(self.data[self.plotLabels['xlabel'][0]])
        self.plot1.set_ydata(self.data[self.plotLabels['ylabel'][0]])      
        self.ax1.relim()
        self.ax1.autoscale_view()    
        self.plot2.set_xdata(self.data[self.plotLabels['xlabel'][1]])
        self.plot2.set_ydata(self.data[self.plotLabels['ylabel'][1]])           
        self.ax2.relim()
        self.ax2.autoscale_view()    
        self.plot3.set_xdata(self.data[self.plotLabels['xlabel'][2]])
        self.plot3.set_ydata(self.data[self.plotLabels['ylabel'][2]])              
        self.ax3.relim()
        self.ax3.autoscale_view() 
        self.plot4.set_xdata(self.data[self.plotLabels['xlabel'][3]])
        self.plot4.set_ydata(self.data[self.plotLabels['ylabel'][3]])                
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  
        return
    
    def PMUplot(self):
        self.plotLabels = {'Title':['Current','Conductance','Voltage','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Voltage_prog[V]','Voltage_prog[V]','Temperature[K]'],
                           'ylabelr':['Current[A]','GNorm[G0]','Voltage_read[V]','Temperature[K]']}
        plt.ion()    
        self.fig,((self.ax1,self.ax2),(self.ax3,self.ax4)) = plt.subplots(2,2,sharex = False, sharey = False, figsize=(13,8))
        self.plot1, = self.ax1.plot(0,0)
        self.ax1.set_ylabel(self.plotLabels['ylabel'][0])
        self.ax1.set_title(self.plotLabels['Title'][0])
        self.ax1.set_xlabel(self.plotLabels['xlabel'][0])  
        self.ax1.plot(self.data[self.plotLabels['xlabel'][0]],self.data[self.plotLabels['ylabel'][0]],color='k')

        self.plot2, = self.ax2.plot(0,0)
        self.ax2.set_ylabel(self.plotLabels['ylabel'][1])
        self.ax2.set_title(self.plotLabels['Title'][1])
        self.ax2.set_xlabel(self.plotLabels['xlabel'][1]) 
        self.ax2.plot(self.data[self.plotLabels['xlabel'][1]],self.data[self.plotLabels['ylabel'][1]],color='k')

        self.plot3, = self.ax3.plot(0,0)
        self.ax3.set_ylabel(self.plotLabels['ylabel'][2])
        self.ax3.set_title(self.plotLabels['Title'][2])
        self.ax3.set_xlabel(self.plotLabels['xlabel'][2])
        self.ax3.plot(self.data[self.plotLabels['xlabel'][2]],self.data[self.plotLabels['ylabel'][2]],color='k')

        self.plot4, = self.ax4.plot(0,0)
        self.ax4.set_ylabel(self.plotLabels['ylabel'][3])
        self.ax4.set_title(self.plotLabels['Title'][3])
        self.ax4.set_xlabel(self.plotLabels['xlabel'][3])
        self.ax4.plot(self.data[self.plotLabels['xlabel'][3]],self.data[self.plotLabels['ylabel'][3]],color='k')

        self.ax5 = self.ax1.twinx()
        self.ax5.set_ylabel(self.plotLabels['ylabelr'][0], color='r')
        self.ax5.tick_params(axis='y', labelcolor='r')
        self.ax5.plot(self.data[self.plotLabels['xlabel'][0]],self.data[self.plotLabels['ylabelr'][0]],color='r')

        self.ax6 = self.ax2.twinx()
        self.ax6.set_ylabel(self.plotLabels['ylabelr'][1], color='r')
        self.ax6.tick_params(axis='y', labelcolor='r')
        self.ax6.plot(self.data[self.plotLabels['xlabel'][0]],self.data[self.plotLabels['ylabelr'][1]],color='r')

        self.ax7 = self.ax3.twinx()
        self.ax7.set_ylabel(self.plotLabels['ylabelr'][2], color='r')
        self.ax7.tick_params(axis='y', labelcolor='r')
        self.ax7.plot(self.data[self.plotLabels['xlabel'][0]],self.data[self.plotLabels['ylabelr'][2]],color='r') 
        plt.tight_layout()
        plt.show()     

        return
    
    
    def PlotSave(self):        
        self.fig.savefig(f"{self.fileName}.png", dpi=300) 
        self.Plot = False
        return
    
    
    def PlotClose(self):
        plt.close('all')
        return
    
    
    def dataInit(self):
        self.data = pd.DataFrame(columns=['Time[s]','Voltage_prog[V]','Voltage_read[V]','Current[A]','Resistance[ohm]','Temperature[K]','TargetTemperature[K]','GNorm[G0]'])
        return
    
    
    def printData(self,header = False):  
        if header:
            head = (" ").join(list(self.data.columns.values))
            print(f"{head}")
        lastRow = (" ").join([str(x) for x in self.data.loc[len(self.data)-1].values.tolist()])
        print(f"{lastRow}\n")  
        return
    
    def printG0(self):  
        lastRow = (" ").join([str(x) for x in self.data.loc[len(self.data)-1].values.tolist()])
        print(f"G = {self.data['GNorm[G0]'].loc[len(self.data)-1]} G0\n")  
        return
    
    
    def saveData(self):        
        self.data.to_csv(f"{self.fileName}.txt",sep = ' ',index = False,na_rep="NaN")  
        return
    
        
    def exitSave(self):        
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        self.PlotUpdate()               
        self.PlotSave()
        self.saveData()       
        return
        
    def PMUGetData(self):            
        data_points = int(self.query(":PMU:DATA:COUNT? 1"))
        df_all_channels_1 = pd.DataFrame(columns=['Voltage', 'Current', 'Timestamp', 'Status'])
        df_all_channels_2 = pd.DataFrame(columns=['Voltage', 'Current', 'Timestamp', 'Status'])
        for start_point in range(0, data_points, 2048):            
            response = self.query(f":PMU:DATA:GET {self.ch_source}, {start_point}, 2048")
            coords = response.split(";")
            coords2d = [value.split(",") for value in coords]
            df_chunk = pd.DataFrame(coords2d, columns=['Voltage', 'Current', 'Timestamp', 'Status'])
            df_all_channels_1 = pd.concat([df_all_channels_1, df_chunk])
            response_2 = self.query(f":PMU:DATA:GET {self.ch_ground}, {start_point}, 2048")
            coords_2 = response_2.split(";")
            coords2d_2 = [value.split(",") for value in coords_2]
            df_chunk_2 = pd.DataFrame(coords2d_2, columns=['Voltage', 'Current', 'Timestamp', 'Status'])
            df_all_channels_2 = pd.concat([df_all_channels_2, df_chunk_2])
        df_all_channels_1.reset_index(drop=True, inplace=True)
        df_all_channels_2.reset_index(drop=True, inplace=True)
        self.data['Time[s]'] = df_all_channels_1['Timestamp'].astype(float)
        self.data['Voltage_read[V]'] = df_all_channels_1['Voltage'].astype(float)
        self.data['Current[A]'] = -df_all_channels_2['Current'].astype(float)
        self.data[['Temperature[K]','TargetTemperature[K]']] = np.nan
        dT = np.mean(np.diff(self.data['Time[s]']))
        voltage_single_seq = []
        """ SEGTIME_vector_full = np.append(self.SEGTIME_vector, self.SEGTIME_vector[-1])
        STARTV_1_vector_full = np.append(self.STARTV_1_vector, self.STARTV_1_vector[-1])
        STOPV_1_vector_full = np.append(self.STOPV_1_vector, self.STOPV_1_vector[-1]) """
        for seg_time, start_v, stop_v in zip(self.SEGTIME_vector, self.STARTV_1_vector, self.STOPV_1_vector):
            # Determine the number of steps in this segment
            num_steps = int(np.floor(seg_time / dT))
            # Generate linearly spaced voltages for this segment
            segment_voltages = np.linspace(start_v, stop_v, num_steps)
            voltage_single_seq.extend(segment_voltages)
        voltage_full_seq = np.tile(voltage_single_seq, self.pulses_number)
        
        num_steps = int(np.floor(self.SEGTIME_vector[0] / dT))
        # Generate linearly spaced voltages for this segment
        segment_voltages = np.linspace(self.STARTV_1_vector[0], self.STOPV_1_vector[0], num_steps)
        voltage_full_seq.extend(segment_voltages)

        self.data['Voltage_prog[V]'] = voltage_full_seq
        self.data['Resistance[ohm]'] = self.data['Voltage_prog[V]']/self.data['Current[A]']
        self.data['GNorm[G0]'] = self.data['Current[A]']/(self.data['Voltage_prog[V]']*self.G0)
        
        return
        
   
        
    def G0Check(self):
        G0TargetLow = [self.G0Target[i]-self.edges[0] for i in range(self.G0Target.__len__())]
        G0TargetHigh = [self.G0Target[i]+self.edges[1] for i in range(self.G0Target.__len__())]
        for tarNum in range(self.G0Target.__len__()):            
            if all(el > G0TargetLow[tarNum] for el in self.data['GNorm[G0]'].loc[-self.AvLen:]) and all(el < G0TargetHigh[tarNum] for el in self.data['GNorm[G0]'].loc[-self.AvLen:]):
                self.G0found = True
                break
        return
    
    
            
# =============================================================================
#                           WAVEFORM GENERATORS
# =============================================================================

    def sweepGenSet(self):
        v_sweep1 = np.linspace(0, self.v_set, int((self.v_set/self.step_set)+1))
        v_sweep2 = v_sweep1[-2:0:-1]    
        self.vInSet = np.concatenate((v_sweep1,v_sweep2))
        self.ptNumSet = self.vInSet.__len__()            
        return
    

    def sweepGenReset(self):    
        v_sweep3 = np.linspace(0, -self.v_reset, int(abs(self.v_reset/self.step_reset)+1))
        v_sweep4 = v_sweep3[-2:0:-1]
        self.vInReset = np.concatenate((v_sweep3,v_sweep4,[0]))
        self.ptNumReset = self.vInReset.__len__()   
        return
    
    
    def sweepGenFull(self): 
        self.sweepGenSet()
        self.sweepGenReset()           
        self.vInFull = np.concatenate((self.vInSet,self.vInReset))
        self.ptNumFull = self.vInFull.__len__()
        return


    def sweepGenPerc(self):
        self.vInPerc = np.linspace(self.v_min_perc, self.v_max_perc, int(((self.v_max_perc-self.v_min_perc)/self.step_perc)+1))
        self.ptNumPerc = self.vInPerc.__len__()        
        return
    
    def sweepGenConst(self):    #EDIT
        self.vInPerc = np.linspace(self.v_min_perc, self.v_max_perc, int(((self.v_max_perc-self.v_min_perc)/self.step_perc)+1))
        self.ptNumPerc = self.vInPerc.__len__() 
        self.ptNumRead = int((self.t_read/self.t_step_read)+1)
        self.vInRead = np.full((self.ptNumRead,), self.v_read, dtype=float)    
        self.vInPerc = np.concatenate([self.vInPerc,self.vInRead])   
        self.ptNumPerc = self.ptNumPerc + self.ptNumRead
        return


    # def sweepGenCurr(v_pos, v_neg, step):
    #     v_sweep1 = np.linspace(v_neg, v_pos, int((v_pos/step)+1))
    #     v_sweep2 = v_sweep1[-2:0:-1]
    #     v_sweep3 = np.linspace(0, v_neg, int(abs(v_neg/step)+1))
    #     v_sweep4 = v_sweep3[-2:0:-1]
    #     # v_sweep_np = np.concatenate((v_sweep1,v_sweep2))
    #     v_sweep_np = np.concatenate((v_sweep1,v_sweep2,v_sweep3,v_sweep4))
    #     ptNum = v_sweep1.__len__()
    #     # v_sweep = ', '.join([str(i) for i in v_sweep_np])
    #     return v_sweep1, ptNum
    
    
    def readTableGen(self):
        self.ptNumRead = int((self.t_read/self.t_step_read)+1)
        self.vInRead = np.full((self.ptNumRead,), self.v_read, dtype=float)    
        self.tInRead = np.linspace(0, self.t_read, self.ptNumRead)        
        return
    
    
    def sinGen(self):
        t_end = 1/self.sin_f*self.perNumSin
        self.ptNumSin = int((t_end/self.t_step_sin)+1) 
        self.tInSin = np.linspace(0, t_end, self.ptNumSin)      
        self.vInSin = self.v_amp * np.sin(2 * np.pi * self.sin_f * self.tInSin) + self.v_bias    
        return
    
    
    def pulseTrainGen(self):
        ptNumPulse = int((self.t_pulse/self.t_step_pulse))
        ptNumReadPre = int((self.t_read_pre/self.t_step_pulse))
        ptNumReadPost = int((self.t_read_post/self.t_step_pulse))
        pulseSeg = np.full(ptNumPulse,self.v_pulse)
        readPreSeg = np.full(ptNumReadPre,self.v_pulse_read)
        readPostSeg = np.full(ptNumReadPost,self.v_pulse_read)
        self.vInPulse = np.concatenate([readPreSeg,pulseSeg,readPostSeg])

        nextSegment = np.concatenate([pulseSeg*(self.t_pulse>0),readPostSeg*(self.t_read_post>0)])
        for i in range(self.perNumPulse-1):
            self.vInPulse = np.concatenate([self.vInPulse,nextSegment])

        if self.t_read_end > 0:
            ptNumReadEnd = int(((self.t_read_end)/self.t_step_pulse))
            readEndSeg = np.full(ptNumReadEnd,self.v_pulse_read)
            self.vInPulse = np.concatenate([self.vInPulse,readEndSeg])

        self.ptNumPulse = self.vInPulse.__len__()
        self.tInPulse = np.linspace(0, self.t_step_pulse*(self.ptNumPulse), self.ptNumPulse)  
        return
    
    def pulseTrainSetResetGen(self):
        ptNumReadPre = int((self.t_read_pre/self.t_step_pulse))
        ptNumPulse = int((self.t_pulse/self.t_step_pulse))
        ptNumReadInter = int((self.t_read_inter/self.t_step_pulse))
        ptNumPulseReset = int((self.t_pulse_reset/self.t_step_pulse))
        ptNumReadPost = int((self.t_read_post/self.t_step_pulse))        

        readPreSeg = np.full(ptNumReadPre,self.v_pulse_read)
        pulseSeg = np.full(ptNumPulse,self.v_pulse)
        readInterSeg = np.full(ptNumReadInter,self.v_pulse_read)
        pulseResetSeg = np.full(ptNumPulseReset,-self.v_pulse_reset)
        readPostSeg = np.full(ptNumReadPost,self.v_pulse_read)
        

        self.vInPulse = np.concatenate([readPreSeg*(self.t_read_pre>0),pulseSeg*(self.t_pulse>0),readInterSeg*(self.t_read_inter>0),pulseResetSeg*(self.t_pulse_reset>0),readPostSeg*(self.t_read_post>0)])
        
        nextSegment = np.concatenate([pulseSeg*(self.t_pulse>0),readInterSeg*(self.t_read_inter>0),pulseResetSeg*(self.t_pulse_reset>0),readPostSeg*(self.t_read_post>0)])

        for i in range(self.perNumPulse-1):
            self.vInPulse = np.concatenate([self.vInPulse,nextSegment])
        
        if self.t_read_end > 0:
            ptNumReadEnd = int(((self.t_read_end)/self.t_step_pulse))
            readEndSeg = np.full(ptNumReadEnd,self.v_pulse_read)
            self.vInPulse = np.concatenate([self.vInPulse,readEndSeg])
            
        self.ptNumPulse = self.vInPulse.__len__()
        self.tInPulse = np.linspace(0, self.t_step_pulse*(self.ptNumPulse), self.ptNumPulse)  
        return
    
    def PMUSquareWaveGen(self):
        self.v_high = self.v_bias_PMU + self.v_amp_PMU
        self.v_low = self.v_bias_PMU - self.v_amp_PMU
        self.SEGTIME_vector =  np.array([self.pulse_width if i % 2 == 0 else self.rising_time for i in range(self.seg_number)])
        self.SEGTIME = ", ".join(map(str,self.SEGTIME_vector))
        self.STARTV_1_vector = np.array([self.v_low if (i // 2) % 2 == 0 else self.v_high for i in range(self.seg_number)])
        self.STARTV_1 = ", ".join(map(str, self.STARTV_1_vector))
        self.STOPV_1_vector = np.roll(np.array([self.v_low if (i // 2) % 2 == 0 else self.v_high for i in range(self.seg_number)]), -1)
        self.STOPV_1 = ", ".join(map(str, self.STOPV_1_vector))
        self.MEASTYPE = ", ".join(map(str, np.full(self.seg_number, 2)))
        self.MEASSTART = ", ".join(map(str, np.zeros(self.seg_number)))
        self.MEASSTOP = self.SEGTIME
        self.STARTV_2 = ", ".join(map(str, np.zeros(self.seg_number)))
        self.STOPV_2 = ", ".join(map(str, np.zeros(self.seg_number)))

        """ self.SEGTIME_last = str(self.SEGTIME_vector[0])
        self.STARTV_1_last = str(self.STARTV_1_vector[0])
        self.STOPV_1_last = str(self.STOPV_1_vector[0])
        self.MEASTYPE_last = str(2)
        self.MEASSTART_last = str(0)
        self.MEASSTOP_last = self.SEGTIME_last
        self.STARTV_2_last = str(0)
        self.STOPV_2_last = str(0)
 """
        self.SEGTIME_last = ", ".join(map(str, np.full(3, self.SEGTIME_vector[0]/3)))
        self.STARTV_1_last = ", ".join(map(str, np.full(3, self.STARTV_1_vector[0])))
        self.STOPV_1_last = ", ".join(map(str, np.full(3, self.STOPV_1_vector[0])))
        self.MEASTYPE_last = ", ".join(map(str, np.full(3, 2)))
        self.MEASSTART_last = ", ".join(map(str, np.full(3, 0)))
        self.MEASSTOP_last = self.SEGTIME_last
        self.STARTV_2_last = ", ".join(map(str, np.full(3, 0)))
        self.STOPV_2_last = ", ".join(map(str, np.full(3, 0)))


        

    
    
# =============================================================================
# #                             EXPERIMENTS
# =============================================================================
        
    def rampRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenPerc()     
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT4200Start(self.i_cc_perc)    
        for rNum in range(self.ptNumPerc):                                            
            self.KT4200Meas(self.vInPerc[rNum],self.i_cc_perc)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()         
        self.saveData()
        self.PlotSave()
        return
        
        
    def rampCheckRun(self,Plot = True):  
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenPerc()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT4200Start(self.i_cc_perc)
        for rNum in range(self.ptNumPerc):                                            
            self.KT4200Meas(self.vInPerc[rNum],self.i_cc_perc)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A 
            self.printG0()
            self.G0Check()
            if self.G0Found:
                break
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPerc-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()   
        self.PlotUpdate()
        self.saveData()
        self.PlotSave() 
        return
    
    def rampConstRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenConst()     
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_perc,self.i_cc_perc,self.vInPerc[0])
        self.KT4200Start(self.i_cc_perc)    
        for rNum in range(self.ptNumPerc):                                            
            self.KT4200Meas(self.vInPerc[rNum],self.i_cc_perc)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()         
        self.saveData()
        self.PlotSave()
        return
        
        
    """ def sweepFullRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenFull()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT4200Start()
        for rNum in range(self.ptNumFull):                                            
            self.KT4200Meas(self.vInFull[rNum])
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumFull-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                self.PlotSave()
        self.saveData()
        self.PlotSave()
        return """
    
    def sweepFullRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenSet()  
        self.sweepGenReset()
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT4200Start(self.i_cc_set)
        for rNum in range(self.ptNumSet):                                            
            self.KT4200Meas(self.vInSet[rNum],self.i_cc_set)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0
        self.rangeSet(self.i_range_reset,self.i_cc_reset)
        for rNum in range(self.ptNumReset):  
            self.KT4200Meas(self.vInReset[rNum],self.i_cc_reset)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0    
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        self.PlotUpdate()
        self.PlotSave()
        self.saveData()        
        return
    
    
    def constRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.readTableGen()  
        
        tDiff = np.diff(self.tInRead)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_const_{self.date}"               
        print(self.fileName)
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_read,self.i_cc_read,self.vInRead[0])
        self.KT4200Start(self.i_cc_read)
        tPre = time.time() 
        for rNum in range(self.ptNumRead):                                            
            self.KT4200Meas(self.vInRead[rNum],self.i_cc_read,const = True)
            count += 1
            
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1 or (self.data['Time[s]'].loc[len(self.data)-1]) >= self.t_read:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
                break
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
        
    def sinRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sinGen()  
        tDiff = np.diff(self.tInSin)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sin_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_sin,self.i_cc_sin,self.vInSin[0])
        self.KT4200Start(self.i_cc_sin)
        tPre = time.time() 
        for rNum in range(self.ptNumSin):                                            
            self.KT4200Meas(self.vInSin[rNum],self.i_cc_sin)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumSin-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
        
    def pulseTrainRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
    def pulseTrainSetResetRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainSetResetGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
    def pulseRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.perNumPulse = 1
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
        
    def customRun(self,tInCustom,vInCustom,Plot = True):
        self.tInCustom = tInCustom
        self.vInCustom = vInCustom
        self.ptNumCustom = tInCustom.__len__()
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0        
        tDiff = np.diff(self.tInCustom)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_custom_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_custom,self.i_cc_custom,self.vInCustom[0])
        self.KT4200Start(self.i_cc_custom)
        tPre = time.time() 
        for rNum in range(self.ptNumCustom):                                            
            self.KT4200Meas(self.vInCustom[rNum],self.i_cc_custom)
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumCustom-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.saveData()
        return
    
    def PMUSquareWaveRun(self,Plot = True):
        self.Plot = Plot
        
        self.dataInit()        
        self.checkPath()
        self.numSearch()        
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_PMU_{self.date}"

        self.PMUSquareWaveGen()
        self.PMUInit()  
        self.PMUAddLastSequence()      
        self.PMUExecute()
        self.PMUGetData()
        self.PMUplot()
        self.PlotSave()
        self.saveData()
        
        return
    
    
    
    
    def pulseTrainVideoRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):    
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.videoRelease()
        self.saveData()
        return
    
    def pulseTrainSetResetVideoRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.pulseTrainSetResetGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.input_A.temperature  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.videoRelease()
        self.saveData()
        return
    
    def pulseVideoRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.perNumPulse = 1
        self.pulseTrainGen()  
        tDiff = np.diff(self.tInPulse)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_pulse_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        
        self.KT4200Init(self.i_range_pulse,self.i_cc_pulse,self.vInPulse[0])
        self.KT4200Start(self.i_cc_pulse)
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumPulse):                                            
            self.KT4200Meas(self.vInPulse[rNum],self.i_cc_pulse)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumPulse-1:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.videoRelease()
        self.saveData()
        return
    
    def constVideoRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','GNorm','Temperature'],
                           'xlabel':['Time[s]','Time[s]','Time[s]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','GNorm[G0]','Temperature[K]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.readTableGen()  
        tDiff = np.diff(self.tInRead)
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_const_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_read,self.i_cc_read,self.vInRead[0])
        self.KT4200Start(self.i_cc_read)
        self.videoInit()
        tPre = time.time() 
        for rNum in range(self.ptNumRead):                                            
            self.KT4200Meas(self.vInRead[rNum],self.i_cc_read,const = True)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0               
            if rNum == self.ptNumRead-1 or (self.data['Time[s]'].loc[len(self.data)-1]-self.t0) >= self.t_read:
                if not self.Plot:
                    self.plotGen()
                    self.Plot = True
                self.PlotUpdate()               
                self.PlotSave() 
                break
            else:             
                tDel = tDiff[rNum] -(time.time()-tPre)                    
                if tDel > 0:
                    time.sleep(tDel)
                tPre = time.time() 
        self.videoRelease()
        self.saveData()
        return
    
    def sweepFullVideoRun(self,Plot = True):
        self.Plot = Plot
        self.plotLabels = {'Title':['Voltage','Current','I-V','GNorm'],
                           'xlabel':['Time[s]','Time[s]','Voltage_prog[V]','Time[s]'],
                           'ylabel':['Voltage_prog[V]','Current[A]','Current[A]','GNorm[G0]']}        
        self.dataInit()        
        self.checkPath()
        self.numSearch()
        count = 0
        self.sweepGenFull()  
        self.fileName = f"{self.savepath}/{str(self.startNum).zfill(3)}_{self.lab}_{self.sample}_{self.cell}_{self.script}_sweep_{self.date}"               
        if self.Plot:
            self.plotGen()
            self.saveDelay = 1
        else:
            self.saveDelay = 1e34
        self.KT4200Init(self.i_range_set,self.i_cc_set,self.vInSet[0])
        self.KT4200Start(self.i_cc_set)
        self.videoInit()
        for rNum in range(self.ptNumSet):                                            
            self.KT4200Meas(self.vInSet[rNum],self.i_cc_set)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0
        self.rangeSet(self.i_range_reset,self.i_cc_reset)
        for rNum in range(self.ptNumReset):  
            self.KT4200Meas(self.vInReset[rNum],self.i_cc_reset)
            self.frameCapture()
            count += 1
            if self.temperatureMeas:
                self.data['Temperature[K]'].loc[len(self.data)-1] = self.LS331.temperature_A  
            if count%self.saveDelay == 0:
                """ if not self.Plot:
                    self.plotGen()
                    self.Plot = True """
                self.PlotUpdate()
                count = 0    
        if not self.Plot:
            self.plotGen()
            self.Plot = True
        self.PlotUpdate()
        self.PlotSave()
        self.videoRelease()
        self.saveData()        
        return
    
    def videoInit(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        self.fps = 20.0 
        self.outputVideo = cv2.VideoWriter(f"{self.fileName}.mp4", fourcc, self.fps, (self.region[2], self.region[3]))  
        return

    """ def frameCapture(self):
        img = pyautogui.screenshot(region=self.region)    
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
        self.outputVideo.write(frame)
        return  """
    
    def frameCapture(self):
        with mss.mss() as sct:
            # Capture the screen based on the specified region
            
            mon = sct.monitors[1]  # monitor[2] refers to the second monitor

        # Adjust the region for the second monitor based on your self.region values
            monitor = {
                "top": mon["top"] + self.region[1],
                "left": mon["left"] + self.region[0],
                "width": self.region[2],
                "height": self.region[3],
                "mon": 1
            }
            # Capture the screenshot
            screenshot = sct.grab(monitor)
            
            # Convert the raw data from the screenshot to a NumPy array
            frame = np.array(screenshot)

            # Convert from BGRA to BGR (mss gives BGRA format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Write the frame to the video output
            self.outputVideo.write(frame) 

    def videoRelease(self):        
        self.outputVideo.release()
        self.adjust_video_timescale()

    def screenRecord(self):
        self.videoInit()
        while True:
            self.frameCapture()
        return
    
    def adjust_video_timescale(self):
        """
        Adjusts the video timescale based on frame capture times.
        """
        # Load the video
        video = VideoFileClip(f"{self.fileName}.mp4")  
        # Total duration of the original video
        original_duration = video.duration
        # Calculate the actual total duration based on the frame capture times
        actual_total_duration = self.data['Time[s]'].to_numpy()[-1] - self.data['Time[s]'].to_numpy()[0]
        # Calculate the speed factor (ratio of original duration to actual duration)
        speed_factor = original_duration / actual_total_duration
        # Apply the speed factor to adjust the video's duration
        adjusted_clip = video.fx(lambda clip: clip.speedx(factor=speed_factor))
        # Write the output file with the adjusted timing
        adjusted_clip.write_videofile(f"{self.fileName}_adj.mp4", codec="libx264")
        return 

        