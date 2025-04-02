import matplotlib.pyplot as plt
from Instruments import Keithley4200
import os

KT4200 = Keithley4200(address = "TCPIP0::10.60.5.103::1225::SOCKET")
KT4200.InstInit()
KT4200.sample = "TestPMU"         # Sample name
KT4200.cell = 'r10k'           # Measured cell
KT4200.savepath = "TestPMU"       # Folder name (automatically created if not existing)
KT4200.lab = 'INRiM4200'       # Lab and instrument of measurement execution
KT4200.script = os.path.basename(__file__).split('.')[0]

KT4200.pulses_number = 3
KT4200.v_bias_PMU = 0
KT4200.v_amp_PMU = 1
period = 10e-6
KT4200.pulse_width = period/2
KT4200.rising_time = period/50

KT4200.checkPath()
KT4200.numSearch()
KT4200.fileName = f"{KT4200.savepath}/{str(KT4200.startNum).zfill(3)}_{KT4200.lab}_{KT4200.sample}_{KT4200.cell}_{KT4200.script}_PMU_{KT4200.date}"

KT4200.PMUSquareGen()
KT4200.PMUInit()
KT4200.PMUExecute()
KT4200.dataInit()
KT4200.PMUGetData()
KT4200.PMUplot()
KT4200.PlotSave()
KT4200.saveData()

#KT4200.closeSession()