"""
Author: Fabio Michieletti

e-mail: fabio.michieletti@polito.it

"""

import pyvisa as visa

names = []
address = []

resourceManager = visa.ResourceManager()
InstList = resourceManager.list_resources()
for nInst in range(InstList.__len__()):
    session = resourceManager.open_resource(InstList[nInst])
    try:
        names.append(session.query('*IDN?'))
        address.append(InstList[nInst])
        print(f"{names[-1]} ---> \n{address[-1]}\n")
    except:
        continue


# GPIB0::3::INSTR