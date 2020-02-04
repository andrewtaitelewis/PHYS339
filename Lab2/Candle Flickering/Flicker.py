# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Serial
import time as t
import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sns


sns.set()

def getGaussRand(currentInt, prevInt):
    sigma = 0.5
    randGauss = -1
    tooSharp = True
    prevIntChange = currentInt - prevInt
    randomInt = 0
    intChangeBeforePower = 0
    intChange = 0
    tooSharpCount = 0
    
    if( currentInt ==0):
        return currentInt
    else:
        while(tooSharp == True):
        
            while((randGauss > 0.5 | randGauss < 0.0 | (tooSharp) == True)):
                rand = r.random()
                rand2 = r.random()
                
                randGauss = (sigma*np.sqrt(2*(-np.log(1-rand)))*np.cos(2*np.pi*rand2) + 0.5)
                tooSharp = False
                
        randomInt = 460.0*randGauss
        intChangeBeforePower = randomInt - currentInt
        
        if(intChangeBeforePower >= 0):
            intChange = (intChangeBeforePower)**(0.3)
        else:
            intChange = -(-intChangeBeforePower)**(0.3)
        
        if(((prevIntChange / intChange) > (0.6)) & ((prevIntChange / intChange) < (1.7))):
            tooSharp = False
        elif((prevIntChange < 0) & (prevIntChange > -1) & (intChange > 0)):
            tooSharp = False
        elif((prevIntChange > 0) & (prevIntChange < 1) & (intChange < 0)):
            tooSharp = False
        elif( tooSharpCount > 500):
            tooSharp = False
        else:
            tooSharp = True
            tooSharpCount = tooSharpCount + 1
    return (currentInt + intChange)

serialPort = serial.Serial()
serialPort.baudrate = 9600
serialPort.port = "COM03"
print(serialPort)
serialPort.open()


currentInt = 1.0
prevInt1 = -10.0

values = np.zeros(500)
ranged = range(500)

for i in ranged:
    if(i==0):
        values[i] = currentInt
    else:
         values[i] = getGaussRand(values[i-1], prevInt1)
         prevInt1 = values[i] - values[i-1]
         
values = values/51.0

f = plt.figure()
plt.plot(ranged,values,label = "Intensities Progression")
plt.xlabel('Iterations')
plt.ylabel('Candle Intensity [V]')
plt.legend(loc=4, prop={'size': 13} )

counter = 0
prevInt2 = -10.0

while(counter< 10):
    prevInt = currentInt
    currentInt = getGaussRand(currentInt,prevInt2)
    
    serialPort.write(chr(int(currentInt)))
    counter = counter + 1
    t.sleep(0.05)
    
while(1):
    
    prevInt = currentInt
    currentInt = getGaussRand(currentInt,prevInt2)
    prevInt2 = currentInt - prevInt
    
    
    serialPort.write(chr(int(currentInt)))
    t.sleep(0.005)

