
import numpy as np
import time as t
import matplotlib.pyplot as plt
import serial

arraySize = 500

serialPort = serial.Serial()
serialPort.baudrate = 9600
serialPort.port = "COM3"

serialPort.open()

means = np.zeros(256)
stds = np.zeros(256)
dout = np.zeros((256, 2))

for i in range(256):
    dataRead = False
    data = []
    dataOut = np.zeros(arraySize)
    
    while(dataRead == False):
        serialPort.write(chr(i))
        t.sleep(0.1)
    
        inByte = serialPort.inWaiting()
    
        byteCount = 0
        
        while((inByte > 0) & (byteCount < arraySize)):
            dataByte = serialPort.read()
            byteCount = byteCount + 1
            data = data + [dataByte]
            if(byteCount == arraySize):
                dataRead = True
    for j in range(arraySize):
        dataOut[j] = ord(data[j])
    means[i] = np.mean(dataOut)
    stds[i] = np.std(dataOut)
    dout[i][0] = means[i]
    dout[i][1] = stds[i]

serialPort.close()
np.savetxt('A2_means_stds.csv', dout, delimiter = ',')
plt.figure()
plt.title("Detected Value at Pin A2 for pin 9")
plt.plot(range(256), means, 'o')
plt.show()
