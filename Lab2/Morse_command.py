
import serial

serialPort = serial.Serial()
serialPort.baudrate = 9600
serialPort.port = "COM3"

serialPort.open()
print(serialPort)

serialPort.open()

serialPort.write("physics")
