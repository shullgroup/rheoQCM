from NITempSensor import TempSensor
import time
class Device():
    def __init__(self):
        self.name = 'Dev3'
        self.product_type = 'USB-TC01'

device = Device()
ai_channel = 'ai0'
thrmpl = 'J'
nsamples = 1
tempsensor = TempSensor(device, ai_channel, thrmpl, nsamples)

n = 5
t0 = time.time()
for i in range(n):
    temp = tempsensor.get_tempC()
t1 = time.time()
total = t1-t0
print(total/n)

t0 = time.time()
for i in range(n):
    temp = tempsensor.readC()
t1 = time.time()
total = t1-t0
print(total/n)

tempsensor.task.close()

