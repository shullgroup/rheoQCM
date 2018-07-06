from module_temp import TempSensor
import time
class Device():
    def __init__(self):
        self.name = 'Dev3'
        self.product_type = 'USB-TC01'

device = Device()
ai_channel = 'ai0'
thrmpl = 'J'

tempsensor = TempSensor(device, ai_channel, thrmpl)

t0 = time.time()
for i in range(1):
    tempsensor.get_temp()
t1 = time.time()

total = t1-t0
print(total)
print(tempsensor.get_temp())