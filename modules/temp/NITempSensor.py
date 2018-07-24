import nidaqmx
from nidaqmx import Task
import matplotlib.pyplot as plt
import numpy as np



# add NI sensors into the dict and the code will check if the devices in its keys.
# the values are the number of samples per test for average
devices_dict = {'USB-TC01': 1, 'PCIe-6321': 100}

def list_devices():
    ''' return a list of connected NI devices '''
    system = nidaqmx.system.System.local()
    # list all connected NI devices
    devices = []
    for device in system.devices:
        print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
            device.name, device.product_category, device.product_type))
        devices.append(device)
    return devices


def available_devices():
    ''' return a list of available devices in the NI_devices '''
    devices = list_devices()
    for device in devices:
        if device.product_type not in devices_dict.keys():
            devices.remove(device)
    return devices            


class TempSensor():
    def __init__(self, device, ai_channel, thrmpl_type):
        self.thrmcpl_chan = device.name + '/' + ai_channel
        self.thrmcpl_type = getattr(nidaqmx.constants.ThermocoupleType, thrmpl_type)
        self.cjc_source = nidaqmx.constants.CJCSource.BUILT_IN
        self.nsample =devices_dict.get(device.product_type, [])
        if not self.nsample:
            print('device is not found or not available.')
            return        
        

    def get_temp(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_thrmcpl_chan(
                self.thrmcpl_chan,
                # name_to_assign_to_channel="", 
                # min_val=0.0,
                # max_val=100.0, 
                # units=nidaqmx.constants.TemperatureUnits.DEG_C,
                thermocouple_type=self.thrmcpl_type,
                # cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE, 
                cjc_source=self.cjc_source, 
                # cjc_val=20.0,
                # cjc_channel=""
                )
            data = task.read(number_of_samples_per_channel=self.nsample)
            return np.mean(data)

############## test code below





if __name__ == "__main__":
    def get_temp(device, ai_channel, thrmpl_type):
        
        nsample = devices_dict.get(device.product_type, [])
        if not nsample:
            print('device is not found of not available.')
            return

        thrmcpl_chan = device.name + '/' + ai_channel
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_thrmcpl_chan(
                'Dev3/ai0',
                # name_to_assign_to_channel="", 
                # min_val=0.0,
                # max_val=100.0, 
                # units=nidaqmx.constants.TemperatureUnits.DEG_C,
                thermocouple_type=nidaqmx.constants.ThermocoupleType.J,
                # cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE, 
                cjc_source=nidaqmx.constants.CJCSource.BUILT_IN, 
                # cjc_val=20.0,
                # cjc_channel=""
                )
            data = task.read(number_of_samples_per_channel=nsample)
            return np.mean(data)
    class Device():
        def __init__(self):
            self.name = 'Dev3'
            self.product_type = 'USB-TC01'



    device = Device()
    ai_channel = 'ai0'
    thrmpl = 'J'
    temp = get_temp(device, ai_channel, thrmpl)
    print(temp)

    tempsensor = TempSensor(device, ai_channel, thrmpl)
    print(tempsensor.get_temp())

    def test():
        plt.ion()
        i = 0
        n = 1000
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_thrmcpl_chan(
                'Dev3/ai0',
                # name_to_assign_to_channel="", 
                # min_val=0.0,
                # max_val=100.0, 
                # units=nidaqmx.constants.TemperatureUnits.DEG_C,
                thermocouple_type=nidaqmx.constants.ThermocoupleType.J,
                # cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE, 
                cjc_source=nidaqmx.constants.CJCSource.BUILT_IN, 
                # cjc_val=20.0,
                # cjc_channel=""
                )
            while i<n:
                data = task.read(number_of_samples_per_channel=1)
                plt.scatter(i, np.mean(data), c='r')
                plt.pause(0.05)
                i += 1
            print(data)