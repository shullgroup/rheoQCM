import nidaqmx
# from nidaqmx import Task
import numpy as np

import logging
logger = logging.getLogger(__name__)

# another way to let the main code find the temp class in this file is just put a dict here
class_list = {
    'NITempSensor': 'NITempSensor', # name to access / name to display
}

class NITempSensor():
    def __init__(self, device, device_params, thrmcpl_type):
        '''
        devices (class NI device): 
            device.name ('Dev1'), 
            device.product_category ('ProductCategory.USBDAQ'), 
            device.product_type ('USB-TC01') 
            ...
        device_params={
            'nsmples': # of points for average,
            'thrmcpl_chan: thermocouple channel,
            'cjc_source': channel for cjc,
        }
        '''

        self.thrmcpl_chan = device.name + '/' + device_params['thrmcpl_chan']
        self.thrmcpl_type = getattr(nidaqmx.constants.ThermocoupleType, thrmcpl_type)
        if device_params['cjc_source']: 
            self.cjc_source = getattr(nidaqmx.constants.CJCSource, device_params['cjc_source'])
        else:
            self.cjc_source = ''   # assign '' if no matching

        self.nsamples = device_params['nsamples']

        if not self.nsamples:
            print('device is not found or not available.')
            return

        # init task
        # self.task = nidaqmx.Task()
        # self.task.ai_channels.add_ai_thrmcpl_chan(
        #     self.thrmcpl_chan,
        #     # name_to_assign_to_channel="", 
        #     # min_val=0.0,
        #     # max_val=100.0, 
        #     # units=nidaqmx.constants.TemperatureUnits.DEG_C,
        #     thermocouple_type=self.thrmcpl_type,
        #     # cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE, 
        #     cjc_source=self.cjc_source, 
        #     # cjc_val=20.0,
        #     # cjc_channel=""
        # )


    # def readC(self):
    #     data = self.task.read(number_of_samples_per_channel=self.nsamples)
    #     return np.mean(data)

    def get_tempC(self):
        if self.cjc_source:
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
                data = task.read(number_of_samples_per_channel=self.nsamples)
                return np.mean(data)
        else:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_thrmcpl_chan(
                    self.thrmcpl_chan,
                    thermocouple_type=self.thrmcpl_type,
                    )
                data = task.read(number_of_samples_per_channel=self.nsamples)
                return np.mean(data)

############## test code below





if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
    logger.info(temp) 

    tempsensor = TempSensor(device, ai_channel, thrmpl)
    logger.info(tempsensor.get_temp()) 

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
            logger.info(data) 