import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

i = 0
n = 1000
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_thrmcpl_chan(
        'Dev1/ai0',
        # name_to_assign_to_channel="", 
        # min_val=0.0,
        # max_val=100.0, 
        units=nidaqmx.constants.TemperatureUnits.DEG_F,
        thermocouple_type=nidaqmx.constants.ThermocoupleType.E,
        # cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE, 
        # cjc_val=20.0,
        # cjc_channel=""
        )
    while i<n:
        data = task.read(number_of_samples_per_channel=100)
        plt.scatter(i, np.mean(data), c='r')
        plt.pause(0.05)
        i += 1
    print(data)