""" 
This module contains functions to find NI devices
and return a list of devices (class NI device)
eg.: device.name ('Dev1'), device.product_category ('ProductCategory.USBDAQ'), device.product_type ('USB-TC01') ...
"""

import nidaqmx


# add NI sensors into the dict and the code will check if the devices in its keys.
# the values are the number of samples per test for average
# devices_dict = {'USB-TC01': 1, 'PCIe-6321': 100}

def list_devices():
    ''' return a list of connected NI devices '''
    system = nidaqmx.system.System.local()
    # list all connected NI devices
    devices = []
    try: # TODO find way to test if device connected
        for device in system.devices:
            print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
                device.name, device.product_category, device.product_type))
            devices.append(device)
    except:
        dcvices = []
    return devices


def available_devices(devices_dict):
    ''' return a list of available devices in the NI_devices '''
    devices = list_devices()
    for device in devices:
        if device.product_type not in devices_dict.keys():
            devices.remove(device)
    return devices            

def dict_available_devs(devices_dict):
    '''
    return a dict of avaiable devices for UI
    '''
    devices = available_devices(devices_dict)

    # return {device.product_type: '{} ({}/ai0)'.format(device.product_type, device.name) for device in devices}
    return {device.product_type: device.product_type for device in devices}

def device_info(devtype):
    '''
    return a single device 
    '''
    print('devtype', devtype) #testprint
    devices = list_devices()
    for device in devices:
        if device.product_type != devtype:
            devices.remove(device)
    return devices[0]
