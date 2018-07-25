'''
modules for GUI
'''


def open_file(path):
    import os, subprocess
    platform = system_check()
    if platform == "win32": # windows
        os.startfile(path)
    else: # mac and linux
        opener ="open" if platform == "darwin" else "xdg-open" 
        subprocess.call([opener, path]) # this opens a new window on Linux every time

def system_check():
    import sys
    return sys.platform

def closest_spanctr_step(l, n):
    if n >= 1:
        return min(l, key=lambda x:abs(x-n))
    elif n < 1:
        return min(l, key=lambda x:abs(x-1/n))

def list_modules(path):
    from os import listdir
    from os.path import isfile
    modules = [f for f in listdir(path) if isfile and f != '__init__.py']
    return modules

def split_path(path):
    pass

def temp_by_unit(data, unit):
    '''
data: double or ndarray
unit: str. C for celsius, K for Kelvin and F for fahrenheit
'''
if unit == 'C':
    pass # temp data is saved as C
elif unit == 'K': # convert to K
    return data + 273.15 
elif unit == 'F': # convert to F
    return data * 9 / 5 + 32

