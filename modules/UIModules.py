'''
modules for GUI
'''


def open_file(path):
    '''
    open the folder given by path
    if path is not given, open active path
    '''
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

def list_modules(module):
    ''' 
    return a dict of available temp modules stored in path 
    '''
    # # from package
    # from os import listdir
    # from os.path import isfile
        
    # modules = {f.replace('.py', ''): f.replace('.py', '') for f in listdir(path) if isfile and '__' not in f}
    # return modules

    # from subclass
    import inspect
    print(dir(module))
    subcls_list = inspect.getmembers(module, inspect.isclass)

    return {subcls[0]: subcls[0] for subcls in subcls_list}
    [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == 'module']
def split_path(path):
    pass

