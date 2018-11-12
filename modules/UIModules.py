'''
modules for GUI
'''

import sys
import os, subprocess
import inspect
import re

def open_file(path):
    '''
    open the folder given by path
    if path is not given, open active path
    '''
    platform = system_check()
    if platform == "win32": # windows
        os.startfile(path)
    else: # mac and linux
        opener ="open" if platform == "darwin" else "xdg-open" 
        subprocess.call([opener, path]) # this opens a new window on Linux every time

def system_check():
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
    print(dir(module))
    subcls_list = inspect.getmembers(module, inspect.isclass)

    return {subcls[0]: subcls[0] for subcls in subcls_list}

    # [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == 'module']


def split_path(path):
    pass

def index_from_str(idx_str, chn_queue_list):
    '''
    convert str to indices
    data_len: int. length of data
    idx_str examples:
    '::' 
    '0'
    '1, 2, 4, 9' # this is not suggested
    '1:5' 
    '1:5:2' 
    '[1:5] [7, 8, 9, 10]' # multiple segments, use [ ]
    '''
    idx = [] # for found indices
    if chn_queue_list == []:
        return idx
    # create a dummy data with index
    data = list(range(max(chn_queue_list)))
    print(data)
    try:
        # check if string contains [ ]
        segs = re.findall(r'\[([0-9\:][^]]*)\]', idx_str) # get [] as seg
        print(segs)
        if segs:
            for seg in segs:
                print('multi')
                print(seg)
                print('data' +'[' + seg + ']')
                print(eval('data' +'[' + seg + ']'))
                new_idx = eval('data' +'[' + seg + ']')
                if isinstance(new_idx, int):
                    idx.append(new_idx)
                elif isinstance(new_idx, list):
                    idx.extend(new_idx)
        else:
            print('single')
            print('data' +'[' + idx_str + ']')
            print(eval('data' +'[' + idx_str + ']'))
            new_idx = eval('data' +'[' + idx_str + ']')
            if isinstance(new_idx, int):
                idx.append(new_idx)
            elif isinstance(new_idx, list):
                idx.extend(new_idx)
        

        return sorted(list(set(idx) & set(chn_queue_list)))

    except Exception as err:
        print(err)
        return idx


