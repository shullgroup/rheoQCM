'''
modules for GUI
'''

import sys
import os, subprocess
import inspect
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)

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
    logger.info(dir(module))
    subcls_list = inspect.getmembers(module, inspect.isclass)

    return {subcls[0]: subcls[0] for subcls in subcls_list}

    # [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == 'module']


def split_path(path):
    pass


def index_from_str(idx_str, chn_idx, join_segs=True):
    '''
    convert str to indices
    data_len: int. length of data
    chn_idx: index of chn
    join_segs: True, use the uint of all segments seperarted by []
               False, return indics in a list of lists [[seg1], [seg2], ...] (this is for multiple segments fitting for temperature expreiment)
    idx_str examples:
    '::' 
    '0'
    '1, 2, 4, 9' # this is not suggested
    '1:5' 
    '1:5:2' 
    '[1:5] [7:10]' # multiple segments, use [ ]
    '''
    idx = [] # for found indices
    if chn_idx == []:
        return idx

    if isinstance(idx_str, list):
        return idx_str

    if isinstance(idx_str, int):
        return [idx_str]

    # create a dummy data with index
    data = list(range(max(chn_idx)+1))
    logger.info(chn_idx) 
    logger.info(data) 
    try:
        # check if string contains [ ]
        segs = re.findall(r'\[([0-9\:][^]]*)\]', idx_str) # get [] as seg
        logger.info(segs) 
        if segs:
            for seg in segs:
                logger.info('multi') 
                logger.info(seg) 
                logger.info('data' +'[' + seg + ']') 
                logger.info(eval('data' +'[' + seg + ']')) 
                new_idx = eval('data' +'[' + seg + ']') 
                logger.info('type(new_idx) %s', type(new_idx))
                if join_segs: # True: combine
                    if isinstance(new_idx, int):
                        idx.append(new_idx)
                    elif isinstance(new_idx, list):
                        idx.extend(new_idx)
                else: # Fasle: keep size
                    if isinstance(new_idx, int):
                        idx.append([new_idx])
                    elif isinstance(new_idx, list):
                        idx.append(new_idx)
                
        else:
            logger.info('single') 
            logger.info('data' +'[' + idx_str + ']') 
            new_idx = eval('data' +'[' + idx_str + ']')
            if isinstance(new_idx, int):
                idx.append(new_idx)
            elif isinstance(new_idx, list):
                idx.extend(new_idx)
        
        # check the index with chn_idx
        if join_segs: # combine all
            logger.info('joined %s', sorted(list(set(idx) & set(chn_idx))))
            return sorted(list(set(idx) & set(chn_idx)))
        else: # keep separate
            # return thel list
            return [sorted(list(set(ind) & set(chn_idx))) for ind in idx]
    except Exception as err:
        logger.warning('exception in index_from_str')
        return idx


def sel_ind_dict(harms, sel_idx_dict, mode, marks):
    '''
    recalculate the indices in sel_idx_dict (from selector of main UI) by mode
    sel_idx_dict = {
        'harm': [index]
    }
    harms: list of harms for recalculating
    mode: 'all', 'marked', 'selpts', 'selidx', 'selharm'
    marks: df of marks in columns
    '''
    data_idx_dict = {}
    for harm in harms:
        if 'mark'+ harm in marks.columns:
            data_idx_dict[harm] = list(marks[marks['mark' + harm].notna()].index) # all the indices with data for each harm

    if mode == 'all':
        sel_idx_dict = data_idx_dict  
    if mode == 'marked':
        for harm in harms:
            logger.info(harm) 
            data_idx_dict[harm] = list(marks[marks['mark' + harm] == 1].index) # all the indices with data for each harm
        sel_idx_dict = data_idx_dict  
        logger.info(sel_idx_dict) 
            
    if mode == 'selpts':
        pass
    elif mode == 'selidx':
        idx_set = set()
        for idx in sel_idx_dict.values():
            idx_set |= set(idx)
        for harm in harms:
            sel_idx_dict[harm] = list(idx_set & set(data_idx_dict[harm]))
    elif mode == 'selharm':
        for harm in sel_idx_dict.keys():
            sel_idx_dict[harm] = data_idx_dict[harm]
    else:
        pass
    return sel_idx_dict


def idx_dict_to_harm_dict(sel_idx_dict):
    '''
    reform sel_idx_dict 
    {
        'harm': [index]
    }
    to sel_harm_dict
    {
        'index': [harm]
    }
    '''
    # get union of indecies 
    idx_set = set()
    for idxs in sel_idx_dict.values():
        idx_set |= set(idxs)
    idx_un = list(idx_set)
    logger.info('idx_un %s', idx_un) 

    sel_harm_dict = {}
    for idx in idx_un:
        sel_harm_dict[idx] = []
        for harm in sel_idx_dict.keys():
            if idx in sel_idx_dict[harm]:
                sel_harm_dict[idx].append(harm)
        
    return sel_harm_dict


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


############# MathModules ###########


def datarange(data):
    '''find the min and max of data'''
    if any(data):
        return [min(data), max(data)]
    else:
        return [None, None]


def num2str(A,precision=None):
    if isinstance(A, np.ndarray):
        if A.any() and not precision:
            return A.astype(str)
        elif A.any() and precision:
            for i in range(len(A)):
                A[i] = format(float(A[i]), '.'+str(precision)+'f').rstrip('0').rstrip('.')
                # A[i] = '{:.6g}'.format(float(A[i]))
            return A.astype(str)
    elif isinstance(A, float) or isinstance(A, int):
        if A and not precision:
            A = str(float(A))
            return A
        elif A and precision:
            A = format(float(A), '.'+str(precision)+'f').rstrip('0').rstrip('.')
            # A = '{:.6g}'.format(float(A))
            return A
    else: 
        return str(A)


def converter_startstop_to_centerspan(f1, f2):
    '''convert start/stop (f1/f2) to center/span (fc/fs)'''
    fc = (f1 + f2) / 2
    fs = f2 - f1
    return [fc, fs]


def converter_centerspan_to_startstop(fc, fs):
    '''convert center/span (fc/fs) to start/stop (f1/f2)'''
    f1 = fc - fs / 2
    f2 = fc + fs / 2
    return [f1, f2]



if __name__ == '__main__':

    idx_str = '[1:3] [5] [7:11]'
    idx_str = '[3:4]'
    chn_queue_list = list(range(13))
    print('---')
    print(index_from_str(idx_str, chn_queue_list, join_segs=True))
    print('---')
    print(index_from_str(idx_str, chn_queue_list, join_segs=False))