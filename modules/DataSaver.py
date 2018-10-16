'''
module for saving data:
initiating file
appending scans
searching path
deleting dataset
reading dataset

Data format: HDF5
This module use a h5py library to store the data in group and dataset
You can browse the data with hdf5 UI software e.g. HDFCompass if you
don't want to extract the data with code

'''

# import h5py
import os
import datetime
import time # for test
import pandas as pd
import numpy as np
import h5py
import json

from UISettings import settings_init

class DataSaver:
    def __init__(self, ver='', settings={}):
        self.path = None
        self.mode = None
        self.queue_list = []
        # self.lastscanid = -1
        self.samp = self._mak_df() # df for data form samp chn
        self.ref  = self._mak_df() # df for data from ref chn
        self.raw  = {} # raw data from last queue
        self.settings  = settings # UI settings of test
        self.ver  = ver # version information
        self._keys = ['samp', 'ref'] # raw data groups

    def _mak_df(self):
        '''
        initiate an empty df for storing the data by type
        type: samp, ref
        '''
        # make an empty df
        return pd.DataFrame(columns=[
            'queue_id', # indices of each timer
            't',
            'temp',
            'marks', # list default [0, 0, 0, 0, 0]
            'delfs',
            'delgs'
        ])


    def init_file(self, path=None, mode='new'):
        '''
        program ver
        '''
        if not path: # save data in the default temp path
            self.path = os.path.join(settings_init['unsaved_path'], datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.h5') # use current time as filename
        else:
            self.path = path
        self.mode = mode.lower()

        if self.mode == 'new':
            # save settings information
            self.save_ver(ver=self.ver)
            # create groups for raw data
            with h5py.File(self.path, 'a') as fh:
                for key in self._keys:
                    fh.create_group(key)
                    setattr(self, key, self._mak_df())

                self.queue_list = []

        elif self.mode == 'append': # append or modify
            # get data information
            with h5py.File(self.path, 'r') as fh:
                key_list = list(fh.keys())
                print(key_list)
                
                # check if the file is data file with right format
                # if all groups/attrs in files

                # get data save to attributes
                
                self.settings = json.loads(fh.attrs['settings'])
                self.settings = json.loads(fh.attrs['ver'])

                for key in self._keys: # self.samp, self.ref
                    setattr(self, key, fh[key]) # get dfs

                # get queue_list
                self.queue_list = list(fh['samp'].keys())
                # self.queue_list = self.samp['test_id'][-1]
                # self.lastscanid = self.raw['scan_id'][-1]


    def dynamic_save(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None, delfs=[np.nan], delgs=[np.nan], marks=[0]):
        '''
        save raw data of ONE QUEUE to self.raw and save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: str
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        delfs: dicts of delta freq delfs[chn_name]
        delgs: dicts of delta gamma delgs[chn_name]
        marks: [0]. by default all will be marked as 0
        '''

        # add current queue id to queue_list as max(queue_list) + 1
        if not self.queue_list:
            self.queue_list.append(0)
        else:
            self.queue_list.append(max(self.queue_list) + 1)

        for i in range(1, settings_init['max_harmonic']+2, 2):
            if str(i) not in harm_list: # tested harmonic
                marks.insert(int((i-1)/2), np.nan)

        # append to the form by chn_name
        for chn_name in chn_names:
            # prepare data: change list to the size of harm_list by inserting nan to the empty harm
            for i in range(1, settings_init['max_harmonic']+2, 2):
                if str(i) not in harm_list: # tested harmonic
                    delfs[chn_name].insert(int((i-1)/2), np.nan)
                    delgs[chn_name].insert(int((i-1)/2), np.nan)

            # up self.samp/ref first
            # create a df to append
            data_new = pd.DataFrame.from_dict({
                'queue_id': [max(self.queue_list)],  
                't': [t],
                'temp': [temp],
                'marks': [marks], # list default [0, 0, 0, 0, 0]
                'delfs': [delfs[chn_name]],
                'delgs': [delgs[chn_name]],
            })
            # print(data_new)
            setattr(self, chn_name, getattr(self, chn_name).append(data_new, ignore_index=True))
            print(self.samp.tail())


        # save raw data to file by chn_names
        self.save_raw(chn_names, harm_list, t=t, temp=temp, f=f, G=G, B=B)


    def save_raw(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None):
        '''
        save raw data of ONE QUEUE to save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: python datetime.datetime or strftime ?
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        marks: [0]. by default all will be marked as 0
        '''

        t0 = time.time()
        with h5py.File(self.path, 'a') as fh:
            for chn_name in chn_names:
                # creat group for test
                g_queue = fh.create_group(chn_name + '/' + str(max(self.queue_list)))
                # add t, temp to attrs
                # store t as string
                g_queue.attrs['t'] = t
                if temp:
                    print(temp)
                    g_queue.attrs['temp'] = temp

                for harm in harm_list:
                    # create data_set for f, G, B of the harm
                    g_queue.create_dataset(harm, data=np.stack((f[chn_name][harm], G[chn_name][harm], B[chn_name][harm]), axis=0))
        t1 = time.time()
        print(t1 - t0)

    def save_data(self):
        '''
        save samp (df), ref (df) to h5 file serializing with json
        '''
        with h5py.File(self.path, 'a') as fh:
            for key in self._keys:
                print(key)
                print(getattr(self, key).to_dict())
                fh.attrs[key] = json.dumps(getattr(self, key).to_dict())        


    def save_settings(self, settings={}):
        '''
        save settings (dict) to file
        '''
        if not settings:
            settings = self.settings
        with h5py.File(self.path, 'a') as fh:
            fh.attrs['settings'] = json.dumps(settings)

    def save_ver(self, ver=''):
        '''
        save ver (str) to file
        '''
        with h5py.File(self.path, 'a') as fh:
            fh.attrs['ver'] = self.ver




# exit(0)
# n = 10
# ln = 400
# fn = r'.\test\h5_test.h5'
# try:
#     os.remove(fn)
#     print('del')
# except:
#     pass
# data = {'samp': {}, 'ref':{}}
# # harm_data = {}

# print(data)
# # with h5py.File(fn, 'w') as fh:
# #     samp = fh.create_group('samp')
# #     ref = fh.create_group('ref')

# with h5py.File(fn, 'a') as fh:
#     for key in ['samp', 'ref']:
#         key = fh.create_group(key)
#         for i in range(n):
#             print(i)
#             t0 = time.time()
#             # idx = fh.create_group(key + '/' + str(i))
#             idx = key.create_group(str(i))
#             # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#             idx.create_dataset('t', data=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#             idx.create_dataset('temp', data=24.5)
#             # data['t'] = time.time()
#             for harm in [1, 3, 5, 7, 9]:
#                 hidx = idx.create_group('h' + str(harm))
#                 hidx.create_dataset('f', data=np.random.rand(ln))
#                 hidx.create_dataset('G', data=np.random.rand(ln))
#                 hidx.create_dataset('P', data=np.random.rand(ln))
#                 # data['h' + str(harm)]['fstar'] = str((np.random.rand(ln) + 1j*(np.random.rand(ln))).tolist()) # size larger
#             t1 = time.time()
#             print(t1 - t0)

# # show and delete
# with h5py.File(fn, 'a') as fh:
#     for k in fh.keys():
#         print(k)
#     fh.visit(lambda n: print(n))
#     del fh['samp']
#     del fh['ref/9/t']