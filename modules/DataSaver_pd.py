'''
module for saving data:
initiating file
appending scans
searching path
deleting dataset
reading dataset

Data format: HDF5
This module use a "dirty" way to store data in pandas.DataFrame
and save into HDF5 file with the power of pandas.HDFStore module

'''

# import h5py
import os
import datetime
import time # for test
import pandas as pd
import numpy as np

from UISettings import settings_init

class DataSaver:
    def __init__(self):
        self.path = None
        self.mode = None
        self.lasttestid = -1
        self.lastscanid = -1
        self.samp = self._mak_df(dftype='data')
        self.ref  = self._mak_df(dftype='data')
        self.raw  = self._mak_df(dftype='raw') # only store the latest test data.
        self.settings  = self._mak_df(dftype='settings')

    def _mak_df(self, dftype=None, data=None, nrows=None):
        '''
        initiate an empty df for storing the data by type
        type: data, raw,settings
        '''
        if dftype.lower() == 'data':
            columns=[
                'test_id', # indices of each timer
                't',
                'temp',
                'marks', # list default [0, 0, 0, 0, 0]
                'fstars',
            ]
        elif dftype.lower() == 'raw':
            columns=[
                # 'scan_id',  # indices of each single scan
                # 'test_id',  # indices of each timer
                # 'chn_name', # 'samp' or 'ref'
                # 'harm',     # harmonic
                # 't',
                # 'temp',
                'f',        # ndarray
                'G',        # ndarray
                'B',        # ndarray
            ]
        elif dftype.lower() == 'settings':
            columns=[
                'ver ',     # program version (str)
                'settings', # test settings from main ({})
                'samp_ref', # sample reference params {}
            ]
        else:
            columns = []

        # set the indices to a range() inorder to make a df with given rows with NANs
        if nrows is None:
            index = None
        else:
            index = range(nrows)   
        # make the df     
        df = pd.DataFrame(data, columns=columns, index=index)

        return df

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
            with pd.HDFStore(self.path) as hdf:
                # put empth dfs in to file
                hdf.put('settings', self.settings)
                hdf.put('samp', self.samp, format='table', data_columns=True)
                hdf.put('ref', self.ref)
                hdf.put('raw', self.raw, format='table', data_columns=True)

        elif self.mode == 'append': # append or modify
            # get data information
            with pd.HDFStore(self.path) as hdf:
                key_list = list(hdf.keys())
                print(key_list)
                
                # get data save to attributes
                for key in key_list:
                    setattr(self, key, hdf['/' + key]) # get dfs
                self.lasttestid = self.samp['test_id'][-1]
                self.lastscanid = self.raw['scan_id'][-1]


    def save_data(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None, fstars=[np.nan], marks=[0]):
        '''
        save raw data of ONE TEST to self.raw and save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: python datetime.datetime or strftime ?
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        fstars: dict of list of complex numbers {'samp': [a + bj, ...]}
        marks: [0]. by default all will be marked as 0
        '''

        # get rows 
        nr = len(harm_list)
        for i in range(1, settings_init['max_harmonic']+2, 2):
            if str(i) not in harm_list: # tested harmonic
                marks.insert(int((i-1)/2), np.nan)

        # append to the form by chn_name
        for chn_name in chn_names:
            # prepare data: change list to the size of harm_list by inserting nan to the empty harm
            for i in range(1, settings_init['max_harmonic']+2, 2):
                if str(i) not in harm_list: # tested harmonic
                    fstars[chn_name].insert(int((i-1)/2), np.nan)

            # up self.samp/ref first
            # create a df to append
            data_new = pd.DataFrame.from_dict({
                'test_id': [self.lasttestid + 1],  # add 1 
                't': [t],
                'temp': [temp],
                'marks': [marks], # list default [0, 0, 0, 0, 0]
                'fstars': [fstars[chn_name]],
            })
            # print(data_new)
            setattr(self, chn_name, getattr(self, chn_name).append(data_new, ignore_index=True))
            # self.samp.append(data_new, ignore_index=True)
            # print('getattr')
        # print(getattr(self, chn_name))

        # clear df self.raw (which is faster?)
        self.raw  = self._mak_df(dftype='raw', nrows=nr)
        # self.raw.drop(self.raw.index, inplace=True)
        print(self.raw)

        self.raw = pd.DataFrame(
            {
                # 'scan_id': range(self.lastscanid+1, self.lastscanid+1+nr),
                # 'test_id': [self.lasttestid+1] * nr,
                # 'chn_name': [chn_name] * nr,
                # 'harm': harm_list,
                # 't': [t] * nr,
                # 'temp': [temp] * nr,
                'f': [f[chn_name][harm] for harm in harm_list],
                'G': [G[chn_name][harm] for harm in harm_list],
                'B': [B[chn_name][harm] for harm in harm_list],
            }
        ) 
        # self.raw = self.raw.convert_objects()
        # print(f[chn_name]['1'])
        print(type(f[chn_name]['1']))
        print(type(f[chn_name]['1']))
        print(self.raw.dtypes)
        # return
        # update self.raw with df_raw_temp
        # this prevent the column difference between input cols and cols in df_raw_temp
        # print(df_raw_temp)
        # print(self.raw.update(df_raw_temp))
        # self.raw = self.raw.update(df_raw_temp)

        print(temp, type(temp))
        # self.raw['temp'].astype(float)
        # self.raw['scan_id'].astype(object)
        # self.raw['test_id'].astype(object)
        print(self.raw.head())

        # save self.raw to file
        t0 = time.time()
        with pd.HDFStore(self.path) as hdf:
            print(hdf)
            print(hdf.keys())
            # print(hdf['/raw'])
            # hdf.append('raw', self.raw, format='table')
            hdf.append('samp', self.samp)
        t1 = time.time()
        print(self.path)
        print('hdfstore time:', t1-t0)

        # update indices
        self.lasttestid += 1
        self.lastscanid += nr





if __name__ == '__main__':
    data_saver = DataSaver(r'.\test\test.h5', mode='append')
    data_saver.init_file()




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