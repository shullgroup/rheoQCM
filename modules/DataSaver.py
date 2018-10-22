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

.h5 --- raw --- samp ---0
     |       |        |-1
     |       |        |-2
     |       |        --...
     |       |
     |       -- ref  ---0
     |                |-1
     |                |-2
     |                --...
     |
     -- data-|-samp (json)
             |-ref  (json)
             --...
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
        self.samp = self._make_df() # df for data form samp chn
        self.ref  = self._make_df() # df for data from ref chn
        self.samp_ref = self._make_df() # df for samp chn reference
        self.ref_ref = self._make_df() # df for ref chn reference
        self.raw  = {} # raw data from last queue
        self.settings  = settings # UI settings of test
        self.exp_ref  = self._make_exp_ref() # experiment reference setup in dict
        self.ver  = ver # version information
        self._keys = ['samp', 'ref'] # raw data groups

    def _make_df(self):
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
            'freqs',
            'gamms',
        ])

    def _make_exp_ref(self):
        '''
        initiact an dict for storing the experiment reference information
        '''
        return {
            't0': None,
            't0_shifted': None,
            'samp':{
                'f0': [], # list for each harmonic
                'g0': [], # list for each harmonic
            },
            'ref':{
                'f0': [], # list for each harmonic
                'g0': [], # list for each harmonic
            },
            # str show the source used as reference
            'samp_ref': '', # 'samp', 'ref', 'ext', 'none'
            'ref_ref': '',  # 'samp', 'ref', 'ext', 'none'
        } # experiment reference setup in dict

    def init_file(self, path=None):
        '''
        initiate hdf5 file for data saving
        '''
        if not path: # save data in the default temp path
            self.path = os.path.join(settings_init['unsaved_path'], datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.h5') # use current time as filename
        else:
            self.path = path

        # create groups for raw data
        # dt = h5py.special_dtype(vlen=str)
        with h5py.File(self.path, 'w') as fh:
            g_data = fh.create_group('data')
            g_raw = fh.create_group('raw')
        
        # save settings information
        self._save_ver(ver=self.ver)

        self.queue_list = []

    def load_file(self, path=None):
        '''
        load data information from exist hdf5 file
        '''
        if not path: # if path is not available
            self.init_file() # create a new file

            return
        self.path = path

        # get data information
        with h5py.File(self.path, 'r') as fh:
            key_list = list(fh.keys())
            print(key_list)
            
            # check if the file is data file with right format
            # if all groups/attrs in files

            # get data save to attributes
            
            self.settings = json.loads(fh['settings'][()])
            self.exp_ref = json.loads(fh['exp_ref'][()])
            self.ver = fh.attrs['ver']
            print(self.ver)
            print(self.exp_ref)
            # get queue_list
            self.queue_list = list(fh['raw/samp'].keys())

            print(fh['data/samp'][()])
            for key in self._keys:
                setattr(self, key, pd.read_json(fh['data/' + key][()]).sort_values(by=['queue_id'])) # df for data form samp/ref chn
                setattr(self, key, pd.read_json(fh['data/' + key + '_ref'][()]).sort_values(by=['queue_id'])) # df for data form samp_ref/ref_ref chn

            self.raw  = {} # raw data from last queue

    def dynamic_save(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None, freqs=[np.nan], gamms=[np.nan], marks=[0]):
        '''
        save raw data of ONE QUEUE to self.raw and save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: dict of str
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        freqs: dicts of delta freq freqs[chn_name]
        gamms: dicts of delta gamma gamms[chn_name]
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
                    freqs[chn_name].insert(int((i-1)/2), np.nan)
                    gamms[chn_name].insert(int((i-1)/2), np.nan)

            # up self.samp/ref first
            # create a df to append
            data_new = pd.DataFrame.from_dict({
                'queue_id': [max(self.queue_list)],  
                't': [t[chn_name]],
                'temp': [temp[chn_name]],
                'marks': [marks], # list default [0, 0, 0, 0, 0]
                'freqs': [freqs[chn_name]],
                'gamms': [gamms[chn_name]],
            })
            # print(data_new)
            setattr(self, chn_name, getattr(self, chn_name).append(data_new, ignore_index=True))
            print(self.samp.tail())


        # save raw data to file by chn_names
        self._save_raw(chn_names, harm_list, t=t, temp=temp, f=f, G=G, B=B)


    def _save_raw(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None):
        '''
        save raw data of ONE QUEUE to save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: dict of strftime 
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        marks: [0]. by default all will be marked as 0
        '''

        t0 = time.time()
        with h5py.File(self.path, 'a') as fh:
            for chn_name in chn_names:
                # creat group for test
                g_queue = fh.create_group('raw/' + chn_name + '/' + str(max(self.queue_list)))
                # add t, temp to attrs
                # store t as string
                g_queue.attrs['t'] = t[chn_name]
                if temp[chn_name]:
                    print(temp)
                    g_queue.attrs['temp'] = temp[chn_name]

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
                # print(fh['data/' + key])
                if key in fh['data']:
                    del fh['data/' + key]
                    del fh['data/' + key + '_ref']
                print(json.dumps(getattr(self, key).to_dict()))
                # fh['data/' + key] = json.dumps(getattr(self, key).to_dict())        
                fh.create_dataset('data/' + key, data=getattr(self, key).to_json(), dtype=h5py.special_dtype(vlen=str))  
                fh.create_dataset('data/' + key + '_ref', data=getattr(self, key).to_json(), dtype=h5py.special_dtype(vlen=str))  

                # , dtype=h5py.special_dtype(vlen=str)

    def save_settings(self, settings={}, exp_ref={}):
        '''
        save settings (dict) to file
        '''
        if not settings:
            settings = self.settings
        if not settings:
            exp_ref = self.exp_ref
        with h5py.File(self.path, 'a') as fh:
            fh['settings'] = json.dumps(settings)
            fh['exp_ref'] = json.dumps(exp_ref)

    def _save_ver(self, ver=''):
        '''
        save ver (str) to file
        '''
        with h5py.File(self.path, 'a') as fh:
            fh.attrs['ver'] = self.ver

    def get_npts(self):
        '''
        get number of total points
        '''
        return len(self.queue_list)



    ####################################################
    ##              data convert functions            ##
    ##  They can also be used external for accessing, ## 
    ##  converting data.
    ####################################################

    def data_exporter(self, fileName):
        '''
        this function export the self.data.samp and ...ref 
        in the ext form
        fileName: string of full path with file name
        '''
        # TODO add exp_ref
        
        # get df of samp and ref channel
        df_samp = self.reshape_data_df('samp', mark=False, dropnanrow=True)
        df_ref = self.reshape_data_df('ref', mark=False, dropnanrow=True)
        df_samp_ref = self.reshape_data_df('samp_ref', mark=False, dropnanrow=True)
        df_ref_ref = self.reshape_data_df('ref_ref', mark=False, dropnanrow=True)

         # get ext
        name, ext = os.path.splitext(fileName)

        # export by ext
        if ext.lower() == '.xlsx':
            with pd.ExcelWriter(fileName) as writer:
                df_samp.to_excel(writer, sheet_name='samp_channel')
                df_ref.to_excel(writer, sheet_name='ref_channel')
                df_samp_ref.to_excel(writer, sheet_name='sample_reference')
                df_ref_ref.to_excel(writer, sheet_name='ref_reference')

        elif ext.lower() == '.csv':
            # add chn_name to samp and ref df
            # and append ref to samp
            df_samp.assign(chn='samp').append(df_ref.assign(chn='ref')).to_csv(fileName)

        elif ext.lower() == '.json':
            with open(fileName, 'w') as f:
                ## lines with indent (this will make the file larger)
                # line = json.dumps({'samp': self.samp.to_dict(), 'ref': self.ref.to_dict()}, indent=4) + "\n"
                # f.write(line)

                ## without separate by lines (smaller file size, but harder to)
                json.dump({
                    'samp': self.samp.to_dict(), 
                    'ref': self.ref.to_dict(),
                    'samp_ref': self.samp_ref.to_dict(), 
                    'ref_ref': self.ref_ref.to_dict(),
                    }, 
                    f
                )
                # json.dump(data, f)

            
    def reshape_data_df(self, chn_name, mark=False, dropnanrow=True, dropnancolumn=True):
        '''
        reshape and tidy data df (samp and ref) for exporting
        '''
        cols = ['freqs', 'gamms']
        df = getattr(self, chn_name).copy()

        # convert t column to datetime object
        df['t'] = self.get_t_s(chn_name)
        print(df.t)

        for col in cols:
            df = df.assign(**self.get_list_column_to_columns(chn_name, col, mark=mark)) # split columns: freqs and gamms
            df = df.drop(columns=col) # drop columns: freqs and gamms

        print(df.head())

        # drop columns with all 
        if dropnancolumn == True:
            df = df.dropna(axis='columns', how='all')
        print(df.head())

        if dropnanrow == True: # rows with marks only
            # select rows with marks
            df = df[self.rows_with_marks(chn_name)][:]
            df = df.drop(columns='marks') # drop marks column
        else:
            print('there is no marked data.\n no data will be deleted.')


        if 'marks' in df.columns:
            # split marks column
            df = df.assign(**self.get_list_column_to_columns(chn_name, 'marks', mark=mark))
            # finally drop the marks column
            df = df.drop(columns='marks') # drop marks column

        return df

    def get_t_s_marked_rows(self, chn_name, dropnanrows=False):
        '''
        return rows with marks of df from self.get_t_s
        '''
        if dropnanrows == True:
            return self.get_t_s(chn_name)[self.rows_with_marks(chn_name)]
        else:
            return self.get_t_s(chn_name)

    def get_t_s(self, chn_name):
        '''
        get time (t) in sec as pd.series
        t: pd.series of str
        '''
        t = getattr(self, chn_name)['t'].copy()
        # convert t column to datetime object
        t = pd.to_datetime(t)
        # convert t to delta t in seconds
        if t.shape[0] == 0:
            return t
        else:
            t = t -  self.get_t_ref() # delta t to reference (t0)
            t = t.dt.total_seconds() # convert to second
            return t

    def get_t_ref(self):
        '''
        get reference time and shift it by delt
        '''
        # find reference t from dict exp_ref first
        if 't0' in self.exp_ref.keys() and self.exp_ref.get('t0', None): # t0 exist and != None or 0
            if self.exp_ref.get('t0_shifted', None):
                t0 = datetime.datetime.strptime(self.exp_ref.get('t0_shifted'), settings_init['time_str_format']) # used shifted t0
            else:
                t0 = datetime.datetime.strptime(self.exp_ref.get('t0'), settings_init['time_str_format']) # use t0
        else: # no t0 saved in self.exp_ref
            # find t0 in self.settings
            t0 = self.settings.get('dateTimeEdit_reftime', None)
            print('t0', t0)
            if not t0:
                if self.samp.shape[0]> 0: # use the first queque time
                    t0 = datetime.datetime.strptime(self.samp['t'][0], settings_init['time_str_format'])
                else:
                    t0 = None
            else:
                t0 = datetime.datetime.strptime(t0, settings_init['time_str_format'])
        
        return t0

    def calc_fg_ref(self, chn_name, mark=False):
        '''
        calculate reference (self.samp_ref, self.ref_ref) of f (f0) and g (g0) and save them in self.exp_ref['f0'] and ['g0']
        '''
        if getattr(self, chn_name + '_ref').shape[0] == 0:
            print('no reference data was selected for {} channel.').format(chn_name)
        else: # there is reference data saved
            # calculate f0 and g0 
            for key, col in zip(['f0', 'g0'], ['freqs', 'gammas']):
                df = self.get_list_column_to_columns_marked_rows(chn_name + '_ref', col, mark=True, dropnanrow=False)

                self.exp_ref[chn_name][key] = df.mean().values.tolist() 
        
    def get_fg_ref(self, chn_name, harm=[]):
        '''
        get reference of f or g from self.exp_ref
        chn_name: 'samp' or 'ref'
        return a dict 
        {'f0': [f0_1, f0_3, ...], 
         'g0': [g0_1, g0_3, ...]}
        '''
        if not harm: # no harmonic is given
            return self.exp_ref[chn_name]

    def copy_to_ref(self, df, chn_name):
        '''
        copy df to self.[chn_name + '_ref'] as reference
        df should be from another file, self.samp or self.ref
        ''' 
        df = self.reset_marks(df) # remove marks
        setattr(self, chn_name + '_ref', df)


    def get_list_column_to_columns_marked_rows(self, chn_name, col, mark=False, dropnanrow=False):
        '''        
        return rows with marks of df from self.get_list_column_to_columns
        '''
        cols_df = self.get_list_column_to_columns(chn_name, col, mark=mark)
        if dropnanrow == True:
            return cols_df[self.rows_with_marks()][:]
        else:
            return cols_df
 
    def get_list_column_to_columns(self, chn_name, col, mark=False):
        '''
        get a df of marks, freqs or gamms by open the columns with list to colums by harmonics
        chn_name: str of channel name ('sam', 'ref')
        col: str of column name ('freqs' or gamms')
        return: df with columns = ['1', '3', '5', '7', '9]
        '''
        

        if mark == False:
            s = getattr(self, chn_name)[col].copy()
            return pd.DataFrame(s.values.tolist(), s.index).rename(columns=lambda x: col[:-1] + str(x * 2 + 1))
        else:
            s = getattr(self, chn_name)[col].copy()
            m = getattr(self, chn_name)['marks'].copy()
            idx = s.index
            # convert s and m to ndarray
            arr_s = np.array(s.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            arr_m = np.array(m.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            
            # replace None with np.nan
            arr_s = arr_s * arr_m # leave values where only marks == 1
            # replace unmarked (marks == 0) with np.nan
            arr_s[arr_s == 0] = np.nan

            return pd.DataFrame(data=arr_s, index=idx).rename(columns=lambda x: col[:-1] + str(x * 2 + 1))

    def rows_with_marks(self, chan_name):
        '''
        return list of booleans of rows with marked harmonics
        if no marked rows, return all
        '''
        marked_rows = getattr(self, chan_name).marks.apply(lambda x: any(x))
        if marked_rows.any(): # there are marked rows
            print('There are marked rows')
            return marked_rows
        else: # no amrked rows, return all
            print('There is no marked row.\nReturn all')
            return ~marked_rows

    def reset_marks(self, df):
        ''' 
        rest marks column in df. 
        set 1 in list element to 0
        '''
        df_new = df.copy()
        df_new.marks = df.marks.apply(lambda x: [0 if mark == 1 else mark for mark in x])
        return df_new



######## functions for unit convertion #################


    def time_s_to_unit(t, unit=None):
        '''
        convert time from in second to given unit
        input:
            t: scaler or array or pd.series of t as float
            unit: None, or str ('s', 'm', 'h', 'd')
        return:
            same size as input
        '''

        factors = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return factors[unit] * t

