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
don't want to extract the data with code.
dictionaries and dataframes are converted to json and are saved as text in the file

.h5 --- raw --- samp ---0---1(harmonic)---column 1: frequency   (f)
     |       |        |   |             |-column 2: conductance (G)
     |       |        |   |             --column 3: susceptance (B)
     |       |        |   |-3
     |       |        |   |-5
     |       |        |   --...
     |       |        |-1
     |       |        |-2
     |       |        --...
     |       |
     |       -- ref  ---0
     |                |-1
     |                |-2
     |                --...
     |
     |- data-|-samp (json)
     |       |-ref  (json)
     |       --...
     |
     |- prop-|-samp--<e.g. 353_3 (named by solving combination and reference harmonic)> (json)
     |       |     |
     |       |     --...
     |       |     
     |       --ref--...
     |
     |-exp_ref       (json) # reference setting information
     |
     |-settings      (json) # UI settings (it can be loaded to set the UI)
     |
     --settings_init (json) # maximum harmonic and time string format for the collected data
'''

import os
import datetime
import time # for test
import pandas as pd
import numpy as np
import h5py
import json
import openpyxl
import csv


class DataSaver:
    def __init__(self, ver='', settings_init={'max_harmonic': None, 'time_str_format': None}):
        '''
        initial values are for initialize the module outside of UI
        '''

        self._chn_keys = ['samp', 'ref'] # raw data groups
        self._ref_keys = {'fs': 'f0', 'gs': 'g0'} # corresponding keys storing the reference
        self.ver  = ver # version information
        self.settings_init = {
            'max_harmonic': settings_init['max_harmonic'],
            'time_str_format': settings_init['time_str_format'],
        }

        self._init_attrs()


    def _init_attrs(self):
        '''
        attributes needs to be initiated with new file
        '''
        self.mode = ''  # mode of datasaver 'init': new file; 'load': append/load file
        self.path = ''
        self.saveflg = True # flag to show if modified data has been saved to file
        self.refflg = {chn_name: False for chn_name in self._chn_keys} # flag if the reference has been set
        self.queue_list = []
        # following attributes will be save in file
        self.settings = {}
        self.samp = self._make_df() # df for data form samp chn
        self.ref  = self._make_df() # df for data from ref chn
        self.samp_ref = self._make_df() # df for samp chn reference
        self.ref_ref = self._make_df() # df for ref chn reference
        self.raw  = {} # raw data from last queue
        self.exp_ref  = self._make_exp_ref() # experiment reference setup in dict
        self.samp_prop = {} # a dict for calculated mechanical results keys: '131'... values: pd.dataframe
        self.ref_prop = {}
        

    def _make_df(self):
        '''
        initiate an empty df for storing the data by type
        '''
        # make an empty df
        return pd.DataFrame(columns=[
            'queue_id', # indices of each timer
            't',
            'temp',
            'marks', # list default [0, 0, 0, 0, 0]
            'fs',
            'gs',
        ])


    def _make_exp_ref(self):
        '''
        initiact an dict for storing the experiment reference information
        '''
        return {
            't0': None,
            't0_shifted': None,
            'samp':{
                'f0': self.nan_harm_list(), # list for each harmonic
                'g0': self.nan_harm_list(), # list for each harmonic
            },
            'ref':{
                'f0': self.nan_harm_list(), # list for each harmonic
                'g0': self.nan_harm_list(), # list for each harmonic
            },
            # str show the source used as reference
            'samp_ref': ['samp', [0,]],  # default
            'ref_ref': ['ref', [0,]],    # default
            
            # structure: (source, [indices]) 
            #   source: 'samp', 'ref', 'ext', 'none'
            #       'samp': use data from samp chan as reference
            #       'ref': use data from ref chan as reference
            #       'ext': use data from external file as reference
            #       'none': no reference
            #   indeces:
            #   if len == 1 (e.g.: [0,]), is a single point 
            #   if [] or [None], reference point by point
        } # experiment reference setup in dict


    def update_mech_df_shape(self, chn_name, nhcalc, rh):
        '''
        initiate an empty df for storing the mechanic data in self.mech with nhcalc_rh as a key
        if there is a key with the same name, check and add missed rows (queue_id)
        nhcalc: str '133'
        rh: in, reference harmonit
        the df will be saved/updated as nhcalc_str(rh)
        return the updated df
        '''
        # data_keys = ['queue_id', 't', 'temp', 'marks', 'delfs', 'delgs',]
        data_keys = ['queue_id']

        # column names with single value for prop df
        mech_keys_single = [
            'drho',
            'drho_err',
            'grho_rh',
            'grho_rh_err',
            'phi',
            'phi_err',
            'dlam_rh',
            'lamrho',
            'delrho',
            'delf_delfsn',
            'rh',
        ]

        # column names with multiple value for prop df
        mech_keys_multiple = [
            'delf_exps',
            'delf_calcs', # n
            'delg_exps',
            'delg_calcs', # n
            'delg_delfsn_exps', # n
            'delg_delfsn_calcs', # n
            'rd_exps', # n
            'rd_calcs', # n
        ]

        mech_key = self.get_mech_key(nhcalc, rh)
        print(mech_key) #testprint

        if mech_key in getattr(self, chn_name + '_prop').keys():
            df_mech = getattr(self, chn_name + '_prop')[mech_key].copy()
            print(df_mech) #testprint
            print(df_mech.columns) #testprint
            mech_queue_id = df_mech['queue_id']
            data_queue_id = getattr(self, chn_name)['queue_id']
            print(mech_queue_id) #testprint
            print(data_queue_id) #testprint
            # check if queue_id is the same as self.chn_name
            if mech_queue_id.equals(data_queue_id):
                return df_mech
            else:
                # delete the extra queue_id
                df_mech = df_mech[df_mech['queue_id'].isin(set(mech_queue_id) & set(data_queue_id))]
                # add the missed queue_id, this will leave other columns as NA
                # df_mech = df_mech.append(pd.DataFrame.from_dict(dict(queue_id=list(set(data_queue_id) - set(mech_queue_id)))), ignore_index = True)
                # print(df_mech) #testprint
                # print(data_queue_id[data_queue_id.isin(set(data_queue_id) - set(mech_queue_id))].to_frame()) #testprint
                df_mech = df_mech.merge(data_queue_id[data_queue_id.isin(set(data_queue_id) - set(mech_queue_id))].to_frame(), how='outer')
                print(df_mech) #testprint
                # replace na with self.nan_harm_list
                for col in mech_keys_single:
                    df_mech[col] = df_mech[col].apply(lambda x: self.nan_harm_list() if isinstance(x, list) or pd.isnull(x) else x) # add list of nan to all null
                for col in mech_keys_multiple:
                    df_mech[col] = df_mech[col].apply(lambda x: self.nan_harm_list() if isinstance(x, list) or pd.isnull(x) else x) # add list of nan to all null
                print(df_mech) # testprint
        else: # not exist, make a new dataframe
            df_mech = pd.DataFrame(columns=data_keys+mech_keys_single+mech_keys_multiple)

            # set values
            nrows = len(getattr(self, chn_name)['queue_id'])
            nan_list = [self.nan_harm_list()] * nrows 
            df_mech['queue_id'] = getattr(self, chn_name)['queue_id']
            # df_mech['t'] = getattr(self, chn_name)['t']
            # df_mech['temp'] = getattr(self, chn_name)['temp']
            for df_key in mech_keys_single:
                # df_mech[df_key] = np.nan
                df_mech[df_key] = nan_list
            for df_key in mech_keys_multiple:
                df_mech[df_key] = nan_list

        print(df_mech) #testprint

        # set it to class
        self.update_mech_df_in_prop(chn_name, nhcalc, rh, df_mech)

        return getattr(self, chn_name + '_prop')[mech_key].copy()


    def init_file(self, path, settings, t0):
        '''
        initiate hdf5 file for data saving
        '''
        # initiated attributes
        self._init_attrs()

        self.mode = 'init'
        self.path = path
        # save some keys from settings_init for future data manipulation
        self.settings = settings

        # self.exp_ref = self._make_exp_ref() # use the settings_int values to format referene dict
        self.exp_ref['t0'] = t0

        # get directory
        direct = os.path.dirname(path)
        # check if folder exist
        if not os.path.isdir(direct): # directory doesn't exist
            os.makedirs(direct) # create directory

        # create groups for raw data
        # dt = h5py.special_dtype(vlen=str)
        with h5py.File(path, 'w') as fh:
            fh.create_group('data')
            fh.create_group('raw')
            fh.create_group('prop')
        
        # save version information
        self._save_ver()
        print(self.ver) #testprint
        # save settings here to make sure the data file has enough information for reading later, even though the program crashes while running
        self.save_data_settings(settings=self.settings, settings_init=self.settings_init)


    def load_file(self, path):
        '''
        load data information from exist hdf5 file
        '''
        self._init_attrs()

        self.mode = 'load'
        self.path = path

        # get data information
        with h5py.File(self.path, 'r') as fh:
            key_list = list(fh.keys())
            print(key_list) #testprint
            
            # check if the file is data file with right format
            # if all groups/attrs in files

            # get data save to attributes
            
            self.settings = json.loads(fh['settings'][()])
            self.settings_init = json.loads(fh['settings_init'][()])
            self.exp_ref = json.loads(fh['exp_ref'][()])
            self.ver = fh.attrs['ver']
            print(self.ver) #testprint
            print(self.exp_ref) #testprint
            # get queue_list
            # self.queue_list = list(fh['raw/samp'].keys())
            # self.queue_list = [int(s) for s in fh['raw/samp'].keys()]

            print(fh['data/samp'][()]) #testprint
            for chn_name in self._chn_keys:
                # df for data from samp/ref chn
                setattr(self, chn_name, pd.read_json(fh['data/' + chn_name][()]).sort_values(by=['queue_id'])) 
                
                # df for data form samp_ref/ref_ref chn
                setattr(self, chn_name + '_ref', pd.read_json(fh['data/' + chn_name + '_ref'][()]).sort_values(by=['queue_id'])) 

                # load prop
                if ('prop' in fh.keys()) and (chn_name in fh['prop'].keys()): # prop exists
                    for mech_key in fh['prop/' + chn_name].keys():
                        getattr(self, chn_name + '_prop')[mech_key] = pd.read_json(fh['prop/' + chn_name + '/' + mech_key][()]).sort_values(by=['queue_id']) 
                
                
            # replace None with nan in self.samp and self.ref
            self.replace_none_with_nan_after_loading() 

            # get queue_list for each channel
            # method 1: from raw. problem of this method is repeat queue_id may be created after deleting data points. 
            queue_samp_raw = []
            queue_ref_raw = []
            if 'samp' in fh['raw'].keys():
                queue_samp_raw = [int(s) for s in fh['raw/samp'].keys()]
            if 'ref' in fh['raw'].keys():
                queue_ref_raw = [int(s) for s in fh['raw/ref'].keys()]
            # method 2: from data
            queue_samp_data = self.samp.queue_id.values # TODO add checking marker != -1
            queue_ref_data = self.ref.queue_id.values
            self.queue_list = sorted(list(set(queue_samp_data) | set(queue_ref_data) | set(queue_samp_raw) | set(queue_ref_raw)))

            self.raw  = {} # raw data from last queue

            self.saveflg = True

            # print(self.samp) #testprint


    def load_settings(self, path):
        '''
        load settings from h5 file
        '''
        try: # try to load settings from file
            with h5py.File(path, 'r') as fh:
                key_list = list(fh.keys())
                print(key_list) #testprint
                
                # check if the file is data file with right format
                # if all groups/attrs in files

                # get data save to attributes
                
                settings = json.loads(fh['settings'][()])
                ver = fh.attrs['ver']
            return settings
        except: # failed to load settings
            return None


    def update_refit_data(self, chn_name, queue_id, harm_list, fs=[np.nan], gs=[np.nan]):
        '''
        update refitted data of queue_id 
        chn_name: str. 'samp' of 'ref'
        harm_list: list of str ['1', '3', '5', '7', '9']
        fs: list of delta freq with the same lenght to harm_list
        gs: list of delta gamma with the same lenght to harm_list
        '''

        # get 
        fs_all = self.get_queue(chn_name, queue_id, col='fs')
        gs_all = self.get_queue(chn_name, queue_id, col='gs')

        # append to the form by chn_name
        # prepare data: change list to the size of harm_list by inserting nan to the empty harm
        for i, harm in enumerate(harm_list):
            harm = int(harm)
            fs_all[int((harm-1)/2)] = fs[i]
            gs_all[int((harm-1)/2)] = gs[i]

        # update values
        self.update_queue_col(chn_name, queue_id, 'fs', fs_all)
        self.update_queue_col(chn_name, queue_id, 'gs', gs_all)

        self.saveflg = False


    def dynamic_save(self, chn_names, harm_list, t=np.nan, temp=np.nan, f=None, G=None, B=None, fs=[np.nan], gs=[np.nan], marks=[0]):
        '''
        save raw data of ONE QUEUE to self.raw and save to h5 file
        NOTE: only update on test a time
        chn_names: list of chn_name ['samp', 'ref']
        harm_list: list of str ['1', '3', '5', '7', '9']
        t: dict of str
        temp: float
        f, G, B: dict of ndarray f[chn_name][harm]
        fs: dicts of delta freq fs[chn_name]
        gs: dicts of delta gamma gs[chn_name]
        marks: [0]. by default all will be marked as 0
        '''

        # add current queue id to queue_list as max(queue_list) + 1
        if not self.queue_list:
            self.queue_list.append(0)
        else:
            print(self.queue_list) #testprint
            self.queue_list.append(max(self.queue_list) + 1)

        for i in range(1, self.settings_init['max_harmonic']+2, 2):
            if str(i) not in harm_list: # tested harmonic
                marks.insert(int((i-1)/2), np.nan)

        # append to the form by chn_name
        for chn_name in chn_names:
            # prepare data: change list to the size of harm_list by inserting nan to the empty harm
            for i in range(1, self.settings_init['max_harmonic']+2, 2):
                if str(i) not in harm_list: # tested harmonic
                    fs[chn_name].insert(int((i-1)/2), np.nan)
                    gs[chn_name].insert(int((i-1)/2), np.nan)

            # up self.samp/ref first
            # create a df to append
            data_new = pd.DataFrame.from_dict({
                'queue_id': [max(self.queue_list)],  
                't': [t[chn_name]],
                'temp': [temp[chn_name]],
                'marks': [marks], # list default [0, 0, 0, 0, 0]
                'fs': [fs[chn_name]],
                'gs': [gs[chn_name]],
            })
            # print(data_new) #testprint
            setattr(self, chn_name, getattr(self, chn_name).append(data_new, ignore_index=True))
            print(self.samp.tail()) #testprint

        # save raw data to file by chn_names
        self._save_raw(chn_names, harm_list, t=t, temp=temp, f=f, G=G, B=B)

        self.saveflg = False


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
                    print(temp) #testprint
                    g_queue.attrs['temp'] = temp[chn_name]

                for harm in harm_list:
                    # create data_set for f, G, B of the harm
                    g_queue.create_dataset(harm, data=np.stack((f[chn_name][harm], G[chn_name][harm], B[chn_name][harm]), axis=0))
        t1 = time.time()
        print(t1 - t0) #testprint


    def save_data(self):
        '''
        save samp (df), ref (df) to h5 file serializing with json
        '''
        with h5py.File(self.path, 'a') as fh:
            for key in self._chn_keys:
                print(key) #testprint
                # print(fh['data/' + key]) #testprint
                if key in fh['data']:
                    del fh['data/' + key]
                    del fh['data/' + key + '_ref']
                print(json.dumps(getattr(self, key).to_dict())) #testprint
                # fh['data/' + key] = json.dumps(getattr(self, key).to_dict())        
                fh.create_dataset('data/' + key, data=getattr(self, key).to_json(), dtype=h5py.special_dtype(vlen=str))  
                fh.create_dataset('data/' + key + '_ref', data=getattr(self, key + '_ref').to_json(), dtype=h5py.special_dtype(vlen=str))  


    def save_settings(self, settings={}, settings_init={}):
        '''
        save settings (dict) to file
        '''
        if not settings:
            settings = self.settings
        if not settings_init:
            settings_init = self.settings_init

        with h5py.File(self.path, 'a') as fh:
            if 'settings' in fh:
                del fh['settings']
            fh.create_dataset('settings', data=json.dumps(settings))
            if 'settings_init' in fh:
                del fh['settings_init']
            fh.create_dataset('settings_init', data=json.dumps(settings_init))


    def save_data_settings(self, settings={}, settings_init={}):
        '''
        wrap up of save_data and save_settings and save_exp_ref
        '''
        self.save_data()
        self.save_settings(settings=settings, settings_init=settings_init)
        self.save_prop()
        self.save_exp_ref()
        self.saveflg = True


    def save_prop(self):
        '''
        save prop data to file
        '''
        with h5py.File(self.path, 'a') as fh:
            for chn_name in self._chn_keys:
                for mech_key, mech_df in getattr(self, chn_name + '_prop').items():
                    if ('prop' not in fh.keys()) or (chn_name not in fh['prop'].keys()):
                        fh.create_dataset('prop/' + chn_name + '/' + mech_key, data=mech_df.to_json(), dtype=h5py.special_dtype(vlen=str))
                    else: 
                        if mech_key in fh['prop/' + chn_name].keys():
                            del fh['prop/' + chn_name + '/' + mech_key]

                        # create data_set for mech_df
                        fh.create_dataset('prop/' + chn_name + '/' + mech_key, data=mech_df.to_json(), dtype=h5py.special_dtype(vlen=str))


    def save_exp_ref(self):
        with h5py.File(self.path, 'a') as fh:
            # save reference
            print(self.exp_ref) #testprint
            if 'exp_ref' in fh:
                del fh['exp_ref']
            fh.create_dataset('exp_ref',data=json.dumps(self.exp_ref))


    def _save_ver(self):
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


    def nan_harm_list(self):
        return [np.nan] * int((self.settings_init['max_harmonic'] + 1) / 2)


    def get_raw(self, chn_name, queue_id, harm):
        '''
        return a set of raw data (f, G, B)
        '''
        with h5py.File(self.path, 'r') as fh:
            self.raw = fh['raw/' + chn_name + '/' + str(queue_id) + '/' + harm][()]
        print(type(self.raw)) #testprint
        return [self.raw[0, :], self.raw[1, :], self.raw[2, :]]


    def get_queue(self, chn_name, queue_id, col=''):
        '''
        get data of a queue_id
        if col == '': return whole row
        if col == column name: return the column
        '''
        df_chn = getattr(self, chn_name).copy()
        if col == '':
            return df_chn.loc[df_chn.queue_id == queue_id, :]
        elif col in df_chn.keys(): # col is a column name
            return df_chn.loc[df_chn.queue_id == queue_id, col]


    def update_queue_col(self, chn_name, queue_id, col, val):
        '''
        update col data of a queue_id
        '''
        print(getattr(self, chn_name).loc[getattr(self, chn_name).queue_id == queue_id, col]) #testprint
        print(val) #testprint
        print(pd.Series([val])) #testprint
        # getattr(self, chn_name).at[getattr(self, chn_name).queue_id == queue_id, [col]] = pd.Series([val])
        getattr(self, chn_name)[getattr(self, chn_name).queue_id == queue_id][col] = [val]
        print(getattr(self, chn_name).loc[getattr(self, chn_name).queue_id == queue_id, col]) #testprint

        self.saveflg = False


    def replace_none_with_nan_after_loading(self):
        '''
        replace the None with nan in marks, fs, gs
        '''
        for chn_name in self._chn_keys:
            for ext in ['', '_ref']: # samp/ref and samp_ref/ref_ref
                df = getattr(self, chn_name + ext)
                col_endswith_s = [col for col in df.columns if col.endswith('s')]
                print(col_endswith_s) #testprint
                
                for col in col_endswith_s:
                    df[col] = df[col].apply(lambda row: [np.nan if x is None else x for x in row])


    def update_mech_df_in_prop(self, chn_name, nhcalc, rh, mech_df):
        '''
        save mech_df to self.'chn_nam'_mech[nhcalc]
        '''
        # set queue_id as int
        mech_df['queue_id'] = mech_df.queue_id.astype(int)

        getattr(self, chn_name + '_prop')[self.get_mech_key(nhcalc, rh)] = mech_df
        print('mech_df in data_saver') #testprint
        print(getattr(self, chn_name + '_prop')[self.get_mech_key(nhcalc, rh)]) #testprint
        self.saveflg = False
    

    def get_mech_df_in_prop(self, chn_name, nhcalc, rh):
        '''
        get mech_df from self.'chn_nam'_mech[nhcalc]
        '''
        return getattr(self, chn_name + '_prop')[nhcalc].copy()


    ####################################################
    ##              data convert functions            ##
    ##  They can also be used external for accessing, ## 
    ##  converting data.
    ####################################################

    def data_exporter(self, fileName, mark=False, dropnanmarkrow=False, dropnancolumn=True, unit_t='s', unit_temp='C'):
        '''
        this function export the self.data.samp and ...ref 
        in the ext form
        fileName: string of full path with file name
        '''
        # TODO add exp_ref

        # get df of samp and ref channel
        on_cols = ['queue_id', 't', 'temp']
        df_samp = pd.merge(
            # f
            self.reshape_data_df('samp', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, unit_t=unit_t, unit_temp=unit_temp, keep_mark=False),
            # delf
            self.reshape_data_df('samp', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, deltaval=True, unit_t=unit_t, unit_temp=unit_temp, keep_mark=True), # keep only on marks
            on=['queue_id', 't', 'temp']
        )

        df_samp_ref = self.reshape_data_df('samp_ref', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, unit_t=unit_t, unit_temp=unit_temp)

        if self.ref.shape[0] > 0:
            df_ref = pd.merge(
                self.reshape_data_df('ref', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, unit_t=unit_t, unit_temp=unit_temp, keep_mark=False),
                self.reshape_data_df('ref', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, deltaval=True, unit_t=unit_t, unit_temp=unit_temp, keep_mark=True), # keep only one marks
                on=['queue_id', 't', 'temp']
            )

            df_ref_ref = self.reshape_data_df('ref_ref', mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn, unit_t=unit_t, unit_temp=unit_temp)

         # get ext
        name, ext = os.path.splitext(fileName)

        # export by ext
        if ext.lower() == '.xlsx':
            with pd.ExcelWriter(fileName) as writer:
                df_samp.to_excel(writer, sheet_name='S_channel')
                df_samp_ref.to_excel(writer, sheet_name='S_reference')
                if self.ref.shape[0] > 0:
                    df_ref.to_excel(writer, sheet_name='R_channel')
                    df_ref_ref.to_excel(writer, sheet_name='R_reference')
                # time reference
                t_ref = {key: self.exp_ref[key] for key in self.exp_ref.keys() if 't0' in key}
                pd.DataFrame.from_dict(t_ref, orient='index').to_excel(writer, sheet_name='time_reference', header=False)
                # property
                for chn_name in self._chn_keys:
                    if getattr(self, chn_name + '_prop'): 
                        for mech_key in getattr(self, chn_name + '_prop').keys(): 
                            mech_df = self.reshape_mech_df(chn_name, mech_key, mark=mark, dropnanmarkrow=dropnanmarkrow, dropnancolumn=dropnancolumn)
                            mech_df.to_excel(writer, sheet_name=chn_name[0].upper() + '_' + mech_key)


        elif ext.lower() == '.csv': # TODO add prop
            # add chn_name to samp and ref df
            # and append ref to samp
            with open(fileName, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(['Version'] + [self.ver] + [''] + ['t0'] + [self.exp_ref['t0']]+ [''] + ['shifted t0'] + [self.exp_ref['t0_shifted']])
            
            if self.ref.shape[0] > 0:
                df_samp.assign(chn='samp').append(df_ref.assign(chn='ref')).append(df_samp_ref.assign(chn='samp_ref')).append(df_ref_ref.assign(chn='ref_ref')).to_csv(fileName, mode='a')
            else:
                df_samp.assign(chn='samp').append(df_samp_ref.assign(chn='samp_ref')).to_csv(fileName, mode='a')

        elif ext.lower() == '.json': # TODO add prop
            with open(fileName, 'w') as f:
                ## lines with indent (this will make the file larger)
                # lines = json.dumps({'samp': self.samp.to_dict(), 'ref': self.ref.to_dict()}, indent=4) + '\n'
                # f.write(lines)

                ## without separate by lines (smaller file size, but harder to)
                if self.ref.shape[0] > 0:
                    json.dump({
                        'samp': df_samp.to_dict(), 
                        'samp_ref': df_samp_ref.to_dict(), 
                        'ref': df_ref.to_dict(),
                        'ref_ref': df_ref_ref.to_dict(),
                        'exp_ref': self.exp_ref,
                        'ver': self.ver,
                        }, 
                        f
                    )
                else:
                    json.dump({
                        'samp': df_samp.to_dict(), 
                        'samp_ref': df_samp_ref.to_dict(), 
                        'exp_ref': self.exp_ref,
                        'ver': self.ver,
                        }, 
                        f
                    )

                # json.dump(data, f)

            
    def reshape_data_df(self, chn_name, mark=False, dropnanmarkrow=True, dropnancolumn=True, deltaval=False, norm=False, unit_t=None, unit_temp=None, keep_mark=True):
        '''
        reshape and tidy data df (samp and ref) for exporting
        keep_mark works when dropnanmarkrow == False
        '''
        cols = ['fs', 'gs']
        df = getattr(self, chn_name).copy()

        # convert t column to datetime object
        df['t'] = self.get_t_by_unit(chn_name, unit=unit_t)
        df['temp'] = self.get_temp_by_unit(chn_name, unit=unit_temp)
        print(df.t) #testprint

        for col in cols:
            df = df.assign(**self.get_list_column_to_columns(chn_name, col, mark=mark, deltaval=deltaval, norm=norm)) # split columns: fs and gs
            df = df.drop(columns=col) # drop columns: fs and gs

        print(df.head()) #testprint

        # drop columns with all 
        if dropnancolumn == True:
            df = df.dropna(axis='columns', how='all')
            if 'temp' not in df.columns: # no temperature data
                df['temp'] = np.nan # add temp column back
        print(df.head()) #testprint

        if dropnanmarkrow == True: # rows with marks only
            # select rows with marks
            print('reshape_data_df: dropnanmarkrow = True') #testprint
            df = df[self.rows_with_marks(chn_name)][:]
            print(df) #testprint
            if 'marks' in df.columns:
                df = df.drop(columns='marks') # drop marks column
        else:
            print('there is no marked data.\n no data will be deleted.') 


        if 'marks' in df.columns:
            # split marks column
            if keep_mark:
                df = df.assign(**self.get_list_column_to_columns(chn_name, 'marks', mark=mark, norm=False))
            # finally drop the marks column
            df = df.drop(columns='marks') # drop marks column

        return df


    def reshape_mech_df(self, chn_name, mech_key, mark=False, dropnanmarkrow=True, dropnancolumn=True):
        '''
        reshape and tidy mech df (mech_key in samp_prop and ref_prop) for exporting
        '''
        print(chn_name) #testprint
        print(mech_key) #testprint
        df = getattr(self, chn_name + '_prop')[mech_key].copy()
        cols = list(df.columns)
        cols.remove('queue_id')
        for col in cols:
            df = df.assign(**self.get_mech_column_to_columns(chn_name, mech_key, col, mark=mark)) # split columns: fs and gs
            df = df.drop(columns=col) # drop columns: fs and gs

        print(df.head()) #testprint

        # drop columns with all 
        if dropnancolumn == True:
            df = df.dropna(axis='columns', how='all')
        print(df.head()) #testprint

        if dropnanmarkrow == True: # rows with marks only
            # select rows with marks
            df = df[self.rows_with_marks(chn_name)][:]
            print(df) #testprint
            if 'marks' in df.columns:
                df = df.drop(columns='marks') # drop marks column
        else:
            print('there is no marked data.\n no data will be deleted.')

        return df


    def get_t_marked_rows(self, chn_name, dropnanmarkrow=False, unit=None):
        '''
        return rows with marks of df from self.get_t_s
        '''
        if dropnanmarkrow == True:
            return self.get_t_by_unit(chn_name, unit=unit)[[self.rows_with_marks(chn_name)]]
        else:
            return self.get_t_by_unit(chn_name, unit=unit)


    def get_t_by_unit(self, chn_name, unit=None):
        '''
        get time in given unit
        '''
        return self.time_s_to_unit(self.get_t_s(chn_name), unit=unit)


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
            print('no data saved!')
            return t
        else:
            print(self.get_t_ref()) #testprint
            print(t) #testprint
            t = t -  self.get_t_ref() # delta t to reference (t0)
            t = t.dt.total_seconds() # convert to second
            return t


    def get_queue_id_marked_rows(self, chn_name, dropnanmarkrow=False):
        '''
        return rows with marks of df from self.get_queue_id
        '''
        if dropnanmarkrow == True:
            return self.get_queue_id(chn_name)[self.rows_with_marks(chn_name)]
        else:
            return self.get_queue_id(chn_name)


    def get_queue_id(self, chn_name):
        '''
        get queue indices as pd.series
        '''
        return getattr(self, chn_name)['queue_id'].copy()


    def get_idx_marked_rows(self, chn_name, dropnanmarkrow=False):
        '''
        return rows with marks of df from self.get_queue_id
        '''
        if dropnanmarkrow == True:
            return self.get_idx(chn_name)[self.rows_with_marks(chn_name)]
        else:
            return self.get_idx(chn_name)


    def get_idx(self, chn_name):
        '''
        get indices as pd.series
        '''
        idx = getattr(self, chn_name).index
        return pd.Series(idx)


    def get_temp_by_uint_marked_rows(self, chn_name, dropnanmarkrow=False, unit='C'):
        '''
        return rows with marks of df from self.get_temp_C
        '''
        if dropnanmarkrow == True:
            return self.get_temp_by_unit(chn_name, unit=unit)[self.rows_with_marks(chn_name)]
        else:
            return self.get_temp_by_unit(chn_name, unit=unit)


    def get_temp_by_unit(self, chn_name, unit='C'):
        '''
        return temp by given unit
        '''
        return self.temp_C_to_unit(self.get_temp_C(chn_name), unit=unit)


    def get_temp_C(self, chn_name):
        '''
        get temperature (temp) in sec as pd.series
        t: pd.series of str
        '''
        return getattr(self, chn_name)['temp'].copy()


    def get_t_ref(self):
        '''
        get reference time and shift it by delt
        '''
        # find reference t from dict exp_ref first
        if 't0' in self.exp_ref.keys() and self.exp_ref.get('t0', None): # t0 exist and != None or 0
            if self.exp_ref.get('t0_shifted', None):
                t0 = datetime.datetime.strptime(self.exp_ref.get('t0_shifted'), self.settings_init['time_str_format']) # used shifted t0
            else:
                t0 = datetime.datetime.strptime(self.exp_ref.get('t0'), self.settings_init['time_str_format']) # use t0
        else: # no t0 saved in self.exp_ref
            # find t0 in self.settings
            t0 = self.settings.get('dateTimeEdit_reftime', None)
            print() #testprint
            print('t0', t0) #testprint
            if not t0:
                if self.samp.shape[0]> 0: # use the first queque time
                    t0 = datetime.datetime.strptime(self.samp['t'][0], self.settings_init['time_str_format'])
                else:
                    t0 = None
            else:
                self.exp_ref['t0'] = t0 # save t0 to exp_ref
                t0 = datetime.datetime.strptime(t0, self.settings_init['time_str_format'])
        
        return t0


    def get_cols(self, chn_name, cols=[]):
        '''
        return a copy of df with all rows and giving columns list
        '''
        return getattr(self, chn_name).loc[:, cols].copy()


    def get_list_column_to_columns_marked_rows(self, chn_name, col, mark=False, dropnanmarkrow=False, deltaval=False, norm=False):
        '''        
        return rows with marks of df from self.get_list_column_to_columns
        '''
        cols_df = self.get_list_column_to_columns(chn_name, col, mark=mark, deltaval=deltaval, norm=norm)
        if dropnanmarkrow == True:
            return cols_df[[self.rows_with_marks(chn_name)]][:]
        else:
            return cols_df


    def get_list_column_to_columns(self, chn_name, col, mark=False, deltaval=False, norm=False):
        '''
        get a df of marks, fs or gs by open the columns with list to colums by harmonics
        chn_name: str of channel name ('sam', 'ref')
        col: str of column name ('fs' or gs')
        mark: if True, show marked data only
        deltaval: if True, convert abs value to delta value
        norm: if True, normalize value by harmonic (works only when deltaval is True)
        return: df with columns = ['1', '3', '5', '7', '9]
        '''

        if deltaval == True:
            s = self.convert_col_to_delta_val(chn_name, col, norm=norm)
            print(s) #testprint
            col = 'del' + col # change column names to 'delfs' or 'delgs' 
            if norm:
                col = col[:-1] + 'n' + col[-1:] # change columns to 'delfns' or 'delgns'
        else:
            s = getattr(self, chn_name)[col].copy()


        if mark == False:
            return pd.DataFrame(s.values.tolist(), s.index).rename(columns=lambda x: col[:-1] + str(x * 2 + 1))
        else:
            m = getattr(self, chn_name)['marks'].copy()
            print('mmmmm', m) #testprint
            idx = s.index
            # convert s and m to ndarray
            arr_s = np.array(s.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            print(arr_s) #testprint
            arr_m = np.array(m.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            print(arr_m) #testprint
            print(np.any(arr_m == 1)) #testprint
            if np.any(arr_m == 1): # there are marks (1)
                print('there are marks (1) in df') #testprint
                # replace None with np.nan
                arr_s = arr_s * arr_m # leave values where only marks == 1
                # replace unmarked (marks == 0) with np.nan
                arr_s[arr_s == 0] = np.nan

                return pd.DataFrame(data=arr_s, index=idx).rename(columns=lambda x: col[:-1] + str(x * 2 + 1))
            else: # there is no marks(1)
                return pd.DataFrame(s.values.tolist(), s.index).rename(columns=lambda x: col[:-1] + str(x * 2 + 1))
            

    def get_mech_column_to_columns_marked_rows(self, chn_name, mech_key, col, mark=False, dropnanmarkrow=False):
        '''        
        return rows with marks of mech_df from self.get_mech_column_to_columns
        '''
        cols_df = self.get_mech_column_to_columns(chn_name, mech_key, col, mark=mark)
        if dropnanmarkrow == True:
            return cols_df[[self.rows_with_marks(chn_name)]][:]
        else:
            return cols_df


    def get_mech_column_to_columns(self, chn_name, mech_key, col, mark=False):
        '''
        make a df of given column (col) in mech_df (all with list) to colums by harmonics
        chn_name: str of channel name ('sam', 'ref')
        col: str of column name ('drho', 'grho', etc.)
        mark: if True, show marked data only by masking others to nan
        return: df with columns = ['1', '3', '5', '7', '9]
        NOTE: The end 's' in column names will not be removed here because the 's' in mech_df shows the value is harmonic dependent.
        '''

        s = getattr(self, chn_name + '_prop')[mech_key][col].copy()
        print('sssss', s) #testprint

        if mark == False:
            return pd.DataFrame(s.values.tolist(), s.index).rename(columns=lambda x: col + str(x * 2 + 1))
        else:
            m = getattr(self, chn_name)['marks'].copy() # marks from data
            idx = s.index
            # convert s and m to ndarray
            arr_s = np.array(s.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            print(arr_s) #testprint
            arr_m = np.array(m.values.tolist(), dtype=np.float) # the dtype=np.float replace None with np.nan
            print(arr_m) #testprint
            print(np.any(arr_m == 1)) #testprint
            if np.any(arr_m == 1): # there are marks (1)
                print('there are marks (1) in df') #testprint
                # replace None with np.nan
                arr_s = arr_s * arr_m # leave values where only marks == 1
                # replace unmarked (marks == 0) with np.nan
                arr_s[arr_s == 0] = np.nan

                return pd.DataFrame(data=arr_s, index=idx).rename(columns=lambda x: col + str(x * 2 + 1))
            else: # there is no marks(1)
                return pd.DataFrame(s.values.tolist(), s.index).rename(columns=lambda x: col + str(x * 2 + 1))
            

    def convert_col_to_delta_val(self, chn_name, col, norm=False):
        '''
        convert fs or gs column to delfs or delgs
        and return the series 
        norm: if True, nomalize value by harmonic
        '''
        # check if the reference is set
        if not self.refflg[chn_name]:
            print(self.exp_ref[chn_name + '_ref']) #testprint
            self.set_ref_set(chn_name, *self.exp_ref[chn_name + '_ref'])

        # get a copy
        col_s = getattr(self, chn_name)[col].copy()
        print(self.exp_ref[chn_name]) #testprint
        if all(np.isnan(np.array(self.exp_ref[chn_name][self._ref_keys[col]]))): # no reference or no constant reference exist
            # check col+'_ref'
            if self.exp_ref[chn_name + '_ref'][1][0] is None or len(self.exp_ref[chn_name + '_ref'][1]) == 0: #start index is None or [], dynamic reference
                print('dynamic reference') 
                ref_s=getattr(self, self.exp_ref[chn_name + '_ref'][0]).copy()

                # convert series value to ndarray
                col_arr = np.array(col_s.values.tolist())
                ref_arr = np.array(ref_s.values.tolist())

                # subtract ref from col elemental wise
                col_arr = col_arr - ref_arr

                # save it back to col_s
                col_s.values[:] = list(col_arr)
                
                if norm: # normalize the data by harmonics
                    col_s = self._norm_by_harm(col_s)
                return col_s
            elif self.exp_ref[chn_name + '_ref'][0] not in ['ext', 'none']: # use constant ref in self.<chn_name>_ref
                # TODO may try to change the ref type self.exp_ref[chn_name + '_ref'][0]
            
                print('ref still not set')
                return col_s.apply(lambda x: list(np.array(x, dtype=np.float) * np.nan)) # return all nan

                # set the ref
                # self.set_ref_set(chn_name, *self.exp_ref[chn_name + '_ref'])

                # print(self.exp_ref) #testprint
                # np.isnan(np.array(self.exp_ref[chn_name][self._ref_keys[col]]))

        else: # there is a constant reference exist
            print('constant reference')
            # get ref
            ref = self.exp_ref[chn_name][self._ref_keys[col]] # return a ndarray
            # return
            print(ref) #testprint
            print(col_s[0]) #testprint
            
            col_s = col_s.apply(lambda x: list(np.array(x, dtype=np.float) - np.array(ref, dtype=np.float)))
            if norm:
                return self._norm_by_harm(col_s)
            else: 
                return col_s


    def _norm_by_harm(self, s):
        '''
        normalize series 'fs' or 'gs' by harmonic
        this function doesn't change the column name of series
        '''
        return s.apply(lambda x: list(np.array(x, dtype=np.float) / np.arange(1, len(x)*2+1, 2)))


    def minus_columns(self, df):
        '''
        change the df value by df * -1 and 
        change the column names by adding m in front
        '''
        # change value
        df = - df 
        # rename
        df.rename(columns=lambda x: 'm'+ x, inplace=True) 
        return df


    def copy_to_ref(self, chn_name, df=None):
        '''
        copy df to self.[chn_name + '_ref'] as reference
        df should be from another file, self.samp or self.ref
        ''' 
        
        # check self.exp_ref
        if self.exp_ref['samp_ref'][0] in self._chn_keys: # use test data
            # reset mark (1 to 0) and copy
            if df is None:
                df = getattr(self, self.exp_ref['samp_ref'][0]).copy()
        else:
            raise ValueError('df should not be None when {0} is reference source.'.fromat(self.exp_ref['samp_ref'][0]))            

        df = self.reset_match_marks(df, mark_pair=(0, 1)) # mark 1 to 0
        setattr(self, chn_name + '_ref', df)


    def set_ref_set(self, chn_name, source, idx_list=[], df=None):
        '''
        set self.exp_ref.<chn_name>_ref value
        source: str in ['samp', 'ref', 'ext', 'none']
        '''
        if getattr(self, chn_name).shape[0] > 0: # data is not empty
            self.exp_ref[chn_name + '_ref'][0] = source
            if len(idx_list) > 0: # 
                self.exp_ref[chn_name + '_ref'][1] = idx_list

                # copy df to ref
                if source in self._chn_keys and self.exp_ref[chn_name + '_ref'][1][0] is not None: # use data from current test
                    df = getattr(self, source)
                    self.copy_to_ref(chn_name, df.loc[idx_list, :]) # copy to reference data set
                elif source == 'ext': # data from external file
                    if df is not None:
                        self.copy_to_ref(chn_name, df) # copy to reference data set
                    else:
                        print('no dataframe is provided!')
                else: # source in self._chn_keys and self.exp_ref[chn_name + '_ref'][1][0] is None:
                    # point by point referencing, don't need copy data
                    pass 
            self.calc_fg_ref(chn_name, mark=True)
        

    def calc_fg_ref(self, chn_name, mark=True):
        '''
        calculate reference of f (f0) and g (g0)  by the set in self.samp_ref, self.ref_ref and save them in self.exp_ref['f0'] and ['g0']
        '''
        if self.exp_ref[chn_name + '_ref'][1][0] is None: #start index is None, dynamic reference
            print('reference element by element! clear self.{}_ref & self.exp_ref.{}'.format(chn_name, chn_name))
            
            # clear self.<chn_name>_ref
            setattr(self, chn_name + '_ref', self._make_df())
            # clear self.exp_ref.<chn_name>
            self.exp_ref[chn_name] = {
                'f0': self.nan_harm_list(), # list for each harmonic
                'g0': self.nan_harm_list(), # list for each harmonic
            }
            self.refflg[chn_name] = True

        else: # there is reference data saved
            if getattr(self, chn_name + '_ref').shape[0] > 0: # there is reference data saved
                print('>0') #testprint
                # calculate f0 and g0 
                for col, key in self._ref_keys.items():
                    print(chn_name, col, key) #testprint
                    df = self.get_list_column_to_columns_marked_rows(chn_name + '_ref', col, mark=mark, dropnanmarkrow=False, deltaval=False)
                    print(getattr(self, chn_name + '_ref')[col]) #testprint
                    print(df) #testprint
                    self.exp_ref[chn_name][key] = df.mean().values.tolist()
                self.refflg[chn_name] = True
            else:
                # no data to saved 
                # clear self.exp_ref.<chn_name>
                self.exp_ref[chn_name] = {
                    'f0': self.nan_harm_list(), # list for each harmonic
                    'g0': self.nan_harm_list(), # list for each harmonic
                }
                self.refflg[chn_name] = False

        print(self.exp_ref) #testprint


    def set_t0(self, t0=None, t0_shifted=None):
        '''
        set reference time (t0) to self.exp_ref
        t0: time string
        t0_shifted: time string
        '''
        if t0 is not None:
            if self.mode != 'load': # only change t0 when it is a new file ('init')
                if isinstance(t0, datetime.datetime): # if t0 is datetime obj
                    t0 = t0.strftime(self.settings_init['time_str_format']) # convert to string
                self.exp_ref['t0'] = t0
                print(self.exp_ref) #testprint

        if t0_shifted is not None:
            if isinstance(t0_shifted, datetime.datetime): # if t0_shifted is datetime obj
                t0_shifted = t0_shifted.strftime(self.settings_init['time_str_format']) # convert to string
            self.exp_ref['t0_shifted'] = t0_shifted


    def get_fg_ref(self, chn_name, harms=[]):
        '''
        get reference of f or g from self.exp_ref
        chn_name: 'samp' or 'ref'
        return a dict 
        {'f0': [f0_1, f0_3, ...], 
         'g0': [g0_1, g0_3, ...]}
        '''
        if not harms: # no harmonic is given
            return self.exp_ref[chn_name]
        else:
            idx = [(int(harm)-1)/2 for harm in harms]
            ref_dict = self.exp_ref[chn_name].copy() # make a copy to make sure not change the original one
            ref_dict['f0'] =  [val for i, val in enumerate(ref_dict['f0']) if i in idx]
            ref_dict['g0'] =  [val for i, val in enumerate(ref_dict['g0']) if i in idx]
            # print(ref_dict) #testprint
            # print(self.exp_ref[chn_name]) #testprint
            return ref_dict


    def get_marks(self, chn_name, tocolumns=False):
        '''
        if not tocolumns:
        return a copy of marks column
        if tocolunms:
        return df with multiple columns
        
        '''
        if tocolumns:
            return self.get_list_column_to_columns(chn_name, 'marks', mark=False)
        else:
            return getattr(self, chn_name).marks.copy()


    def get_harm_marks(self, chn_name, harm):
        '''
        return a df of marks for harm
        '''
        # get df of marks in separated colmns
        df_mark = self.get_list_column_to_columns(chn_name, 'marks', mark=False)

        # return the marks of given harm
        return df_mark['mark' + harm]


    def datadf_with_data(self, chn_name, tocolumns=False):
        '''
        df of boolean if corrensponding test with data
        '''
        pass
        

    def harm_with_data(self, chn_name, harm):
        '''
        return a series of booleans to show if the harm has data
        '''
        df_mark = self.get_harm_marks(chn_name, harm)
        return df_mark.notna()


    def rows_with_marks(self, chn_name):
        '''
        return a series of booleans of rows with marked (1) harmonics
        if no marked rows, return all
        '''
        marked_rows = getattr(self, chn_name).marks.apply(lambda x: True if 1 in x else False)
        if marked_rows.any(): # there are marked rows
            print('There are marked rows')
            return marked_rows
        else: # no amrked rows, return all
            print('There is no marked row.\nReturn all')
            return ~marked_rows


    def with_marks(self, chn_name):
        '''
        return if there are marks in data (True/False)
        '''
        marked_rows = getattr(self, chn_name).marks.apply(lambda x: True if 1 in x else False)
        if marked_rows.any(): # there are marked rows
            return True
        else: # no amrked rows
            return False


    def rows_all_nan_marks(self, chn_name):
        '''
        return list of booleans of rows with nan in all harmonics of marks
        This function can be used as ~self.rows_all_nan_marks() to return the rows with data
        '''
        return getattr(self, chn_name).marks.apply(lambda x: True if np.isnan(np.array(x)).all() else False)


    def reset_match_marks(self, df, mark_pair=(0, 1)):
        ''' 
        rest marks column in df. 
        set 1 in list element to 0
        '''
        new_mark, old_mark = mark_pair
        df_new = df.copy()
        print(type(df_new)) #testprint
        print(df_new.tail()) #testprint
        print(df_new.fs) #testprint
        print(df_new.marks) #testprint
        df_new.marks = df_new.marks.apply(lambda x: [new_mark if mark == old_mark else mark for mark in x])
        return df_new


    def mark_data(self, df, idx=[], harm=None, mark_val=1):
        '''
        mark data by given information
        df: data as dataframe
        idx: list of indices (rows)
        harm: str of a single harmonic to mark
        mark_val: int 
        '''
        df_new = df.copy()
        def mark_func(row_marks):
            print(row_marks) #testprint
            new_marks = []
            for i, mark in enumerate(row_marks):
                if str(i*2+1) == harm and mark != np.nan and mark is not None:
                    new_marks.append(mark_val)
                else:
                    new_marks.append(mark)
                return new_marks

        print(df_new.marks) #testprint
        # df_new.marks.loc[idx].apply(mark_func)
        df_new.marks.loc[idx] = df_new.marks.loc[idx].apply(lambda x: [mark_val if str(i*2+1) == harm and mark != np.nan and mark is not None else mark for i, mark in enumerate(x)])
        print(df_new.marks) #testprint

        return df_new


    def mark_all_to(self, df, mark_val=1):
        ''' 
        mark all to given mark_val e.g.: 0, 1
        '''
        df_new = df.copy()
        df_new.marks = df_new.marks.apply(lambda x: [mark_val if mark != np.nan and mark is not None else mark for mark in x])
        print(df_new.marks) #testprint
        return df_new


    ###### selector functions ######
    def selector_mark_all(self, chn_name, mark_val):
        '''
        selector function
        mark all data in chn_name to mark_val
        '''
        setattr(self, chn_name, self.mark_all_to(getattr(self, chn_name), mark_val=mark_val))


    def selector_mark_sel(self, chn_name, sel_idx_dict, mark_val):
        '''
        selector function
        mark selected points in chn_name to mark_val
        sel_idx_dict = {
            'harm': [index]
        }
        '''
        df_chn = getattr(self, chn_name)
        for harm, idx in sel_idx_dict.items():
            df_chn = self.mark_data(df_chn, idx=idx, harm=harm, mark_val=mark_val)
            # print(df_chn.marks) #testprint
        setattr(self, chn_name, df_chn)


    def selector_del_sel(self, chn_name, sel_idx_dict):
        '''
        selector function
        del selected points in chn_name to mark_val
        sel_idx_dict = {
            'harm': [index]
        }
        This function changes the date (fs, gs) to [nan, ...], marks to nan, and delete the raw data
        '''
        df_chn = getattr(self, chn_name).copy()

        for harm, idx in sel_idx_dict.items():
            # set marks to -1
            df_chn = self.mark_data(df_chn, idx=idx, harm=harm, mark_val=np.nan) # set the marks to nan
            # set fs, gs to nan
            df_chn.fs[idx] = df_chn.fs[idx].apply(lambda x: [np.nan if str(i*2+1) in harm else val for i, val in enumerate(x)]) # set all to nan. May not necessary
            df_chn.gs[idx] = df_chn.gs[idx].apply(lambda x: [np.nan if str(i*2+1) in harm else val for i, val in enumerate(x)]) # set all to nan. May not necessary
        
        # delete rows marks are all nan
        df_chn = df_chn[~self.rows_all_nan_marks(chn_name)]
        # reindex df
        df_chn = df_chn.reset_index(drop=True)
        # save back to class
        setattr(self, chn_name, df_chn)

        with h5py.File(self.path, 'a') as fh:
            for harm, idxs in sel_idx_dict.items():
                # delete from raw
                for idx in idxs:
                    print(df_chn.queue_id[idx]) #testprint
                    print(fh['raw/' + chn_name + '/' + str(int(df_chn.queue_id[idx])) + '/' + harm]) #testprint
                    del fh['raw/' + chn_name + '/' + str(int(df_chn.queue_id[idx])) + '/' + harm]


    ######## functions for unit convertion #################


    def time_s_to_unit(self, t, unit=None):
        '''
        convert time from in second to given unit
        input:
            t: scaler or array or pd.series of t as float
            unit: None, or str ('s', 'm', 'h', 'd')
        return:
            same size as input
        '''
        if unit is None:
            return t
        if t.shape[0] == 0: # no data
            return t
        else:
            factors = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
            return t / factors[unit] 


    def temp_C_to_unit(self, temp, unit='C'):
        '''
        convert temp from in C to given unit
        input:
            t: scaler or array or pd.series of t as float
            unit: None, or str ('C', 'K', 'F')
        return:
            same size as input
        '''

        if temp.shape[0] == 0: # no data
            return temp
        else:
            factors = {
                'C': lambda x: x, 
                'K': lambda x: x + 273.15, 
                'F': lambda x: x * 9/5 + 32
                }
            return factors[unit](temp)

    # def abs_to_del(self, abs_list, ref_list):
    #     '''
    #     calculate relative value elemental wisee in two lists by (abs_list - ref_list)
    #     this function is used for calculate delf and delg for a single queue
    #     '''

    #     return [abs_val - ref_val for abs_val, ref_val in zip(abs_list, ref_list)]


    def df_qcm(self, chn_name):
        '''
        convert delfs and delgs in df to delfstar for calculation and 
        return a df with ['queue_id', 'marks', 'fstars', 'fs', 'gs', 'delfstars', 'delfs', 'delgs']
        '''
        df = self.get_queue_id(chn_name).astype('int64').to_frame()
        df['t'] = self.get_t_marked_rows(chn_name, dropnanmarkrow=False, unit=None)
        df['temp'] = self.get_temp_by_uint_marked_rows( chn_name, dropnanmarkrow=False, unit=
        'C')
        df['marks'] = self.get_marks(chn_name)

        # get freqs and gamms in form of [n1, n3, n5, ...]
        fs = self.get_cols(chn_name, cols=['fs'])
        gs = self.get_cols(chn_name, cols=['gs'])
        # get delf and delg in form of [n1, n3, n5, ...]
        delfs = self.convert_col_to_delta_val(chn_name, 'fs', norm=False)
        delgs = self.convert_col_to_delta_val(chn_name, 'gs', norm=False)

        # convert to array
        f_arr = np.array(fs.values.tolist())
        g_arr = np.array(gs.values.tolist())

        delf_arr = np.array(delfs.values.tolist())
        delg_arr = np.array(delgs.values.tolist())

        # get delfstar as array
        fstar_arr = f_arr + 1j * g_arr
        delfstar_arr = delf_arr + 1j * delg_arr

        df['fstars'] = list(fstar_arr)
        df['delfstars'] = list(delfstar_arr)
        df['fs'] = fs
        df['gs'] = gs
        df['delfs'] = delfs
        df['delgs'] = delgs

        return df


    def get_mech_key(self, nhcalc, rh):
        '''
        return a str which represents the key in self.<chn_name>_mech[key]
        '''
        return nhcalc + '_' + str(rh)


    ######## Following functions are for QCM-D 

    def convert_D_to_gamma(self, D, harm):
        '''
        this function convert given D (from QCM-D) to gamma used in this program
        D: dissipation from QCM-D
        harm: str. 
        '''
        return 0.5 * int(harm) * 5 * D


    def import_qcmd(self, path, f1=None, settings=None, t0=None, init_file=True):
        '''
        import QCM-D data to data_saver
        path: excel file path
        f1: base frequency in MHz
        settings: UI settings (dict)
        init_file: True, intialize h5 file for storing the data. By default the UI will save the h5 file with the same name as excel file.
        NOTE: f1 and settings should be given one at least. If both are given, the function will use the settings fist.
        '''
        self.mode = 'qcmd'
        name, ext = os.path.splitext(path)

        # make a fake reference time t0
        if settings:
            t0_str = settings['dateTimeEdit_reftime']
            t0 = datetime.datetime.strptime(t0_str, self.settings_init['time_str_format'])
        else:
            if not t0:
                t0 = datetime.datetime.now()
            t0_str = t0.strftime(self.settings_init['time_str_format'])

        # initialize file
        if init_file and settings:
            self.init_file(name + '.h5', settings=settings, t0=t0_str)
            f1 = self.settings['comboBox_base_frequency'] * 1e6 # in Hz
        elif f1:
            f1 = f1 * 1e6 # in Hz
            self.path = name + '.h5'  # convert xlsx file to h5
            self.set_t0(t0=t0_str) # save t0 (str)

        g1 = 0 # in Hz

        # read QCM-D data
        df = pd.read_excel(path)
        print(df.columns) #testprint
        print(df.shape) #testprint
        print(df.head()) #testprint

        # rename time column (it could be ['time(s)', 't(s)'])
        # df = df.rename(columns={'t(s)': 't'})
        df['t'] = df.filter(regex='(s)')
        df = df.drop(columns=df.filter(regex='(s)').columns)
        # save t as (delt + t0) 
        # df['t'] = (df['t'] + t0).strftime(self.settings_init['time_str_format'])
        df['t'] = df['t'].apply(lambda x: (t0 + datetime.timedelta(seconds=x)).strftime(self.settings_init['time_str_format']))

        harm_list = []
        ref_fs = {'ref':[]}
        ref_gs = {'ref':[]}
        fGB = {'samp': {}, 'ref': {}} # a make up dict for f, G, B
        fgcol_list = df.filter(regex='del').columns
        print(fgcol_list) #testprint
        if len(fgcol_list) % 2: # odd columns
            print('delf and delg column number is not the same!\nPlease check the column name!')
            return
        for harm in range(1, len(fgcol_list), 2):
            harm_list.append(str(harm))
            ref_fs['ref'].append(f1 * harm)
            ref_gs['ref'].append(g1 * harm)
            fGB['samp'][str(harm)] = np.nan
            fGB['ref'][str(harm)] = np.nan

            # convert delf to f
            df['delf' + str(harm)] = df['delf' + str(harm)] + f1 * harm
            # convert D to gamma
            df['delg' + str(harm)] = self.convert_D_to_gamma(df['delg' + str(harm)], harm)
            # convert delg to g
            df['delg' + str(harm)] = df['delg' + str(harm)] + g1 * harm
        
        # f/g to one column fs/gs
        df['fs'] = df.filter(regex='delf').values.tolist()
        df['gs'] = df.filter(regex='delg').values.tolist()
        # delete delf<n> and deg<n>
        df = df.drop(columns=fgcol_list)
        # queue_list 
        df['queue_id'] = list(df.index.astype(int))
        # marks
        single_marks = [0 for _ in harm_list]
        # for i in range(1, self.settings_init['max_harmonic']+2, 2):
        #     if str(i) not in harm_list: # tested harmonic
        #         single_marks.insert(int((i-1)/2), np.nan)
        # print(harm_list) #testprint
        # print(single_marks) #testprint
        # df['marks'] = [single_marks] * df.shape[0]

        ## save to self.samp 
        # self.samp = self.samp.append(df, ignore_index=True, sort=False)

        ## ANOTHER WAY to save is use self.dynamic_save and save the data to self.samp row by row
        for i in df.index: # loop each row
            self.dynamic_save(['samp'], harm_list, t={'samp': df.t[i]}, temp={'samp': np.nan}, f=fGB, G=fGB, B=fGB, fs={'samp': df.fs[i]}, gs={'samp': df.gs[i]}, marks=[0 for _ in harm_list])

        # print(self.samp.head()) #testprint
        print(self.samp.fs[0]) #testprint
        print(self.samp['marks'][0]) #testprint

        self.queue_list = list(self.samp.index)

        # set ref
        # save ref_fs ref_gs to self.ref as reference
        self.dynamic_save(['ref'], harm_list, t={'ref': t0_str}, temp={'ref': np.nan}, f=fGB, G=fGB, B=fGB, fs=ref_fs, gs=ref_gs, marks=[0 for _ in harm_list])
        print(self.ref.head()) #testprint
        print(self.ref['marks'][0]) #testprint
        
        # set reference
        self.set_ref_set('samp', 'ref', idx_list=[0])

        print(self.exp_ref) #testprint

        self.raw = {}

        self.saveflg = False

