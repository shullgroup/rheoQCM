'''
The easist way to export the data is use the export function in the UI 
This is an example file showing how to use DataSaver module to extract 
the data from the data file in Python code.
You may find more functions in DataSaver module. E.g.: changing marks, 
reference frequency/gamma, reference time etc.
'''

import numpy as np
import pandas as pd
import DataSaver # use your own path to DataSaver module

chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
data_saver = DataSaver.DataSaver()
# load data
path = 'string of file path'
data_saver.load_file(path)


########## get single varible #########

# set the data channel you want to access
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
data_col = 'fs' # column name in data dataframe 'fs', 'gs', 'marks'
mark = False # False: all data; True: only marked rows
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
deltaval = True # True: delta value for freq or gamma; False: absolute value for freq or gamma
norm = False # True: normalize delta values with harmonic number (e.g.: delfreq/harm)

col_data = data_saver.get_list_column_to_columns_marked_rows(chn_name, data_col, mark=mark, dropnanrow=dropnanrow, deltaval=deltaval, norm=norm)
print(col_data.head()) 
print(col_data.columns)
# col_data has columns for each harmonic

# get time (referenced to reference time, t0)
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
unit_t = 's' # unit for the temperature data. It can be one from ['s', 'm', 'h', 'd'] for second, min, hour, day, respectively
t = data_saver.get_t_marked_rows(chn_name, dropnanrow=dropnanrow, unit=unit_t)
print(t.head())

# get temperature by given unit string 
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
unit_temp = 'C' # unit for the temperature data. It can be one from ['C', 'F', 'K']
temp = data_saver.get_temp_by_uint_marked_rows(chn_name, dropnanrow=dropnanrow, unit=unit_temp)
print(temp.head())

# get index of data (not the dataframe index)
idx = data_saver.get_queue_id_marked_rows(chn_name, dropnanrow=dropnanrow)
print(idx.head())

# get property data
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
mark = False # False: all data; True: only marked rows
mech_key = '353_3' # string of combination of solving combination and reference harmonic
prop_col = 'drho' # column name in property dataframe
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
prop_data = data_saver.get_mech_column_to_columns_marked_rows(chn_name, mech_key, prop_col, mark=mark, dropnanrow=dropnanrow)


########## get full dataframe ##########

# get QCM data
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
mark = False # False: all data; True: only marked rows
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
dropnancolumn = True # True: delete the harmonics didn't tested; False: keep all harmonics with untested ones
deltaval = True # True: delta value for freq or gamma; False: absolute value for freq or gamma
norm = False # True: normalize delta values with harmonic number (e.g.: delfreq/harm)
unit_t = 's' # unit for the temperature data. It can be one from ['s', 'm', 'h', 'd'] for second, min, hour, day, respectively
unit_temp = 'C' # unit for the temperature data. It can be one from ['C', 'F', 'K']

data_df = data_saver.reshape_data_df(chn_name, mark=mark, dropnanrow=dropnanrow, dropnancolumn=dropnancolumn, deltaval=deltaval, norm=norm, unit_t=unit_t, unit_temp=unit_temp)
print(data_df.head())
print(data_df.columns)

# get property data
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
mark = False # False: all data; True: only marked rows
dropnanrow = False # False: leave the row with all nan as it is; True: remove the rows with no marks
dropnancolumn = True # True: delete the harmonics didn't tested; False: keep all harmonics with untested ones
mech_key = '353_3' # string of combination of solving combination and reference harmonic
prop_df = data_saver. reshape_mech_df(chn_name, mech_key, mark=mark, dropnanrow=dropnanrow, dropnancolumn=dropnancolumn)
print(prop_df.head())
print(prop_df.columns)

# list the property combinations saved in file
chn_name = 'samp' # 'samp' for chnnel 'S'; 'ref' for channel 'R'
print(getattr(data_saver, chn_name + '_prop').keys())