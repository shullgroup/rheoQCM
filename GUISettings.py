'''
Setting factors for GUI
Change the following factors will change the apperiance of the GUI

'''

# title to display in the main GUI
window_title = 'QCM-R'

# highest harmonic to display. the maximum value is 11
max_harmonic = 11

# default harmonics 
default_harmonics = [1, 3, 5]

# default open/save data file type
default_datafiletype = ';;'.join([
    'Json file (*.json)',
])

# export  data file type
export_datafiletype = ';;'.join([
    'json file (*.json)',
    'hdf5 file (*.h5)',
    'Python file (*.py)',
    'Matlab file (*.mat)',
    'csv file (*.csv)',
])

# default label_actual_interval
actual_interval = 2
acquisition_interval = 2
refresh_resolution = 1
# choice for plotting data shown in comboBox_plt1_choice & comboBox_plt2_choice
plt_choice = {
    'none':   'none',
    'df_t':   u'\u0394' + 'f - time',
    'dfn_t':  u'\u0394' + 'f/n - time',
    'dg_t':   u'\u0394\u0393' + ' - time',
    'dgn_t':  u'\u0394\u0393' + '/n - time',
    'f_t':    'f - time',
    'g_t':    'g - time',
    'temp_t': 'temp. - time'
}

# comboBox_fit_method
span_mehtod_choose = {
    'gmax':   'Gmax',
    'dev':    'Derivative',
    'bmax':   'Bmax',
    'prev':   'Previous value',
    'usrdef': 'User-defined...'
}

# track_method
track_mehtod_choose = {
    'fixspan':   'Fix span',
    'fixcenter': 'Fix center',
    'fixrange':  'Fix range',
    'usrdef':    'User-defined...'
}

# sample_channel
sample_channel_choose = {
# key: number; val: for display in combobox
    '1': 'ADC 1',
    '2': 'ADC 2'
}

# available base frequency of crystals
# key: number; val: for display in combobox
base_frequency_choose = {
    '5':  '5 MHz',
    '6':  '6 MHz',
    '9':  '9 MHz',
    '10': '10 MHz',
}

# available bandwidth limitation for each harmonic
# key: number; val: for display in combobox
bandwidth_choose = {
    '2':  '2 MHz',
    '1':  '1 MHz',
    '0_5':  '0.5 MHz',
    '0_25': '0.25 MHz',
    '0_1': '0.1 MHz',
}

# reference type for showing delta f and delta gamma
# key: number; val: for display in combobox
ref_type_choose = {
    't0':  'First point',
    't1t2':  'Selected range',
    'input':  'Input value',
    'file': 'File',
}

max_mpl_toolbar_height = 23 # in px