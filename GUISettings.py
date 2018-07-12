'''
Setting factors for GUI
Change the following factors will change the apperiance of the GUI
'''
settings_init = {
    # title to display in the main GUI
    'window_title': 'QCM-R',

    # window default size
    'window_size': [1200, 800], # px

    # highest harmonic to display. the maximum value is 11
    'max_harmonic': 11, # do not change
 
    # default open/save data file type
    'default_datafiletype': ';;'.join([
        'Json file (*.json)',
    ]),

    # export  data file type
    'export_datafiletype': ';;'.join([
        'json file (*.json)',
        'hdf5 file (*.h5)',
        'Python file (*.py)',
        'Matlab file (*.mat)',
        'csv file (*.csv)',
    ]),

    # choice for plotting data shown in comboBox_plt1_choice & comboBox_plt2_choice
    'data_plt_choose': {
        'none':   'none',
        'df_t':   u'\u0394' + 'f - time',
        'dfn_t':  u'\u0394' + 'f/n - time',
        'dg_t':   u'\u0394\u0393' + ' - time',
        'dgn_t':  u'\u0394\u0393' + '/n - time',
        'f_t':    'f - time',
        'g_t':    'g - time',
        'temp_t': 'temp. - time'
    },

    # comboBox_fitfactor
    'fit_factor_choose': {
        '6':   '6',
        '5':   '5',
        '4':   '4',
        '3':   '3',
        '2':   '2',

    },

    # comboBox_fit_method
    'span_mehtod_choose': {
        'gmax':   'Gmax',
        'dev':    'Derivative',
        'bmax':   'Bmax',
        'prev':   'Previous value',
        'usrdef': 'User-defined...'
    },

    # track_method
    'track_mehtod_choose': {
        'fixspan':   'Fix span',
        'fixcenter': 'Fix center',
        'fixrange':  'Fix range',
        'usrdef':    'User-defined...'
    },

    # sample_channel
    'sample_channel_choose': {
    # key: number; val: for display in combobox
        '1': 'ADC 1',
        '2': 'ADC 2'
    },

    'ref_channel_choose': {
    # key: number; val: for display in combobox
        'none': '--',
        '1': 'ADC 1',
        '2': 'ADC 2'
    },

    'thrmcpl_choose': {
    # key: number; val: for display in combobox
        'J': 'J',
        'K': 'K',
        'N': 'N',
        'R': 'R',
        'S': 'S',
        'T': 'T',
        'B': 'B',
        'E': 'E',
    },

    'time_unit_choose': {
    # key: number; val: for display in combobox
        's': 's',
        'm': 'min',
        'h': 'h',
        'd': 'day',
    },

    'temp_unit_choose': {
    # key: number; val: for display in combobox
        'C': '°C',
        'K': 'K',
        # 'F': '°F',
    },

    'time_scale_choose': {
    # key: number; val: for display in combobox
        'linear': 'linear',
        'log': 'log',
        # 'F': '°F',
    },

    'gamma_scale_choose': {
    # key: number; val: for display in combobox
        'linear': 'linear',
        'log': 'log',
        # 'F': '°F',
    },


    # available base frequency of crystals
    # key: number; val: for display in combobox
    'base_frequency_choose': {
        5:  '5 MHz',
        6:  '6 MHz',
        9:  '9 MHz',
        10: '10 MHz',
    },

    # available bandwidth limitation for each harmonic
    # key: number; val: for display in combobox
    'bandwidth_choose': {
        2:  '2 MHz',
        1:  '1 MHz',
        0.5:  '0.5 MHz',
        0.25: '0.25 MHz',
        0.1: '0.1 MHz',
    },

    # reference type for showing delta f and delta gamma
    # key: number; val: for display in combobox
    'ref_type_choose': {
        't0':  'First point',
        't1t2':  'Selected range',
        # 'input':  'Input value',
        'file': 'Other file',
    },

    'max_mpl_toolbar_height': 20, # in px
}

settings_default = {
    'harmonic_tab': 1,
    'base_freq': 5,
    'bandwidth': 0.1,

    # default harmonics 
    'harmonics_check': [1, 3, 5],

    # default label_actual_interval
    'actual_interval': 2,
    'acquisition_interval': 2,
    'refresh_resolution': 1,
    
    #################################
    'checkBox_harm1': 1,
    'lineEdit_startf1'
    'lineEdit_endf1'
    'lineEdit_reftime'
    'lineEdit_acquisitioninterval': 2,
    'lineEdit_refreshresolution': 2,
    'label_actualinterval': 1,
    'frame_sp1': 1
}