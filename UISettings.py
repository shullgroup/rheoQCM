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
    #### default settings control ####
    # default harmonics 
    'harmonics_check': [1, 3, 5],

    #'checkBox_harm1'
    #'checkBox_harm3'
    #'checkBox_harm5'
    #'checkBox_harm7'
    #'checkBox_harm9'
    #'checkBox_harm11'

    'lineEdit_startf1': 4.9,
    'lineEdit_startf3': 14.9,
    'lineEdit_startf5': 24.9,
    'lineEdit_startf7': 34.9,
    'lineEdit_startf9': 44.9,
    'lineEdit_startf11': 54.9,
    'lineEdit_endf1': 5.1,
    'lineEdit_endf3': 15.1,
    'lineEdit_endf5': 25.1,
    'lineEdit_endf7': 35.1,
    'lineEdit_endf9': 45.1,
    'lineEdit_endf11': 55.1,

    #'lineEdit_reftime'

    'lineEdit_acquisitioninterval': 2,
    'lineEdit_refreshresolution': 2,
    'label_actualinterval': 1,

    #'frame_sp1'
    #'frame_sp3'
    #'frame_sp5'
    #'frame_sp7'
    #'frame_sp9'
    #'frame_sp11'

    # default label_actual_interval
    'actual_interval': 2,
    'acquisition_interval': 2,
    'refresh_resolution': 1,

    # default fitting and display options
    #'checkBox_dynamicfit'
    #'checkBox_polarplot'
    #'checkBox_showsusceptance'
    #'checkBox_showchi'
    #'comboBox_fitfactor'

    #### default settings settings ####
    #'tab_settings_settings_harm1'
    #'tab_settings_settings_harm3'
    #'tab_settings_settings_harm5'
    #'tab_settings_settings_harm7'
    #'tab_settings_settings_harm9'
    #'tab_settings_settings_harm11'

    #'comboBox_fit_method'
    #'comboBox_track_method'

    #'comboBox_harmfitfactor'

    #'comboBox_sample_channel'
    #'comboBox_ref_channel'

    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1

    #'Temperature'
    #'Module'
    #'comboBox_thrmcpltype'

    #'comboBox_timeunit'
    #'comboBox_tempunit'
    #'comboBox_timescale'
    #'comboBox_gammascale'
    #'Link Time' 
}
print(settings_default['lineEdit_acquisitioninterval'])