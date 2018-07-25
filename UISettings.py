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
    
    # temperature modules path
    'tempmodules_path': r'./modules/temp/', 

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

    # scan mode
    'scan_mode': {
        'startstop':  'Start/Stop',
        'centerspan': 'Center/Span',
    },

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
        'free':      'Free',
        'fixspan':   'Fix span',
        'fixcenter': 'Fix center',
        'fixcntspn':  'Fix center/span',
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

    'span_ctrl_steps': [1, 2, 5, 10, 20, 50, 100],


    # mpl setings
    'max_mpl_toolbar_height': 20, # in px

    'contour': {
        'levels': 20, # contour levels
        'num': 100, # percentage of step increase for phi and dlam
        'phi_lim': [0, 90], # phi limit in degree
        'dlam_lim': [0, 1], # d/lambda limit
    },

    # font size for mpl_sp figures
    'mpl_sp_fontsize': 5,
    # font size for normal figures
    'mpl_fontsize': 8,
    # legend font size for mpl_sp figures
    'mpl_sp_legfontsize': 5,
    # legend font size for normal figures
    'mpl_legfontsize': 8,
}

settings_default = {
#### default settings control ####
    # default harmonics 
    'harmonics_check': [1, 3, 5],

    # default checkbox harm states
    'checkBox_harm1': True,
    'checkBox_harm3': True,
    'checkBox_harm5': True,
    'checkBox_harm7': False,
    'checkBox_harm9': False,
    'checkBox_harm11': False,

    # default start/end freq lineEdit values
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

    # default time settings
    'lineEdit_reftime': 0,
    
    'lineEdit_recordinterval': 2,
    'lineEdit_refreshresolution': 1,
    'lineEdit_scaninterval': 2,

    # default fitting and display options
    'checkBox_dynamicfit': True,
    'checkBox_showsusceptance': False,
    'checkBox_showchi': False,
    'checkBox_showpolar': False,
    'comboBox_fitfactor': '6',

    # default frequency ranges for each harmonic
    'freq_range': {
        1: [4, 6],
        3: [14, 16],
        5: [24, 26],
        7: [34, 36],
        9: [44, 46],
        11: [54, 56]
    },

#### default settings settings ####
    'tab_settings_settings_harm1': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    'tab_settings_settings_harm3': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    'tab_settings_settings_harm5': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    'tab_settings_settings_harm7': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    'tab_settings_settings_harm9': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    'tab_settings_settings_harm11': {
        # default scan settings
        'start_freq': 4.9,
        'end_freq': 5.1,
        
        # default span settings
        'comboBox_fit_method': 'gmax',
        'comboBox_track_method': 'fixspan',
        
        # default fit settings
        'comboBox_harmfitfactor': '6',
        
        # default VNA settings
        # 'comboBox_sample_channel': '1',
        # 'comboBox_ref_channel': 'none',
        
        # default crystal settings
        #'comboBox_base_frequency': 5,
        #'comboBox_bandwidth': 0.1,

        # default temperature settings
        #'checkBox_settings_temp_sensor': False,
        #'Module': '...',
        #'comboBox_thrmcpltype': 'J',

        # default plots settings
        #'comboBox_timeunit': 's',
        #'comboBox_tempunit': 'C',
        #'comboBox_timescale': 'linear',
        #'comboBox_gammascale': 'linear',
        #'checkBox_settings_settings_linktime': False
    },

    ### default hardware settings ###
    # default VNA settings
    'comboBox_sample_channel': '1',
    'comboBox_ref_channel': 'none',

    # default crystal settings
    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1,

    # default temperature settings
    'checkBox_settings_temp_sensor': False,
    'Module': '...',
    'comboBox_thrmcpltype': 'J',

    # default plots settings
    'comboBox_timeunit': 's',
    'comboBox_tempunit': 'C',
    'comboBox_timescale': 'linear',
    'comboBox_gammascale': 'linear',
    'checkBox_settings_settings_linktime': False,

    ### default plot selections ###
    # default selections for spectra show
    'radioButton_spectra_showBp': True,
    'radioButton_spectra_showpolar': False,
    'checkBox_spectra_shoechi': False,

    # default selections for plot 1 elements
    'comboBox_plt1_choice': 'dfn_t', 
    'checkBox_plt1_h1': False,
    'checkBox_plt1_h3': False,
    'checkBox_plt1_h5': False,
    'checkBox_plt1_h7': False,
    'checkBox_plt1_h9': False,
    'checkBox_plt1_h11': False,
    'radioButton_plt1_samp': True,
    'radioButton_plt1_ref': False,

    # default selections for plot 2 elements
    'comboBox_plt2_choice': 'dg_t',
    'checkBox_plt2_h1': False,
    'checkBox_plt2_h3': False,
    'checkBox_plt2_h5': False,
    'checkBox_plt2_h7': False,
    'checkBox_plt2_h9': False,
    'checkBox_plt2_h11': False,
    'radioButton_plt2_samp': True,
    'radioButton_plt2_ref': False
}

