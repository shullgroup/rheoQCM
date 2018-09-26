'''
Setting factors for GUI
Change the following factors will change the apperiance of the GUI
'''

# for create ordered dictionary 
# (dictionary in Python >=3.6 is ordered by default)
# Use OrderedDict for those dicts need to be shown in order
from collections import OrderedDict

settings_init = {
    # title to display in the main GUI
    'window_title': 'QCM-R',

    # window default size
    'window_size': [1200, 800], # px

    # highest harmonic to display. the maximum value is 11
    'max_harmonic': 11, # do not change
    
    # temperature modules path
    'tempmodules_path': r'./modules/temp/', 
    
    # add NI sensors into the dict and the code will check if the devices in its keys.
    # the values are the number of samples per test for average
    'devices_dict': {
        'USB-TC01': {
            'nsamples': 1,            # number of points for average,
            'thrmcpl_chan': 'ai0',    # thermocouple channel,
            'cjc_source': 'BUILT_IN', # channel for cjc,
        }, 
        'PCIe-6321': {
            'nsamples': 100,       # number of points for average,
            'thrmcpl_chan': 'ai0', # thermocouple channel,
            'cjc_source': '',      # channel for cjc,
        }, 
    },
    
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
    'display_choose': OrderedDict([
        ('startstop',  'Start/Stop'),
        ('centerspan', 'Center/Span'),
    ]),

    # choice for plotting data shown in comboBox_plt1_choice & comboBox_plt2_choice
    'data_plt_choose': OrderedDict([
        ('none',   'none'),
        ('df_t',   u'\u0394' + 'f - time'),
        ('dfn_t',  u'\u0394' + 'f/n - time'),
        ('mdf_t',  '-' + u'\u0394' + 'f - time'),
        ('mdfn_t', '-' + u'\u0394' + 'f/n - time'),
        ('dg_t',   u'\u0394\u0393' + ' - time'),
        ('dgn_t',  u'\u0394\u0393' + '/n - time'),
        ('f_t',    'f - time'),
        ('g_t',    'g - time'),
        ('temp_t', 'temp. - time'),
    ]),

    # spinBox_fitfactor

    # comboBox_tracking_method
    'span_mehtod_choose': OrderedDict([
        ('auto',   'Auto'),
        ('gmax',   'Gmax'),
        ('bmax',   'Bmax'),
        ('derv',    'Derivative'),
        ('prev',   'Previous value'),
        ('usrdef', 'User-defined...'),
    ]),

    # track_method
    'span_track_choose': OrderedDict([
        ('auto',      'Auto'),
        ('fixspan',   'Fix span'),
        ('fixcenter', 'Fix center'),
        ('fixcntspn',  'Fix center&span'),
        # ('usrdef',    'User-defined...'),
    ]),

    # sample_channel
    'sample_channel_choose': OrderedDict([
    # key: number; val: for display in combobox
        (1, 'ADC 1'),
        (2, 'ADC 2'),
    ]),

    'ref_channel_choose': OrderedDict([
    # key: number; val: for display in combobox
        ('none', '--'),
        (1, 'ADC 1'),
        (2, 'ADC 2'),
    ]),

    'thrmcpl_choose': OrderedDict([
    # key: number; val: for display in combobox
        ('J', 'J'),
        ('K', 'K'),
        ('N', 'N'),
        ('R', 'R'),
        ('S', 'S'),
        ('T', 'T'),
        ('B', 'B'),
        ('E', 'E'),
    ]),

    'time_unit_choose': OrderedDict([
    # key: number; val: for display in combobox
        ('s', 's'),
        ('m', 'min'),
        ('h', 'h'),
        ('d', 'day'),
    ]),

    'temp_unit_choose': OrderedDict([
    # key: number; val: for display in combobox
        ('C', '°C'),
        ('K', 'K'),
        # 'F', '°F',
    ]),

    'scale_choose': OrderedDict([
    # key: number; val: for display in combobox
        ('linear', 'linear'),
        ('log'   , 'log'),
    ]),

    # available base frequency of crystals
    # key: number; val: for display in combobox
    'base_frequency_choose': OrderedDict([
        (5 , '5 MHz'),
        (6 , '6 MHz'),
        (9 , '9 MHz'),
        (10, '10 MHz'),
    ]),

    # available bandwidth limitation for each harmonic
    # key: number; val: for display in combobox
    'bandwidth_choose': OrderedDict([
        (2,  '2 MHz'),
        (1,  '1 MHz'),
        (0.5,  '0.5 MHz'),
        (0.25, '0.25 MHz'),
        (0.1, '0.1 MHz'),
    ]),

    # reference type for showing delta f and delta gamma
    # key: number; val: for display in combobox
    'ref_type_choose': OrderedDict([
        ('t0',  'First point'),
        ('t1t2',  'Selected range'),
        # ('input',  'Input value'),
        ('file', 'Other file'),
    ]),

    # steps ofr span control slider
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


    # default checkbox harm states
    'checkBox_harm1': True,
    'checkBox_harm3': True,
    'checkBox_harm5': True,
    'checkBox_harm7': False,
    'checkBox_harm9': False,
    'checkBox_harm11': False,

    # default frequency display mode
    'comboBox_settings_control_dispmode': 'startstop',
    # default time settings
    'dateTimeEdit_reftime': None,
    
    'lineEdit_recordinterval': 5,
    'lineEdit_refreshresolution': 1,
    'lineEdit_scaninterval': 5,

    # default fitting and display options
    'checkBox_dynamicfit': True,
    'spinBox_fitfactor': 6,
    'checkBox_dynamicfitbyharm': False,
    'checkBox_fitfactorbyharm': False,

    # default crystal settings
    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1,

    #NOTUSING
    'harm_set':{
        'freq_range': {

        },
        'freq_span': {

        },
        'steps': {
            1:  400, 
            3:  400, 
            5:  400, 
            7:  400, 
            9:  400, 
            11: 400, 
        },
        'span_method': {
            1:  'auto', 
            3:  'auto', 
            5:  'auto', 
            7:  'auto', 
            9:  'auto', 
            11: 'auto', 
        },
        'span_track': {
            1:  'auto', 
            3:  'auto', 
            5:  'auto', 
            7:  'auto', 
            9:  'auto', 
            11: 'auto', 
        },
        'harmfit': {
            1:  True, 
            3:  True, 
            5:  True, 
            7:  True, 
            9:  True, 
            11: True, 
        },
        'harmfitfactor': {
            1:  6, 
            3:  6, 
            5:  6, 
            7:  6, 
            9:  6, 
            11: 6, 

        },
        'peaks_maxnum': {
            1:  1, 
            3:  1, 
            5:  1, 
            7:  1, 
            9:  1, 
            11: 1, 

        },
        'peaks_threshold': {
            1:  0.2, 
            3:  0.2, 
            5:  0.2, 
            7:  0.2, 
            9:  0.2, 
            11: 0.2, 
        },
        'peaks_prominence': {
            1:  0.005, 
            3:  0.005, 
            5:  0.005, 
            7:  0.005, 
            9:  0.005, 
            11: 0.005, 
        },
    },

    # default frequency ranges for each harmonic
    'freq_range': {},
    # default frequency span for each harmonic
    'freq_span': {},

    #### default settings settings ####
    # structure: 
    # {
        # 'samp':{
        #     harm:{

        #     }
        # }
        # 'ref':{
        #     harm:{

        #     }
        # }
    # }
    'harmdata': {
        'samp':{
            1: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            3: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',   
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            5: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            7: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            9: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            11: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },
        },

        # for reference channel
        'ref':{
            1: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            3: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',   
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            5: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            7: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            9: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },

            11: {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 6,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.02,
                'lineEdit_peaks_prominence': 0.005,
            },
        },
    },
    ### default hardware settings ###
    # default VNA settings
    'comboBox_sample_channel': 1,
    'comboBox_ref_channel': 'none',

    # default crystal settings
    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1,

    # default temperature settings
    'checkBox_settings_temp_sensor': False,
    'comboBox_settings_mechanics_selectmodel': '',
    'comboBox_thrmcpltype': 'J',

    # default plots settings
    'comboBox_timeunit': 's',
    'comboBox_tempunit': 'C',
    'comboBox_timescale': 'linear',
    'comboBox_yscale': 'linear',
    'checkBox_linktime': False,

    ### default plot selections ###
    # default selections for spectra show
    'radioButton_spectra_showGp': True,
    'radioButton_spectra_showBp': False,
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

