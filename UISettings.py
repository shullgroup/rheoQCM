'''
Setting factors for GUI
Change the following factors will change the apperiance of the GUI
'''

# for create ordered dictionary 
# (dictionary in Python >=3.6 is ordered by default)
# Use OrderedDict for those dicts need to be shown in order
from collections import OrderedDict

settings_init = {

    # window default size
    'window_size': [1200, 800], # px

    # myVNA path
    'vna_path': r'C:\Program Files (x86)\G8KBB\myVNA\myVNA.exe',

    # highest harmonic can be shown in the UI. 
    'max_harmonic': 9, # do not change
    
    # time string format
    'time_str_format': '%Y-%m-%d %H:%M:%S.%f',

    'analysis_mode_disable_list':[
        'pushButton_runstop',
        # 'actionOpen_MyVNA',
        # 'actionNew_Exp',
        # 'actionClear_All',
        # 'actionLoad_Settings',

        'comboBox_samp_channel',
        'comboBox_ref_channel',
        'comboBox_base_frequency',
        'comboBox_bandwidth',
        'checkBox_settings_temp_sensor',
        'comboBox_settings_mechanics_selectmodel',
        'comboBox_tempdevice',
        'comboBox_thrmcpltype',
    ],

    # enable and disable list
    'pushButton_runstop_enable_list': [
        # 'treeWidget_settings_settings_hardware',
        'pushButton_newfile',
        'pushButton_appendfile',
        'actionNew_Exp',
        'actionLoad_Exp',
    ],

    'pushButton_runstop_disable_list': [
        # 'treeWidget_settings_settings_hardware',
        'dateTimeEdit_reftime',
        'pushButton_resetreftime',
        'pushButton_newfile',
        'pushButton_appendfile',
        'actionNew_Exp',
        'actionLoad_Exp',
    ],

    'pushButton_newfile_enable_list':[
        'dateTimeEdit_reftime',
        'pushButton_resetreftime',
    ],

    'pushButton_appendfile_disable_list':[
        'dateTimeEdit_reftime',
        'pushButton_resetreftime',
    ],

    'temp_settings_enable_disable_list': [
        'comboBox_tempmodule',
        'comboBox_tempdevice',
        'comboBox_thrmcpltype',
    ],

    'temp_device_setting_disable_list': [
        'checkBox_control_rectemp',
        'checkBox_settings_temp_sensor',
    ],

    # list for disable/hide widges for manual fit
    'manual_refit_enable_disable_list':[
        'pushButton_manual_refit',
        'pushButton_settings_data_tostart',
        'pushButton_settings_data_toprevious',
        'pushButton_settings_data_tonext',
        'pushButton_settings_data_toend',
    ],
    # list for disable/hide widges for manual fit
    'manual_refit_enable_disable_harmtree_list':[
        'lineEdit_scan_harmstart',
        'lineEdit_scan_harmend',
        'lineEdit_scan_harmsteps',
        'comboBox_tracking_condition',
        'checkBox_harmfit',
    ],

    # list for disabled widges for current version
    'version_hide_list':[
        'groupBox_settings_output',

        'groupBox_settings_fitting',

        'pushButton_settings_harm_cntr',

        'pushButton_spectra_fit_autocntr',

        'pushButton_settings_data_tostart',
        'pushButton_settings_data_toprevious',
        'pushButton_settings_data_tonext',
        'pushButton_settings_data_toend',

        # 'toolButton_settings_mechanics_solve',
        # 'groupBox_nhplot',
        # 'groupBox_settings_mechanics_nhcalc',
        # 'checkBox_settings_mechanics_witherror',
        'pushButton_settings_mechanics_errorsettings',
        # 'label_6',
        # 'comboBox_settings_mechanics_refG',
        # 'comboBox_settings_mechanics_selectmodel',
        'tableWidget_settings_mechanics_setmodel',

        # 'groupBox_settings_mechanics_contour',
        # 'pushButton_settings_mechanics_simulator',
        # 'groupBox_settings_mechanics_simulator',
    ],

    # list of widges to delete in current version
    'version_delete_list':[
        'tab_settings_mechanics',
    ],

        
    # default open/save data file type
    'default_datafiletype': ';;'.join([
        # 'Json file (*.json)',
        'hdf5 file (*.h5)',
    ]),

    # default load data file type
    'default_settings_load_filetype': ';;'.join([
        'hdf5 file (*.h5)',
        'Json file (*.json)',
    ]),

    # default export data file type
    'default_settings_export_filetype': ';;'.join([
        'Json file (*.json)',
    ]),

    # export  data file type
    'export_datafiletype': ';;'.join([
        'csv file (*.csv)',
        'excel file (*.xlsx)',
        'json file (*.json)',
        # 'hdf5 file (*.h5)',
        # 'Python file (*.py)',
        # 'Matlab file (*.mat)',

    ]),

    # scan mode
    'display_opts': OrderedDict([
        ('startstop',  'Start/Stop'),
        ('centerspan', 'Center/Span'),
    ]),

    # choice for plotting data shown in comboBox_plt1_optsy & comboBox_plt2_optsy
    # 'data_plt_opts': OrderedDict([
    #     ('none',   'none'),
    #     ('df_t',   u'\u0394' + 'f - time'),
    #     ('dfn_t',  u'\u0394' + 'f/n - time'),
    #     ('mdf_t',  '-' + u'\u0394' + 'f - time'),
    #     ('mdfn_t', '-' + u'\u0394' + 'f/n - time'),
    #     ('dg_t',   u'\u0394\u0393' + ' - time'),
    #     ('dgn_t',  u'\u0394\u0393' + '/n - time'),
    #     ('f_t',    'f - time'),
    #     ('g_t',    'g - time'),
    #     ('temp_t', 'temp. - time'),
    # ]),
    'data_plt_opts': OrderedDict([
        # ('none',   'none'),
        ('df',   u'\u0394' + 'f'),
        ('dfn',  u'\u0394' + 'f/n'),
        ('mdf',  '-' + u'\u0394' + 'f'),
        ('mdfn', '-' + u'\u0394' + 'f/n'),
        ('dg',   u'\u0394\u0393'),
        ('dgn',  u'\u0394\u0393' + '/n'),
        ('f',    'f'),
        ('g',     u'\u0393'),
        ('temp', 'temp.'),
        ('t', 'time'),
        ('idx', 'index'),
    ]),

    'data_plt_axis_label': {
        'df':   r'$\Delta$f (Hz)',
        'dfn':  r'$\Delta$f/n (Hz)', 
        'mdf':  r'-$\Delta$f (Hz)', 
        'mdfn': r'-$\Delta$f/n (Hz)', 
        'dg':   r'$\Delta\Gamma$ (Hz)',
        'dgn':  r'$\Delta\Gamma$/n (Hz)',
        'f':    r'f (Hz)',
        'g':    r'$\Gamma$ (Hz)',
        'temp': r'Temp. (unit)', # unit is going to be replaced by temperature unit
        't':    r'Time (unit)', # unit is going to be replaced by time unit
        'idx':  r'Index', 

        # for property
        'delfs':        r'$\Delta$f (Hz)',
        'delf_calcs':   r'$\Delta$f$_{calc}$ (Hz)',
        'delgs':        r'$\Delta\Gamma$ (Hz)',
        'delg_calcs':   r'$\Delta\Gamma_{calc}$ (Hz)',
        'drho':         r'd$\rho$ (g/cm$^2$)',
        'grho_rh':      r'$|G_{rh}^*|\rho$ (Pa $\cdot$ g/cm$^3$)',
        'phi':          r'$\phi$ ($\degree$)',
        'dlam_rh':      r'd/$\lambda_{rh}$',
        'lamrho':       r'$\lambda\rho$ (g/cm$^2$)',
        'delrho':       r'$\delta\rho$ (g/cm$^2$)',
        'delf_delfsn':  r'$\Delta$f/$\Delta$f$_{sn}$',
        'delg_delfsns': r'$\Delta\Gamma$/$\Delta$f$_{sn}$',
        'rh':           r'r$_h$',
        'rds':          r'r$_d$',
    },

    # spinBox_harmfitfactor max value
    'fitfactor_max': 20, # int

    # comboBox_tracking_method
    'span_mehtod_opts': OrderedDict([
        ('auto',   'Auto'),
        ('gmax',   'Gmax'),
        ('bmax',   'Bmax'),
        ('derv',    'Derivative'),
        ('prev',   'Previous value'),
        # ('usrdef', 'User-defined...'),
    ]),

    # track_method
    'span_track_opts': OrderedDict([
        ('auto',      'Auto'),
        ('fixspan',   'Fix span'),
        ('fixcenter', 'Fix center'),
        ('fixcntspn',  'Fix center&span'),
        # ('usrdef',    'User-defined...'),
    ]),

    # sample_channel
    'vna_channel_opts': OrderedDict([
    # key: str(number); val: for display in combobox
        ('none', '--'),
        ('1', 'ADC 1'),
        ('2', 'ADC 2'),
    ]),

    'ref_channel_opts': OrderedDict([
    # key: str(number); val: for display in combobox
        # ('none', 'none'),
        ('samp', 'S chn.'),
        ('ref', 'R chn.'),
        ('ext', 'ext'), # always read from the reference channel
    ]),

    'thrmcpl_opts': OrderedDict([
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

    'time_unit_opts': OrderedDict([
    # key: number; val: for display in combobox
        ('s', r's'),
        ('m', r'min'),
        ('h', r'h'),
        ('d', r'day'),
    ]),

    'temp_unit_opts': OrderedDict([
    # key: number; val: for display in combobox
        ('C', '°C'),
        ('K', 'K'),
        ('F', '°F'),
    ]),

    'scale_opts': OrderedDict([
    # key: number; val: for display in combobox
        ('linear', 'linear'),
        ('log'   , 'log'),
    ]),

    # available base frequency of crystals
    # key: number; val: for display in combobox
    'base_frequency_opts': OrderedDict([
        ('5' , '5 MHz'),
        ('6' , '6 MHz'),
        ('9' , '9 MHz'),
        ('10', '10 MHz'),
    ]),

    # available bandwidth limitation for each harmonic
    # key: number; val: for display in combobox
    'bandwidth_opts': OrderedDict([
        ('0.1', '0.1 MHz'),
        ('0.25', '0.25 MHz'),
        ('0.5',  '0.5 MHz'),
        ('1',  '1 MHz'),
        ('2',  '2 MHz'),
    ]),

    # # reference type for showing delta f and delta gamma
    # # key: number; val: for display in combobox
    # 'ref_type_opts': OrderedDict([
    #     ('t0',  'First point'),
    #     ('t1t2',  'Selected range'),
    #     # ('input',  'Input value'),
    #     ('file', 'Other file'),
    # ]),

    # options for comboBox_settings_mechanics_selectmodel
    'qcm_model_opts': {
        'onelayer': 'One layer',
        'twolayers': 'Two layers',
        'bulk': 'Bulk',
        # 'multiple': 'Multiple models',
    },

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
    # cap size for prop figures
    'mpl_capsize': 4,

    # progressBar settings
    # the programe will try to separate the scan_interval to 
    # steps of 'progressbar_update_steps' to update progressbar.
    # if the time interval smaller than 'progressbar_min_interval', 
    # it will update after every 'progressbar_min_interval' time
    # if the time interval bigger than 'progressbar_max_interval',
    # it will update after every 'progressbar_max_interval'
    'progressbar_update_steps': 100, 
    'progressbar_min_interval': 100, # in ms
    'progressbar_max_interval': 1000, # in ms

    # rowheader for tableWidget_spectra_mechanics_table
    # DON't change the value of this key
    'mech_table_rowheaders':{
        'delfs':         u'\u0394' + 'f (Hz)', # Δf (Hz)
        'delf_calcs':     u'\u0394' + 'fcalc (Hz)', # Δfcalc (Hz)
        'delgs':         u'\u0394\u0393' + ' (Hz)', # ΔΓ (Hz)
        'delg_calcs':    u'\u0394\u0393' + 'calc (Hz)', # ΔΓcalc (Hz)
        'drho':         'd' + u'\u03C1' + ' (g/cm'+ u'\u00B2' + ')', # dρ (g/cm²)
        'grho_rh':      '|G*|' + u'\u03C1' + ' (Pa' + u'\u2219' + 'g/cm' + u'\u00B3' + ')', # |G*|ρ (Pa∙g/cm³)
        'phi':          u'\u03A6' + ' (' + u'\u00B0' + ')', # Φ (°)
        'dlam_rh':      'd/' + u'\u03BB\u2099', # d/λₙ
        'lamrho':       u'\u03BB\u03C1' + ' (g/cm' + u'\u00B2' + ')', # λρ (g/cm²)
        'delrho':       u'\u03B4\u03C1' + ' (g/cm' + u'\u00B2' + ')', # δρ (g/cm²)
        'delf_delfsn':  u'\u0394' + 'f/' + u'\u0394' + 'f' + u'\u209B\u2099', # Δf/Δfₛₙ
        'delg_delfsns': u'\u0394\u0393' + '/' + u'\u0394' + 'f' + u'\u209B\u2099', # ΔΓ/Δfₛₙ
        'rh':           'rh',
        'rds':           'rd',
        # 't':         'Time (s)', # Time (s)
        # 'temp':         'Temp. (' + u'\u00B0' + 'C)', # Temp. (°C)
    },

    'prop_plot_minmum_row_height': 300, # height of property figure when plotted in line


    ############ params for temperature modules ###########
    # temperature modules path
    'tempmodules_path': r'./modules/temp/', 
    
    # add NI sensors into the dict and the code will check if the devices in its keys.
    # the values are the number of samples per test for average
    'tempdevices_dict': {
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

    'tempdevs_opts': {}, # this key will be updated while running and for the updating of 'comboBox_tempdevice'

    'temp_class_opts_list': [], # this key will be updated while running and for the updating of 'comboBox_tempmodule'

    ######## params for PekTracker module #########
    # minium distance between the peakes to be found in Hz
    'peak_min_distance_Hz': 1e3, 
    # minium fwhw fo peaks to be found in HZ
    'peak_min_width_Hz': 10, 
    # tolerance for peak fitting 
    'xtol': 1e-10, # -18
    'ftol': 1e-10, # -18

    ######### params for DataSaver module #########
    'unsaved_path': r'.\unsaved', 

}

#####################################################


#####################################################

settings_default = {
#### default settings control ####
    
    # # highest harmonic to display MUST <= settings_init['max_harmonic']
    # 'max_disp_harmonic': 9, 


    # default checkbox harm states
    'checkBox_harm1': True,
    'checkBox_harm3': True,
    'checkBox_harm5': True,
    'checkBox_harm7': False,
    'checkBox_harm9': False,
    # 'checkBox_harm11': False,

    # default frequency display mode
    'comboBox_settings_control_dispmode': 'centerspan',
    # default time settings
    # NOTE: keep this key commented
    # 'dateTimeEdit_reftime': '2000-01-01 00:00:00.000',
    
    'lineEdit_recordinterval': 5,
    'lineEdit_refreshresolution': 1,
    'lineEdit_scaninterval': 5,

    # default fitting and display options
    'checkBox_dynamicfit': True,
    'spinBox_fitfactor': 6,
    'checkBox_dynamicfitbyharm': False,
    'checkBox_fitfactorbyharm': False,

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
            1:  3, 
            3:  3, 
            5:  3, 
            7:  3, 
            9:  3, 
            11: 3, 

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
            1:  0.001, 
            3:  0.001, 
            5:  0.001, 
            7:  0.001, 
            9:  0.001, 
            11: 0.001, 
        },
        'peaks_prominence': {
            1:  0.001, 
            3:  0.001, 
            5:  0.001, 
            7:  0.001, 
            9:  0.001, 
            11: 0.001, 
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
            '1': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '3': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',   
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '5': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '7': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '9': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            # 11: {
            #     # default scan settings
            #     'lineEdit_scan_harmsteps': 400,
            #     # default span settings
            #     'comboBox_tracking_method': 'auto',
            #     'comboBox_tracking_condition': 'auto',
            #     # default fit settings
            #     'checkBox_harmfit': True,
            #     'spinBox_harmfitfactor': 3,
            #     'spinBox_peaks_num': 1, 
            #     'radioButton_peaks_num_max': True,
            #     'radioButton_peaks_num_fixed': False,
            #     'radioButton_peaks_policy_minf': False,
            #     'radioButton_peaks_policy_maxamp': True,
            #     'lineEdit_peaks_threshold': 0.001,
            #     'lineEdit_peaks_prominence': 0.001,
            # },
        },

        # for reference channel
        'ref':{
            '1': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '3': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',   
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '5': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '7': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            '9': {
                # default scan settings
                'lineEdit_scan_harmsteps': 400,
                # default span settings
                'comboBox_tracking_method': 'auto',
                'comboBox_tracking_condition': 'auto',
                # default fit settings
                'checkBox_harmfit': True,
                'spinBox_harmfitfactor': 3,
                'spinBox_peaks_num': 1, 
                'radioButton_peaks_num_max': True,
                'radioButton_peaks_num_fixed': False,
                'radioButton_peaks_policy_minf': False,
                'radioButton_peaks_policy_maxamp': True,
                'lineEdit_peaks_threshold': 0.001,
                'lineEdit_peaks_prominence': 0.001,
            },

            # 11: {
            #     # default scan settings
            #     'lineEdit_scan_harmsteps': 400,
            #     # default span settings
            #     'comboBox_tracking_method': 'auto',
            #     'comboBox_tracking_condition': 'auto',
            #     # default fit settings
            #     'checkBox_harmfit': True,
            #     'spinBox_harmfitfactor': 3,
            #     'spinBox_peaks_num': 1, 
            #     'radioButton_peaks_num_max': True,
            #     'radioButton_peaks_num_fixed': False,
            #     'radioButton_peaks_policy_minf': False,
            #     'radioButton_peaks_policy_maxamp': True,
            #     'lineEdit_peaks_threshold': 0.001,
            #     'lineEdit_peaks_prominence': 0.001,
            # },
        },
    },
    ### default hardware settings ###
    # 'tabWidget_settings_settings_samprefchn': 1,
    # default VNA settings
    'comboBox_samp_channel': '1',
    'comboBox_ref_channel': 'none',

    # default crystal settings
    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1,

    # default temperature settings
    'checkBox_settings_temp_sensor': False,
    'comboBox_settings_mechanics_selectmodel': '',
    'comboBox_thrmcpltype': 'J',

    # default plots settings
    'comboBox_timeunit': 'm',
    'comboBox_tempunit': 'C',
    'comboBox_xscale': 'linear',
    'comboBox_yscale': 'linear',
    'checkBox_linkx': False,

    ### default plot selections ###
    # default selections for spectra show
    'radioButton_spectra_showGp': True,
    'radioButton_spectra_showBp': False,
    'radioButton_spectra_showpolar': False,
    'checkBox_spectra_showchi': False,

    # default selections for plot 1 elements
    'comboBox_plt1_optsy': 'dfn', 
    'comboBox_plt1_optsx': 't', 
    'checkBox_plt1_h1': False,
    'checkBox_plt1_h3': False,
    'checkBox_plt1_h5': False,
    'checkBox_plt1_h7': False,
    'checkBox_plt1_h9': False,
    'checkBox_plt1_h11': False,
    'radioButton_plt1_samp': True,
    'radioButton_plt1_ref': False,

    # default selections for plot 2 elements
    'comboBox_plt2_optsy': 'dg',
    'comboBox_plt2_optsx': 't',
    'checkBox_plt2_h1': False,
    'checkBox_plt2_h3': False,
    'checkBox_plt2_h5': False,
    'checkBox_plt2_h7': False,
    'checkBox_plt2_h9': False,
    'checkBox_plt2_h11': False,
    'radioButton_plt2_samp': True,
    'radioButton_plt2_ref': False,

    ### settings_data
    'radioButton_data_showall': True,
    'radioButton_data_showmarked': False,
    'comboBox_settings_data_samprefsource': 'samp',
    'lineEdit_settings_data_samprefidx': [0],
    'comboBox_settings_data_refrefsource': 'ref',
    'lineEdit_settings_data_refrefidx': [0],

    ### settings_mech
    'checkBox_settings_mech_liveupdate': True,
    'checkBox_nhplot1': True,
    'checkBox_nhplot3': True,
    'checkBox_nhplot5': True,
    'checkBox_nhplot7': False,
    'checkBox_nhplot9': False,

    'spinBox_settings_mechanics_nhcalc_n1': 3,
    'spinBox_settings_mechanics_nhcalc_n2': 5,
    'spinBox_settings_mechanics_nhcalc_n3': 3,

    'comboBox_settings_mechanics_refG': '3', # reference harmonic for property
    'checkBox_settings_mechanics_witherror': True, # errorbar

    'comboBox_settings_mechanics_selectmodel': 'onelayer',
    
}

