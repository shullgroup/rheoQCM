'''
Setting factors for GUI
Change the following factors will change the apperiance of the GUI
'''

# for create ordered dictionary 
# (dictionary in Python >=3.6 is ordered by default)
# Use OrderedDict for those dicts need to be shown in order
from collections import OrderedDict
import numpy as np

settings_init = {

    # window default size
    'window_size': [1200, 800], # px

    # UI will looking for the file to load the default setup
    'default_settings_file_name': 'user_settings.json',

    # myVNA path
    'vna_path': [
        r'C:\Program Files (x86)\G8KBB\myVNA\myVNA.exe',
        r'C:\Program Files\G8KBB\myVNA\myVNA.exe',
    ],

    # where the calibration files saved (not necessary)
    'vna_cal_file_path': r'./cal/', 

    # highest harmonic can be shown in the UI. 
    'max_harmonic': 9, # MUST >= 1
    
    # time string format
    'time_str_format': '%Y-%m-%d %H:%M:%S.%f',

    # if marked data shown when showing all data
    'show_marked_when_all': True,

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
        # hide reference time widgets to simplify the setup
        # reference time can always be changed by shifted_t0
        'dateTimeEdit_reftime',
        'pushButton_resetreftime',
        'label_reftime',

        'groupBox_settings_output',

        'groupBox_settings_fitting',

        'pushButton_settings_harm_cntr',

        'pushButton_spectra_fit_autocntr',

        'pushButton_settings_data_tostart',
        'pushButton_settings_data_toprevious',
        'pushButton_settings_data_tonext',
        'pushButton_settings_data_toend',
        
        'frame_settings_data_tempref',
        'comboBox_settings_data_ref_crystmode',
        'comboBox_settings_data_ref_tempmode',
        'comboBox_settings_data_ref_fitttype',

        # 'toolButton_settings_mechanics_solve',
        # 'groupBox_nhplot',
        # 'groupBox_settings_mechanics_nhcalc',
        # 'checkBox_settings_mechanics_witherror',
        'pushButton_settings_mechanics_errorsettings',
        # 'label_6',
        # 'comboBox_settings_mechanics_refG',
        # 'comboBox_settings_mechanics_selectmodel',
        'groupBox_settings_mechanics_mech_film',

        # 'groupBox_settings_mechanics_contour',
        # 'pushButton_settings_mechanics_simulator',
        # 'groupBox_settings_mechanics_simulator',

        'actionSolve_test',

        # statusbar
        'pushButton_status_reftype',
        # 'pushButton_status_signal_ch',

    ],

    # list of widges to delete in current version
    'version_delete_list':[
        # 'tab_settings_mechanics',
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

    # export data file type
    'export_datafiletype': ';;'.join([
        # 'csv file (*.csv)',
        'excel file (*.xlsx)',
        # 'json file (*.json)',
        # 'hdf5 file (*.h5)',
        # 'Matlab file (*.mat)',
    ]),

    # export raw data file type
    'export_rawfiletype': ';;'.join([
        'csv file (*.csv)',
        'excel file (*.xlsx)',
        'json file (*.json)',
    ]),

    # import QCM-D data file type
    'external_qcm_datafiletype': ';;'.join([
        'excel file (*.xlsx)',
        'csv file (*.csv)',
        # 'json file (*.json)',
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
        ('id', 'test ID')
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
        'id':  r'Test ID', # queue_id
        'idx':  r'Index', # dataframe index

        # for property
        'delf_calcs':        r'$\Delta$f (Hz)',
        'delf_exps':         r'$\Delta$f$_{exp}$ (Hz)',
        'delg_calcs':        r'$\Delta\Gamma$ (Hz)',
        'delg_exps':         r'$\Delta\Gamma_{exp}$ (Hz)',
        'drho':              r'd$\rho$ ($\mu$m$\cdot$g/cm$^3$)',
        'grho_rh':           r'$|G_{rh}^*|\rho$ (Pa$\cdot$g/cm$^3$)',
        'phi':               r'$\phi$ ($\degree$)',
        'dlam_rh':           r'd/$\lambda_{rh}$',
        'lamrho':            r'$\lambda\rho$ ($\mu$m$\cdot$g/cm$^3$)',
        'delrho':            r'$\delta\rho$ ($\mu$m$\cdot$g/cm$^3$)',
        'delf_delfsn':       r'$\Delta$f/$\Delta$f$_{sn}$',
        'delg_delfsn_exps':  r'$(\Delta\Gamma$/$\Delta$f$_{sn})_{exp}$',
        'delg_delfsn_calcs': r'$\Delta\Gamma$/$\Delta$f$_{sn}$',
        'rh':                r'r$_h$',
        'rd_exps':           r'r$_{d,exp}$',
        'rd_calcs':          r'r$_d$',
    },

    # rowheader for tableWidget_spectra_mechanics_table
    # DON't change the value of this key
    'mech_table_rowheaders':{
        'delf_exps':         u'\u0394' + 'f (Hz)', # Δf (Hz)
        'delf_calcs':        u'\u0394' + 'fcalc (Hz)', # Δfcalc (Hz)
        'delg_exps':         u'\u0394\u0393' + ' (Hz)', # ΔΓ (Hz)
        'delg_calcs':        u'\u0394\u0393' + 'calc (Hz)', # ΔΓcalc (Hz)
        'drho':              'd' + u'\u03C1' + ' (' + u'\u03BC' + 'm' + u'\u2219' 'g/cm'+ u'\u00B3' + ')', # dρ (μm∙g/m³)
        'grho_rh':           '|G*|' + u'\u03C1' + ' (Pa' + u'\u2219' + 'g/cm' + u'\u00B3' + ')', # |G*|ρ (Pa∙g/cm³)
        'phi':                u'\u03A6' + ' (' + u'\u00B0' + ')', # Φ (°)
        'dlam_rh':           'd/' + u'\u03BB\u2099', # d/λₙ
        'lamrho':            u'\u03BB\u03C1' + ' (' + u'\u03BC' + 'm' + u'\u2219' 'g/cm'+ u'\u00B3' + ')', # λρ (μm∙g/m³)
        'delrho':            u'\u03B4\u03C1' + ' (' + u'\u03BC' + 'm' + u'\u2219' 'g/cm'+ u'\u00B3' + ')', # δρ (μm∙g/m³)
        'delf_delfsn':       u'\u0394' + 'f/' + u'\u0394' + 'f' + u'\u209B\u2099', # Δf/Δfₛₙ
        'delg_delfsn_calcs': u'\u0394\u0393' + '/' + u'\u0394' + 'f' + u'\u209B\u2099', # ΔΓ/Δfₛₙ
        'rh':                'rh',
        'rd_calcs':          'rd',
        't':                 'Time (s)', # Time (s)
        'temp':              'Temp. (' + u'\u00B0' + 'C)', # Temp. (°C)
    }, # add the colum when activate column here

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
        # ('ext', 'ext'), # always read from the reference channel
    ]),

    # the value here is the same as the value of 'kind'
    # of scipy.interpolate.interp1d()
    'ref_interp_opts': OrderedDict({
        'linear': 'linear', 
        'nearest': 'nearest', 
        'zero': 'zero', 
        'slinear': 'slinear', 
        'quadratic': 'quadratic', 
        'cubic': 'cubic',
    }),

    'ref_crystal_opts': OrderedDict({
        'single': 'Single',
        'dual': 'Dual',
        # '': '',
    }),

    'ref_temp_opts': OrderedDict({
        'const': 'Constant T',
        'var': 'Variable T',
        # '': '',
    }),

    # crystal cuts options
    'crystal_cut_opts':{
        'AT': 'AT',
        'BT': 'BT',
    },

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

    # minimum/maximum layer for mechanic property calculation
    'min_mech_layers': 0,
    'max_mech_layers': 1,

    # options for comboBox_settings_mechanics_selectmodel
    'qcm_model_opts': {
        'onelayer': 'One layer',
        # 'twolayers': 'Two layers',
        'bulk': 'Bulk',
        # 'multiple': 'Multiple models',
    },

    # calctype
    'calctype_opts':{
        'SLA': 'SLA',
        'LL': 'LL',
    },

    'qcm_layer_known_source_opts': {
        'ind': 'Index',
        'prop': 'Prop.',
        'fg': u'\u0394' + 'f&' u'\u0394\u0393',
        'name': 'Name',
        # 'none': '--',
    },

    'qcm_layer_unknown_source_opts': {
        'none': '--',
        'ind': 'Index',
        'guess': 'Guess',
    },

    'qcm_layer_bulk_name_opts': {
        'air': {
            'drho': np.inf,
            'grho': 0,
            'phi': np.pi / 2,
            'rh': 3,
        },
        'water': {
            'drho': np.inf, 
            'grho': 1e5, # in Pa
            'phi': np.pi /2,
            'rh': 3,
        },
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
    # text size for normal figures
    'mpl_txtfontsize': 8,
    # text size in mpl_sp figures
    'mpl_sp_txtfontsize': 5,
    # text (harmonic number) size in mpl_sp figures
    'mpl_sp_harmfontsize': 9,

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


    'prop_plot_minmum_row_height': 300, # height of property figure when plotted in line


    ############ params for temperature modules ###########
    # # temperature modules path
    # 'tempmodules_path': r'./modules/temp/', 
    
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
    'unsaved_path': r'.\unsaved_data', 
    'unsaved_filename': r'%Y%m%d%H%M%S',

    ######### DataSaver module: import data format #####
    # a number is going to replace '{}'
    'data_saver_import_data':{
        't': ['time(s)', 'time (s)', 't(s)'],
        'temp': ['temp(c)', 'temp'],
        'fs': ['freq{}'],
        'gs': ['gamma{}'],
        'delfs': ['delf{}'], 
        'delgs': ['delg{}'],
        # 'harm_number_org': '1', 
    },
    'data_saver_import_raw':{
        't': ['time(s)', 't(s)', 't'],
        'temp': ['temp', 'temp(c)'],
        'f': ['{} Sweep [MHz]'],
        'G': ['{} G [S]'],
        'B': ['{} B [S]'],
        # 'D': ['D(Hz)'],
        # 'harm_number_org': '5', 
    },


    ######### params for PeakTracker module #######
    'cen_range': 0.05, # peak center limitation (cen of span +/- cen_range*span)
    'big_move_thresh': 1.5, # if the peak out abs(cen - span_cen) > 'big_move_thresh' * 'cen_range' * span, is is considered as a big move. center will be moved to the opposite side.
    'wid_ratio_range': (8, 20), # width_ratio[0]*HWHM <= span <= width_ratio[1]*HWHM
    'change_thresh': (0.05, 0.5), # span change threshold. the value should be between (0, 1)
    # min(
    #     max(
    #         wid_ratio_range[1/0] * half_wid * 2,
    #         current_span * (1 - change_thresh[1/0]),
    #         ), # lower bound of wid_ratio_range
    #     current_span * (1 - change_thresh[0/1]) # 
    # )

}



#####################################################









#####################################################

settings_default = {
#### default settings control ####

    # add na_path on your computer if it is not in the 
    # default path listed in settings_init['vna_path']
    # add the string as 
    # 'vna_path': r'C:/...../myVNA.exe'.
    # if this key is empty, the program will look for the file in the default list in settings_init['vna_path']
    'vna_path': r'',

    # default checkbox harm states (checkbox with False value can be omit from this list)
    'checkBox_harm1': True,
    'checkBox_harm3': True,
    'checkBox_harm5': True,

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

    # default sampe discription
    'plainTextEdit_settings_samplediscription': '',

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
        'samp':{},
        # for reference channel
        'ref':{},
    },

    'mechchndata': {
        'samp': {}, 
        'ref': {}
    }, # dictionary for saving the widgets for layers defination

    ### default hardware settings ###
    # 'tabWidget_settings_settings_samprefchn': 1,
    # default VNA settings
    'comboBox_samp_channel': '1',
    'comboBox_ref_channel': 'none',

    # default crystal settings
    'comboBox_base_frequency': 5,
    'comboBox_bandwidth': 0.1,
    'comboBox_crystalcut': 'AT',

    # default temperature settings
    'checkBox_settings_temp_sensor': False,
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
    # (checkbox with False value can be omit from this list)
    'checkBox_plt1_h1': True,
    'checkBox_plt1_h3': True,
    'checkBox_plt1_h5': True,
    'radioButton_plt1_samp': True,
    'radioButton_plt1_ref': False,

    # default selections for plot 2 elements
    'comboBox_plt2_optsy': 'dg',
    'comboBox_plt2_optsx': 't',
    # (checkbox with False value can be omit from this list)
    'checkBox_plt2_h1': True,
    'checkBox_plt2_h3': True,
    'checkBox_plt2_h5': True,
    'radioButton_plt2_samp': True,
    'radioButton_plt2_ref': False,

    ### settings_data
    'radioButton_data_showall': True,
    'radioButton_data_showmarked': False,
    'comboBox_settings_data_samprefsource': 'samp',
    'lineEdit_settings_data_samprefidx': [0],
    'comboBox_settings_data_refrefsource': 'ref',
    'lineEdit_settings_data_refrefidx': [0],
    'comboBox_settings_data_ref_crystmode': 'single',
    'comboBox_settings_data_ref_tempmode': 'const',
    'comboBox_settings_data_ref_fitttype': 'linear',

    ### settings_mech
    'checkBox_settings_mech_liveupdate': True,
    'checkBox_nhplot1': True,
    'checkBox_nhplot3': True,
    'checkBox_nhplot5': True,

    'spinBox_settings_mechanics_nhcalc_n1': 3,
    'spinBox_settings_mechanics_nhcalc_n2': 5,
    'spinBox_settings_mechanics_nhcalc_n3': 3,

    'comboBox_settings_mechanics_refG': '3', # reference harmonic for property
    'spinBox_mech_expertmode_layernum': 1, # number of layers for expert mode mechanic 

    'comboBox_settings_mechanics_calctype': 'LL', # 'LL' or 'SLA'

    'checkBox_settings_mechanics_witherror': True, # errorbar

    'comboBox_settings_mechanics_selectmodel': 'onelayer',
}


# initial harm_tree value for a single harmonic 
harm_tree = {
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
    'checkBox_settings_settings_harmzerophase': False,
    'lineEdit_peaks_threshold': 0.0001,
    'lineEdit_peaks_prominence': 0.0001,
}

# set harmdata value
for harm in range(1, settings_init['max_harmonic']+2, 2):
    harm = str(harm)
    settings_default['harmdata']['samp'][harm] = harm_tree.copy()
    settings_default['harmdata']['ref'][harm] = harm_tree.copy()
