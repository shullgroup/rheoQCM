#!/usr/bin/env python
'''
This is the main code of the QCM acquization program
'''

import os
import sys
import subprocess
import traceback
import copy

import threading
import multiprocessing 

import logging
import logging.config
# Get the logger specified in the file
logger = logging.getLogger(__name__)
# logger = logging.getLogger('infoLogger')
# logger = logging.getLogger('fileLogger')
# logger.setLevel(logging.ERROR)

# import csv
# import importlib
import math
import json
import shutil
import datetime, time
import numpy as np
import pandas as pd
import scipy.signal
# from collections import OrderedDict
# import types
from PyQt5.QtCore import pyqtSlot, Qt, QEvent, QTimer, QEventLoop, QCoreApplication, QSize, qFatal, QT_VERSION 
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QRadioButton, QMenu, QAction, QMessageBox, QTableWidgetItem, QSizePolicy, QFrame, QLabel, QPlainTextEdit
)
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QValidator, QIntValidator, QDoubleValidator, QRegExpValidator

# packages
from MainWindow import Ui_MainWindow # UI from QT5
import UISettings # UI basic settings module
from UISettings import harm_tree as harm_tree_default


print(os.getcwd())


config_default = UISettings.get_config()
settings_default = UISettings.get_settings()

# copy some data related sets from settigs_init to settings_default if not exist
if 'max_harmonic' not in settings_default:
    settings_default['max_harmonic'] = config_default['max_harmonic']
if 'time_str_format' not in settings_default:
    settings_default['time_str_format'] = config_default['time_str_format']
if 'vna_path' not in settings_default:
    settings_default['vna_path'] = config_default['vna_path']


# packages from program itself
from modules import UIModules, PeakTracker, DataSaver
from modules import QCM as QCM 
from modules.MatplotlibWidget import MatplotlibWidget

import _version
print('Version: {}'.format(_version.__version__))


if UIModules.system_check() == 'win32': # windows
    import struct
    if struct.calcsize('P') * 8 == 32: # 32-bit version Python
        try:
            # from modules.AccessMyVNA_dummy import AccessMyVNA
            from modules.AccessMyVNA import AccessMyVNA
            logger.info(AccessMyVNA) 
            # test if MyVNA program is available
            with AccessMyVNA() as vna:
                if vna.Init() == 0: # connection with myVNA is available
                    try:
                        from modules import TempDevices,TempModules
                    except Exception as e:
                        logger.warning('Failed to import TempDevices and/or TempModules.\nTemperature functions of the UI will not avaiable!')
                        logger.exception('Failed to import TempDevices and/or TempModules.')

        except Exception as e: # no myVNA connected. Analysis only
            logger.error('Failed to import AccessMyVNA module!')
            logger.exception('Failed to import AccessMyVNA module!')
    else: # 64-bit version Python which doesn't work with AccessMyVNA
        # A 32-bit server may help 64-bit Python work with 32-bit dll
        logger.warning('Current version of MyVNA does not work with 64-bit Python!\nData analysis only!')
else: # linux or MacOS
    # for test only
    # from modules.AccessMyVNA_dummy import AccessMyVNA
        logger.warning('Current version of MyVNA does not work with MacOS and Linux!\nData analysis only!')


############# end of importing modules ####################


def setup_logging():
    '''Setup logging configuration'''

    try:
        if os.path.exists(config_default['logger_setting_config']['config_file']): # use json file
            with open(config_default['logger_setting_config']['config_file'], 'r') as f:
                logger_config = json.load(f)
        else: # default setting
                logger_config = config_default['logger_config']
        logging.config.dictConfig(logger_config)
    except Exception as e:
        print('Setting logger failed! Use default logging level!')
        logging.basicConfig(level=config_default['logger_setting_config']['default_level'])
        print(e)


class VNATracker:
    def __init__(self):
        self.f =None       # current end frequency span in Hz (ndarray([float, float])
        self.steps = None   # current number of steps (int)
        self.chn = None     # current reflection ADC channel (1 or 2)
        self.avg = None     # average of scans (int)
        self.speed = None   # vna speed set up (int 1 to 10)
        self.instrmode = 0  # instrument mode (0: reflection)
        self.cal = self.get_cal_filenames()

        self.setflg = {} # if vna needs to reset (set with reset selections)
        self.setflg.update(self.__dict__) # get all attributes in a dict
        self.setflg.pop('setflg', None) # remove setflg itself
        logger.info('setflg: %s', self.setflg) 


    def get_cal_filenames(self):
        '''
        find calc file for ADC1 and ADC2 separately
        The fill should be stored in config_default['vna_cal_file_path'] for each channel
        '''
        cal = {'ADC1': '', 'ADC2': ''}
        if (UIModules.system_check() == 'win32') and (struct.calcsize('P') * 8 == 32): # windows (if is win32, struct will already be imported above)
            vna_cal_path = os.path.abspath(config_default['vna_cal_file_path'])
            if not os.path.isdir(vna_cal_path): # directory doesn't exist
                os.makedirs(vna_cal_path) # create directory
            else:
                files = os.listdir(vna_cal_path) # list all file in the given folder
                logger.info('cal folder: %s', files) 
                for key in cal.keys():
                    for file in files:
                        if (key + '.myVNA.cal').lower() in file.lower():
                            cal[key] = os.path.join(vna_cal_path, file) # use absolute path
                            break
                logger.info(cal) 
        return cal


    def set_check(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray): # self.f
                val = val.tolist()
                # if not np.array_equal(val, getattr(self, key)): # if self.<key> changed
                #     setattr(self, key, val) # save val to class
                # self.setflg[key] = val # add set key and value to setflg
            # else:
            if getattr(self, key) != val: # if self.<key> changed
                setattr(self, key, val) # save val to class
            self.setflg[key] = val # add set key and value to setflg
        return self.setflg


    def reset_flag(self):
        ''' set to vna doesn't neet rest '''
        self.setflg = {}


def mp_solve_single_queue(ind, qcm_queue, mech_queue, prop_dict_ind, nh, calctype, bulklimit, qcm):
    
    mech_queue['queue_id'] = mech_queue['queue_id'].astype('int')

    # obtain the solution for the properties
    if qcm.all_nhcaclc_harm_not_na(nh, qcm_queue):
        # solve a single queue
        mech_queue = qcm.solve_single_queue(nh, qcm_queue, mech_queue, calctype=calctype, film=prop_dict_ind, bulklimit=bulklimit)

        # save back to mech_df
        mech_queue.index = [ind] # not necessary
        # mech_df.update(mech_queue)
        # mech_df['queue_id'] = mech_df['queue_id'].astype('int')

        return mech_queue
    else:
        # since the df already initialized with nan values, nothing to do here
        return None


def mp_solve_single_queue_to_prop(ind, qcm_queue, nh, calctype, bulklimit, refh, brief_report, solve_single_queue_to_prop):
    '''
    used to solve the known layers to prop
    '''
    
    # get prop
    brief_props, props = solve_single_queue_to_prop(nh, qcm_queue, calctype=calctype, bulklimit=bulklimit, nh_interests=[refh], brief_report=brief_report)

    return (ind, dict(grho=brief_props['grho_refh'], phi=brief_props['phi'], drho=brief_props['drho'], n=refh))


class QCMApp(QMainWindow):
    '''
    The settings of the app is stored in a dict by widget names
    '''
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.tempPath = '' # to store the file name before it is initiated

        self.settings = settings_default.copy() # import default settings. It will be initalized later

        self.peak_tracker = PeakTracker.PeakTracker(max_harm=self.settings['max_harmonic'])
        self.vna_tracker = VNATracker()
        self.qcm = QCM.QCM()

        # define instrument state variables

        self.UITab = 0 # 0: Control; 1: Settings;, 2: Data; 3: Mechanics
        #### initialize the attributes for data saving
        self.data_saver = DataSaver.DataSaver(ver=_version.__version__, settings=self.settings)

        self.vna = None # vna class
        self.temp_sensor = None # class for temp sensor
        self.idle = True # if test is running
        self.reading = False # if myVNA/tempsensor is scanning and reading data
        self.writing = False # if UI is saving data
        self.counter = 0 # for counting the saving interval

        self.settings_harm = '1' # active harmonic in Settings Tab
        self.settings_chn = {'name': 'samp', 'chn': '1'} # active channel 'samp' or 'ref' in Settings Tab
        self.active = {} # active factors e.g.: harm, chnn_name, plt_str, l_str, ind,
        self.mech_chn = 'samp'
        self.chn_set_store = {} # used for storing the channal setup self.settings.freq_span and self.settings.harmdata during manual refit
        self.prop_plot_list = [] # a list to store handles of prop plots


        # check system
        self.system = UIModules.system_check()
        # initialize AccessMyVNA
        #TODO add more code to disable settings_control tab and widges in settings_settings tab
        if self.system == 'win32': # windows
            try:
                # test if MyVNA program is available
                with AccessMyVNA() as vna:
                    if vna.Init() == 0: # is available
                        # self.vna = AccessMyVNA() # save class AccessMyVNA to vna
                        self.vna = vna # save class AccessMyVNA to vna
                    else: # not available
                        pass
                logger.info(vna) 
                logger.info(vna._nsteps) 
            except:
                print('Initiating MyVNA failed!\nMake sure analyser is connected and MyVNA is correctly installed!')
        else: # other system, data analysis only
            # self.vna = AccessMyVNA() # for test only
            pass
        logger.info(self.vna) 

        # does it necessary???
        # if self.vna is not None: # only set the timer when vna is available
        # initiate a timer for test
        self.timer = QTimer()
        # self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.data_collection)

        # initiate a timer for progressbar
        self.bartimer = QTimer()
        self.bartimer.timeout.connect(self.update_progressbar)


        self.init_ui()

        # hide widges not necessary
        self.hide_widgets(
            'version_hide_list'
        )

        # disable widges not necessary
        self.disable_widgets(
            'version_disable_list'
        )

        # hide widgets not for analysis mode
        if self.vna is None:
            self.hide_widgets(
                'analysis_mode_disable_list'
            )

        self.load_settings()


    def init_ui(self):
        #region ###### initiate UI #################################

        #region main UI
        # link tabWidget_settings and stackedWidget_spectra and stackedWidget_data
        self.ui.tabWidget_settings.currentChanged.connect(self.link_tab_page)

        #endregion


        #region cross different sections
        ##### harmonic widgets #####
        # Add widgets related number of harmonics here

        # loop for setting harmonics
        _translate = QCoreApplication.translate
        for i in self.all_harm_list():
            harm = str(i)
            if not getattr(self.ui, 'checkBox_harm' + harm, None): # check if the item exist
                # check box_harm<n>, lineEdit_<start/end>f<n></_r> all come togeter. So, only check one
                ## create widget check box_harm<n>
                setattr(self.ui, 'checkBox_harm' + harm, QCheckBox(self.ui.groupBox_settings_harmonics))
                sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(getattr(self.ui, 'checkBox_harm' + harm).sizePolicy().hasHeightForWidth())
                getattr(self.ui, 'checkBox_harm' + harm).setSizePolicy(sizePolicy)
                getattr(self.ui, 'checkBox_harm' + harm).setObjectName('checkBox_harm'+harm)
                getattr(self.ui, 'checkBox_harm' + harm).setText(_translate('MainWindow', harm))

                for st1 in ['start', 'end']:
                    for st2 in ['', '_r']:
                        setattr(self.ui, 'lineEdit_'+st1+'f'+harm+st2, QLineEdit(self.ui.groupBox_settings_harmonics))
                        getattr(self.ui, 'lineEdit_'+st1+'f'+harm+st2).setReadOnly(True)
                        getattr(self.ui, 'lineEdit_'+st1+'f'+harm+st2).setObjectName('lineEdit_'+st1+'f'+harm+st2)

                # add to layout
                self.ui.gridLayout_settings_control_harms.addWidget(getattr(self.ui, 'checkBox_harm' + harm), (i+1)/2, 0, 1, 1)
                self.ui.gridLayout_settings_control_harms.addWidget(getattr(self.ui, 'lineEdit_startf'+harm), (i+1)/2, 1, 1, 1)
                self.ui.gridLayout_settings_control_harms.addWidget(getattr(self.ui, 'lineEdit_endf'+harm), (i+1)/2, 2, 1, 1)
                self.ui.gridLayout_settings_control_harms.addWidget(getattr(self.ui, 'lineEdit_startf'+harm+'_r'), (i+1)/2, 4, 1, 1)
                self.ui.gridLayout_settings_control_harms.addWidget(getattr(self.ui, 'lineEdit_endf'+harm+'_r'), (i+1)/2, 5, 1, 1)

            ## create frame_sp<n>
            if not getattr(self.ui, 'frame_sp'+harm, None): # check if the item exist
                setattr(self.ui, 'frame_sp'+harm, QFrame(self.ui.page_spectra_show))
                sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(1)
                sizePolicy.setHeightForWidth(getattr(self.ui, 'frame_sp'+harm).sizePolicy().hasHeightForWidth())
                getattr(self.ui, 'frame_sp'+harm).setSizePolicy(sizePolicy)
                # getattr(self.ui, 'frame_sp'+harm).setMinimumSize(QSize(0, 100))
                getattr(self.ui, 'frame_sp'+harm).setMaximumSize(QSize(16777215, 400))
                getattr(self.ui, 'frame_sp'+harm).setFrameShape(QFrame.StyledPanel)
                getattr(self.ui, 'frame_sp'+harm).setFrameShadow(QFrame.Sunken)
                getattr(self.ui, 'frame_sp'+harm).setObjectName('frame_sp'+harm)
                self.ui.verticalLayout_sp.addWidget(getattr(self.ui, 'frame_sp'+harm))

            ## create checkBox_plt<1/2>_h<n>
            for plt_str in ['plt1', 'plt2']:
                if not getattr(self.ui, 'checkBox_'+plt_str+'_h'+harm, None): # check if the item exist
                    setattr(self.ui, 'checkBox_'+plt_str+'_h'+harm, QCheckBox(getattr(self.ui,'frame_'+plt_str+'set')))
                    getattr(self.ui, 'checkBox_'+plt_str+'_h'+harm).setObjectName('checkBox_'+plt_str+'_h'+harm)
                    getattr(self.ui, 'horizontalLayout_'+plt_str+'set_harm').addWidget(getattr(self.ui, 'checkBox_'+plt_str+'_h'+harm))
                    getattr(self.ui, 'checkBox_'+plt_str+'_h'+harm).setText(_translate('MainWindow', harm))

            ## create tab_settings_settings_harm<n>
            if not getattr(self.ui, 'tab_settings_settings_harm'+harm, None): # check if the item exist
                setattr(self.ui, 'tab_settings_settings_harm'+harm, QWidget())
                getattr(self.ui, 'tab_settings_settings_harm'+harm).setObjectName('tab_settings_settings_harm'+harm)
                setattr(self.ui, 'verticalLayout_tab_settings_settings_harm'+harm, QVBoxLayout(getattr(self.ui, 'tab_settings_settings_harm'+harm)))
                getattr(self.ui, 'verticalLayout_tab_settings_settings_harm'+harm).setContentsMargins(0, 0, 0, 0)
                getattr(self.ui, 'verticalLayout_tab_settings_settings_harm'+harm).setSpacing(0)
                getattr(self.ui, 'verticalLayout_tab_settings_settings_harm'+harm).setObjectName('verticalLayout_tab_settings_settings_harm'+harm)
                self.ui.tabWidget_settings_settings_harm.addTab(getattr(self.ui, 'tab_settings_settings_harm'+harm), '')
                self.ui.tabWidget_settings_settings_harm.setTabText(self.ui.tabWidget_settings_settings_harm.indexOf(getattr(self.ui, 'tab_settings_settings_harm'+harm)), _translate('MainWindow', harm))


            ## creat checkBox_nhplot<n>
            if not getattr(self.ui, 'checkBox_nhplot'+harm, None): # check if the item exist
                setattr(self.ui, 'checkBox_nhplot'+harm, QCheckBox(self.ui.groupBox_nhplot))
                getattr(self.ui, 'checkBox_nhplot'+harm).setObjectName('checkBox_nhplot'+harm)
                self.ui.horizontalLayout_nhplot_harms.addWidget(getattr(self.ui, 'checkBox_nhplot'+harm))
                getattr(self.ui, 'checkBox_nhplot'+harm).setText(_translate('MainWindow', harm))


            # set to visable which is default. nothing to do

            # set all frame_sp<n> hided
            getattr(self.ui, 'frame_sp' +harm).setVisible(False)

            # add checkbox to tabWidget_ham for harmonic selection
            setattr(self.ui, 'checkBox_tree_harm' + harm, QCheckBox())
            self.ui.tabWidget_settings_settings_harm.tabBar().setTabButton(
                self.ui.tabWidget_settings_settings_harm.indexOf(
                    getattr(self.ui, 'tab_settings_settings_harm' + harm)
                ),
                QTabBar.LeftSide,
                getattr(self.ui, 'checkBox_tree_harm' + harm
                )
            )

            # set signal
            getattr(self.ui, 'checkBox_tree_harm' + harm).toggled['bool'].connect(
                getattr(self.ui, 'checkBox_harm' + harm).setChecked
            )
            getattr(self.ui, 'checkBox_harm' + harm).toggled['bool'].connect(
                getattr(self.ui, 'checkBox_tree_harm' + harm).setChecked
            )
            # getattr(self.ui, 'checkBox_tree_harm' + harm).toggled['bool'].connect(
            #     getattr(self.ui, 'frame_sp' +harm).setVisible
            # )
            getattr(self.ui, 'checkBox_harm' + harm).toggled['bool'].connect(
                getattr(self.ui, 'frame_sp' +harm).setVisible
            )

            getattr(self.ui, 'checkBox_harm' + harm).toggled['bool'].connect(self.update_widget)

            # checkBox_nhplot<n>
            getattr(self.ui, 'checkBox_nhplot' + harm).toggled.connect(self.update_widget)

        ########

        # show samp & ref related widgets
        self.setvisible_samprefwidgets()
        # set statusbar icon pushButton_status_signal_ch
        self.statusbar_signal_chn_update()


        # set comboBox_plt1_optsy/x, comboBox_plt2_optsy/x
        # dict for the comboboxes
        self.build_comboBox(self.ui.comboBox_plt1_optsy, 'data_plt_opts')
        self.build_comboBox(self.ui.comboBox_plt1_optsx, 'data_plt_opts')
        self.build_comboBox(self.ui.comboBox_plt2_optsy, 'data_plt_opts')
        self.build_comboBox(self.ui.comboBox_plt2_optsx, 'data_plt_opts')

        # set RUN/STOP button
        self.ui.pushButton_runstop.toggled.connect(self.on_clicked_pushButton_runstop)

        # set arrows (la and ra) to change pages
        self.ui.pushButton_settings_la.clicked.connect(
            lambda: self.set_stackedwidget_index(self.ui.stackedWidget_spectra, diret=-1)
        ) # set index-1
        self.ui.pushButton_settings_ra.clicked.connect(
            lambda: self.set_stackedwidget_index(self.ui.stackedWidget_spectra, diret=1)
        ) # set index+1
        self.ui.pushButton_data_la.clicked.connect(
            lambda: self.set_stackedwidget_index(self.ui.stackedWidget_data, diret=-1)
        ) # set index -1
        self.ui.pushButton_data_ra.clicked.connect(
            lambda: self.set_stackedwidget_index(self.ui.stackedWidget_data, diret=1)
        ) # set index 1

        #endregion


        #region settings_control
        # set lineEdit_startf<n> & lineEdit_endf<n> & lineEdit_startf<n>_r & lineEdit_endf<n>_r background
        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'lineEdit_startf' + harm).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_endf' + harm).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_startf' + harm + '_r').setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_endf' + harm + '_r').setStyleSheet(
                "QLineEdit { background: transparent; }"
            )

        # dateTimeEdit_reftime on dateTimeChanged
        self.ui.dateTimeEdit_reftime.dateTimeChanged.connect(self.on_dateTimeChanged_dateTimeEdit_reftime)
        # set pushButton_resetreftime
        self.ui.pushButton_resetreftime.clicked.connect(self.reset_reftime)

        # set recording time value
        self.ui.spinBox_recordinterval.valueChanged.connect(self.update_widget)
        self.ui.spinBox_recordinterval.valueChanged.connect(self.set_recording_time)
        self.ui.spinBox_refreshresolution.valueChanged.connect(self.update_widget)
        self.ui.spinBox_refreshresolution.valueChanged.connect(self.set_recording_time)
        self.ui.spinBox_scaninterval.valueChanged.connect(self.update_widget)
        self.ui.spinBox_scaninterval.valueChanged.connect(self.set_recording_time)
        # # set spinBox_scaninterval background
        # self.ui.spinBox_scaninterval.setStyleSheet(
        #     "QLineEdit { background: transparent; }"
        # )

        # add value to the comboBox_settings_control_dispmode
        self.build_comboBox(self.ui.comboBox_settings_control_dispmode, 'display_opts')
        self.ui.comboBox_settings_control_dispmode.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_control_dispmode.currentIndexChanged.connect(self. update_freq_display_mode)

        # set pushButton_gotofolder
        self.ui.pushButton_gotofolder.clicked.connect(self.on_clicked_pushButton_gotofolder)

        # set pushButton_newfile
        self.ui.pushButton_newfile.clicked.connect(self.on_triggered_new_exp)

        # set pushButton_appendfile
        self.ui.pushButton_appendfile.clicked.connect(self.on_triggered_load_exp)


        self.ui.checkBox_dynamicfit.stateChanged.connect(self.update_widget)
        self.ui.spinBox_fitfactor.valueChanged.connect(self.update_widget)
        self.ui.checkBox_dynamicfitbyharm.clicked['bool'].connect(self.update_widget)
        self.ui.checkBox_fitfactorbyharm.clicked['bool'].connect(self.update_widget)

        # plainTextEdit_settings_sampledescription
        self.ui.plainTextEdit_settings_sampledescription.textChanged.connect(lambda: self.update_widget(
            self.ui.plainTextEdit_settings_sampledescription.document().toPlainText()
        ))

        # set signals to update spectra show display options
        self.ui.radioButton_spectra_showGp.toggled.connect(self.update_widget)
        self.ui.radioButton_spectra_showGp.clicked.connect(self.mpl_sp_clr_lines_set_label)
        self.ui.radioButton_spectra_showBp.toggled.connect(self.update_widget)
        self.ui.radioButton_spectra_showBp.clicked.connect(self.mpl_sp_clr_lines_set_label)
        self.ui.radioButton_spectra_showpolar.toggled.connect(self.update_widget)
        self.ui.radioButton_spectra_showpolar.clicked.connect(self.mpl_sp_clr_lines_set_label)
        self.ui.checkBox_spectra_showchi.toggled.connect(self.update_widget)
        self.ui.checkBox_spectra_showchi.toggled.connect(self.mpl_sp_clr_chis)

        # set signals to checkBox_control_rectemp
        self.ui.checkBox_control_rectemp.clicked['bool'].connect(self.on_clicked_set_temp_sensor)

        # set checkBox_dynamicfitbyharm
        self.ui.checkBox_dynamicfitbyharm.clicked['bool'].connect(self.on_clicked_checkBox_dynamicfitbyharm)

        # set checkBox_fitfactorbyharm
        self.ui.checkBox_fitfactorbyharm.clicked['bool'].connect(self.on_clicked_checkBox_fitfactorbyharm)

        # set lineEdit_datafilestr background
        self.ui.lineEdit_datafilestr.setStyleSheet(
            "QLineEdit { background: transparent; }"
        )

        #endregion


        #region settings_settings

        self.ui.checkBox_activechn_samp.clicked.connect(self.check_checked_activechn)
        self.ui.checkBox_activechn_samp. stateChanged.connect(self.update_widget)
        self.ui.checkBox_activechn_samp. stateChanged.connect(self.update_vnachannel)

        self.ui.checkBox_activechn_ref.clicked.connect(self.check_checked_activechn)
        self.ui.checkBox_activechn_ref. stateChanged.connect(self.update_widget)
        self.ui.checkBox_activechn_ref. stateChanged.connect(self.update_vnachannel)

        # add checkbox to tabWidget_settings_settings_samprefchn
        self.ui.tabWidget_settings_settings_samprefchn.tabBar().setTabButton(
            self.ui.tabWidget_settings_settings_samprefchn.indexOf(self.ui.tab_settings_settings_harmchnsamp),
            QTabBar.LeftSide,
            self.ui.checkBox_activechn_samp,
        )
        self.ui.tabWidget_settings_settings_samprefchn.tabBar().setTabButton(
            self.ui.tabWidget_settings_settings_samprefchn.indexOf(self.ui.tab_settings_settings_harmchnref),
            QTabBar.LeftSide,
            self.ui.checkBox_activechn_ref,
        )

        # set signal
        self.ui.tabWidget_settings_settings_samprefchn.currentChanged.connect(self.update_widget)
        self.ui.tabWidget_settings_settings_samprefchn.currentChanged.connect(self.update_settings_chn)

        # remove tab_settings_settings_harmchnrefit from index
        self.add_manual_refit_tab(False)

        ### add combobox into treewidget
        self.ui.tabWidget_settings_settings_harm.currentChanged.connect(self.update_harmonic_tab)
        # move lineEdit_scan_harmstart
        self.move_to_col(
            self.ui.lineEdit_scan_harmstart,
            self.ui.treeWidget_settings_settings_harmtree,
            'Start',
            100,
        )

        # move lineEdit_scan_harmend
        self.move_to_col(
            self.ui.lineEdit_scan_harmend,
            self.ui.treeWidget_settings_settings_harmtree,
            'End',
            100,
        )

        # move lineEdit_scan_harmsteps
        self.move_to_col(
            self.ui.lineEdit_scan_harmsteps,
            self.ui.treeWidget_settings_settings_harmtree,
            'Steps',
            100,
        )

        # move frame_peaks_num
        self.move_to_col(
            self.ui.frame_peaks_num,
            self.ui.treeWidget_settings_settings_harmtree,
            'Num.',
            160,
        )

        # move frame_peaks_policy
        self.move_to_col(
            self.ui.frame_peaks_policy,
            self.ui.treeWidget_settings_settings_harmtree,
            'Policy',
            160,
        )

        # move frame_settings_settings_harmphase
        self.move_to_col(
            self.ui.frame_settings_settings_harmphase,
            self.ui.treeWidget_settings_settings_harmtree,
            'Phase',
            160,
        )

        # move lineEdit_peaks_threshold
        self.move_to_col(
            self.ui.lineEdit_peaks_threshold,
            self.ui.treeWidget_settings_settings_harmtree,
            'Threshold',
            100,
        )

        # move lineEdit_peaks_prominence
        self.move_to_col(
            self.ui.lineEdit_peaks_prominence,
            self.ui.treeWidget_settings_settings_harmtree,
            'Prominence',
            100,
        )

        # move checkBox_harmfit
        self.move_to_col(
            self.ui.checkBox_harmfit,
            self.ui.treeWidget_settings_settings_harmtree,
            'Fit',
            100,
        )

        # move spinBox_harmfitfactor
        self.move_to_col(
            self.ui.spinBox_harmfitfactor,
            self.ui.treeWidget_settings_settings_harmtree,
            'Factor',
            100,
        )

        # set max value availabe
        self.ui.spinBox_harmfitfactor.setMaximum(config_default['fitfactor_max'])

        # comboBox_tracking_method
        self.create_combobox(
            'comboBox_tracking_method',
            config_default['span_mehtod_opts'],
            100,
            'Method',
            self.ui.treeWidget_settings_settings_harmtree
        )

        # add span track_method
        self.create_combobox(
            'comboBox_tracking_condition',
            config_default['span_track_opts'],
            100,
            'Condition',
            self.ui.treeWidget_settings_settings_harmtree
        )

        # insert samp_channel
        self.create_combobox(
            'comboBox_samp_channel',
            config_default['vna_channel_opts'],
            100,
            'S Channel',
            self.ui.treeWidget_settings_settings_hardware
        )

        # inser ref_channel
        self.create_combobox(
            'comboBox_ref_channel',
            config_default['vna_channel_opts'],
            100,
            'R Channel',
            self.ui.treeWidget_settings_settings_hardware
        )

        ## check comboBox_samp_channel & comboBox_ref_channel list by calibration file
        # get current chn from myvna
        curr_chn = None if self.vna is None else self.vna._chn
        logger.info('curr_chn: %s\n', curr_chn) 
        if not self.vna_tracker.cal['ADC1'] and (curr_chn != 1): # no calibration file for ADC1
            # delete ADC1 from both lists
            if self.ui.comboBox_samp_channel.findData(1) != -1:
                self.ui.comboBox_samp_channel.removeItem(self.ui.comboBox_samp_channel.findData(1))
            if self.ui.comboBox_ref_channel.findData(1) != -1:
                self.ui.comboBox_ref_channel.removeItem(self.ui.comboBox_ref_channel.findData(1))
        if not self.vna_tracker.cal['ADC2'] and (curr_chn != 2): # no calibration file for ADC1
            # delete ADC1 from both lists
            if self.ui.comboBox_samp_channel.findData(2) != -1:
                self.ui.comboBox_samp_channel.removeItem(self.ui.comboBox_samp_channel.findData(2))
            if self.ui.comboBox_ref_channel.findData(2) != -1:
                self.ui.comboBox_ref_channel.removeItem(self.ui.comboBox_ref_channel.findData(2))



        # connect ref_channel
        # self.ui.comboBox_ref_channel.currentIndexChanged.connect() #TODO add function checking if sample and ref have the same channel

        # insert base_frequency
        self.create_combobox(
            'comboBox_base_frequency',
            config_default['base_frequency_opts'],
            100,
            'Base Frequency',
            self.ui.treeWidget_settings_settings_hardware
        )

        # insert range
        self.create_combobox(
            'comboBox_range',
            config_default['range_opts'],
            100,
            'Range',
            self.ui.treeWidget_settings_settings_hardware
        )

        # insert crystal cut
        self.create_combobox(
            'comboBox_crystalcut',
            config_default['crystal_cut_opts'],
            100,
            'Cut',
            self.ui.treeWidget_settings_settings_hardware
        )

        # insert comboBox_settings_settings_analyzer
        self.create_combobox(
            'comboBox_settings_settings_analyzer',
            config_default['analyzer_opts'],
            100,
            'Analyzer',
            self.ui.treeWidget_settings_settings_hardware
        )

        self.ui.comboBox_settings_settings_analyzer.currentIndexChanged.connect(self.update_widget)
        # TODO connect module importing function here
        self.ui.comboBox_settings_settings_analyzer.currentIndexChanged.connect(self.update_widget)

        # add comBox_tempmodule to treeWidget_settings_settings_hardware
        try:
            config_default['temp_class_opts_list'] = TempModules.class_list # when TempModules is loaded
        except:
            config_default['temp_class_opts_list'] = None # no temp module is loaded
        self.create_combobox(
            'comboBox_tempmodule',
            # UIModules.list_modules(TempModules),
            config_default['temp_class_opts_list'],
            100,
            'Module',
            self.ui.treeWidget_settings_settings_hardware,
        )
        self.settings['comboBox_tempmodule'] = self.ui.comboBox_tempmodule.itemData(self.ui.comboBox_tempmodule.currentIndex())
        self.ui.comboBox_tempmodule.activated.connect(self.update_widget)

        # add comboBox_tempdevice to treeWidget_settings_settings_hardware
        if self.vna and self.system == 'win32':
            config_default['tempdevs_opts'] = TempDevices.dict_available_devs(config_default['tempdevices_dict'])
            self.create_combobox(
                'comboBox_tempdevice',
                config_default['tempdevs_opts'],
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware,
            )
            self.settings['comboBox_tempdevice'] = self.ui.comboBox_tempdevice.itemData(self.ui.comboBox_tempdevice.currentIndex())
            self.ui.comboBox_tempdevice.currentIndexChanged.connect(self.update_tempdevice)
        else: # vna is not available
            self.create_combobox(
                'comboBox_tempdevice',
                [],  # an empty list
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware,
            )
            self.settings['comboBox_tempdevice'] = None # set to None

        # insert thrmcpl type
        self.create_combobox(
            'comboBox_thrmcpltype',
            config_default['thrmcpl_opts'],
            100,
            'Thrmcpl Type',
            self.ui.treeWidget_settings_settings_hardware
        )

        if not self.settings['comboBox_tempdevice']: # vna or tempdevice are not availabel
            # set temp related widgets unavailable
            self.disable_widgets(
                'temp_device_setting_disable_list',
                'temp_settings_enable_disable_list',
            )


        # insert time_unit
        self.create_combobox(
            'comboBox_timeunit',
            config_default['time_unit_opts'],
            100,
            'Time Unit',
            self.ui.treeWidget_settings_settings_plots
        )

        # insert temp_unit
        self.create_combobox(
            'comboBox_tempunit',
            config_default['temp_unit_opts'],
            100,
            'Temp. Unit',
            self.ui.treeWidget_settings_settings_plots
        )

        # insert X Scale
        self.create_combobox(
            'comboBox_xscale',
            config_default['scale_opts'],
            100,
            'X Scale',
            self.ui.treeWidget_settings_settings_plots
        )

        # insert gamma scale
        self.create_combobox(
            'comboBox_yscale',
            config_default['scale_opts'],
            100,
            'Y Scale',
            self.ui.treeWidget_settings_settings_plots
        )

        # move checkBox_linkx to treeWidget_settings_settings_plots

        self.move_to_col(
            self.ui.checkBox_linkx,
            self.ui.treeWidget_settings_settings_plots,
            'Link Time'
        )

        # set treeWidget_settings_settings_harmtree expanded
        self.ui.treeWidget_settings_settings_harmtree.expandToDepth(0)
        # set treeWidget_settings_settings_hardware expanded
        self.ui.treeWidget_settings_settings_hardware.expandToDepth(0)
        # set treeWidget_settings_settings_plots expanded
        self.ui.treeWidget_settings_settings_plots.expandToDepth(0)


        # move center pushButton_settings_harm_cntr to treeWidget_settings_settings_harmtree
        # comment it for now, this button is not using
        # move it to other place will disable the hide function ran above
        # self.move_to_col(
        #     self.ui.pushButton_settings_harm_cntr,
        #     self.ui.treeWidget_settings_settings_harmtree,
        #     'Scan',
        #     50
        # )

        # move center checkBox_settings_temp_sensor to treeWidget_settings_settings_hardware
        self.move_to_col(
            self.ui.checkBox_settings_temp_sensor,
            self.ui.treeWidget_settings_settings_hardware,
            'Temperature'
        )

        # set tabWidget_settings background
        self.ui.tabWidget_settings.setStyleSheet(
            # "QTabWidget, QTabWidget::pane, QTabBar { background: transparent; }"
            "QTabWidget::pane { border: 0;}"
            # "QTabWidget, QTabWidget::pane, QTabBar { border-width: 5px; border-color: red; }"
            # "QTabBar::tab-bar { background: transparent; }"
        )

        # set treeWidget_settings_settings_harmtree background
        self.ui.treeWidget_settings_settings_harmtree.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )
        # set treeWidget_settings_settings_hardware background
        self.ui.treeWidget_settings_settings_hardware.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # set treeWidget_settings_settings_plots background
        self.ui.treeWidget_settings_settings_plots.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # resize the TabBar.Button
        self.ui.tabWidget_settings_settings_harm.setStyleSheet(
            "QTabWidget::pane { height: 0; border: 0px; }"
            "QTabWidget {background-color: transparent;}"
            "QTabWidget::tab-bar { left: 5px; /* move to the right by 5px */ }"
            "QTabBar::tab { border: 1px solid #9B9B9B; border-top-left-radius: 1px; border-top-right-radius: 1px;}"
            "QTabBar::tab { height: 20px; width: 42px; padding: 0px; }"
            "QTabBar::tab:selected, QTabBar::tab:hover { background: white; }"
            "QTabBar::tab:selected { height: 22px; width: 46px; border-bottom-color: none; }"
            "QTabBar::tab:selected { margin-left: -2px; margin-right: -2px; }"
            "QTabBar::tab:first:selected { margin-left: 0; width: 42px; }"
            "QTabBar::tab:last:selected { margin-right: 0; width: 42px; }"
            "QTabBar::tab:!selected { margin-top: 2px; }"
            )

        self.ui.lineEdit_scan_harmstart.setValidator(QDoubleValidator(1, math.inf, 6))
        self.ui.lineEdit_scan_harmend.setValidator(QDoubleValidator(1, math.inf, 6))
        self.ui.lineEdit_scan_harmsteps.setValidator(QIntValidator(0, 2147483647))
        self.ui.lineEdit_peaks_threshold.setValidator(QDoubleValidator(0, math.inf, 6))
        self.ui.lineEdit_peaks_prominence.setValidator(QDoubleValidator(0, math.inf, 6))

        # set signals of widgets in tabWidget_settings_settings_harm
        self.ui.lineEdit_scan_harmstart.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.lineEdit_scan_harmend.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.comboBox_base_frequency.currentIndexChanged.connect(self.update_base_freq)
        self.ui.comboBox_range.currentIndexChanged.connect(self.update_range)

        # set signals to update span settings_settings
        self.ui.lineEdit_scan_harmsteps.textEdited.connect(self.update_harmwidget)
        self.ui.comboBox_tracking_method.activated.connect(self.update_harmwidget)
        self.ui.comboBox_tracking_condition.activated.connect(self.update_harmwidget)
        self.ui.checkBox_harmfit.toggled['bool'].connect(self.update_harmwidget)
        self.ui.spinBox_harmfitfactor.valueChanged.connect(self.update_harmwidget)
        self.ui.spinBox_peaks_num.valueChanged.connect(self.update_harmwidget)
        self.ui.lineEdit_peaks_threshold.textEdited.connect(self.update_harmwidget)
        self.ui.lineEdit_peaks_prominence.textEdited.connect(self.update_harmwidget)
        self.ui.radioButton_peaks_num_max.toggled['bool'].connect(self.update_harmwidget)
        self.ui.radioButton_peaks_num_fixed.toggled['bool'].connect(self.update_harmwidget)
        self.ui.radioButton_peaks_policy_minf.toggled['bool'].connect(self.update_harmwidget)
        self.ui.radioButton_peaks_policy_maxamp.toggled['bool'].connect(self.update_harmwidget)
        self.ui.spinBox_peaks_policy_peakidx.valueChanged.connect(self.update_harmwidget)
        self.ui.checkBox_settings_settings_harmlockphase.toggled['bool'].connect(self.update_harmwidget)
        self.ui.doubleSpinBox_settings_settings_harmlockphase.valueChanged.connect(self.update_harmwidget)

        # set signals to update hardware settings_settings
        self.ui.comboBox_samp_channel.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_samp_channel.currentIndexChanged.connect(self.update_vnachannel)
        self.ui.comboBox_samp_channel.currentIndexChanged.connect(self.update_settings_chn)
        self.ui.comboBox_ref_channel.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_ref_channel.currentIndexChanged.connect(self.update_vnachannel)
        self.ui.comboBox_ref_channel.currentIndexChanged.connect(self.update_settings_chn)

        # self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.update_tempsensor)
        self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.on_clicked_set_temp_sensor)
        # self.ui.comboBox_thrmcpltype.currentIndexChanged.connect(self.update_tempdevice) # ??
        self.ui.comboBox_thrmcpltype.currentIndexChanged.connect(self.update_thrmcpltype)

        # set signals to update plots settings_settings
        self.ui.comboBox_timeunit.currentIndexChanged.connect(self.update_timeunit)
        self.ui.comboBox_timeunit.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_timeunit.currentIndexChanged.connect(self.update_mpl_plt12)

        self.ui.comboBox_tempunit.currentIndexChanged.connect(self.update_tempunit)
        self.ui.comboBox_tempunit.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_tempunit.currentIndexChanged.connect(self.update_mpl_plt12)

        self.ui.comboBox_xscale.currentIndexChanged.connect(self.update_timescale)
        self.ui.comboBox_xscale.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_xscale.currentIndexChanged.connect(self.update_mpl_plt12)

        self.ui.comboBox_yscale.currentIndexChanged.connect(self.update_yscale)
        self.ui.comboBox_yscale.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_yscale.currentIndexChanged.connect(self.update_mpl_plt12)

        self.ui.checkBox_linkx.stateChanged.connect(self.update_linkx)
        self.ui.checkBox_linkx.stateChanged.connect(self.update_data_axis)
        self.ui.checkBox_linkx.stateChanged.connect(self.update_mpl_plt12)

        #endregion


        #region settings_data

        # set treeWidget_settings_data_refs background
        self.ui.treeWidget_settings_data_refs.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # load opts to combox
        self.build_comboBox(self.ui.comboBox_settings_data_samprefsource, 'ref_channel_opts')
        self.build_comboBox(self.ui.comboBox_settings_data_refrefsource, 'ref_channel_opts')

        # move pushButton_settings_data_resetshiftedt0
        self.move_to_col(
            self.ui.pushButton_settings_data_resetshiftedt0,
            self.ui.treeWidget_settings_data_refs,
            'Time Shift',
            100,
        )
        self.ui.pushButton_settings_data_resetshiftedt0.clicked.connect(self.reset_shiftedt0)

        # move label_settings_data_t0
        self.move_to_col(
            self.ui.label_settings_data_t0,
            self.ui.treeWidget_settings_data_refs,
            't0',
            # 100,
        )

        # move dateTimeEdit_settings_data_t0shifted
        self.move_to_col(
            self.ui.dateTimeEdit_settings_data_t0shifted,
            self.ui.treeWidget_settings_data_refs,
            'Shifted t0',
            # 180,
        )
        self.ui.dateTimeEdit_settings_data_t0shifted.dateTimeChanged.connect(self.on_dateTimeChanged_dateTimeEdit_t0shifted)

        # move frame_settings_data_recalcref
        self.move_to_col(
            self.ui.frame_settings_data_recalcref,
            self.ui.treeWidget_settings_data_refs,
            'Reference',
            # 100,
        )
        self.ui.pushButton_settings_data_recalcref.clicked.connect(self.recalc_refs)

        # move lineEdit_settings_data_sampidx
        self.move_to_col(
            self.ui.lineEdit_settings_data_sampidx,
            self.ui.treeWidget_settings_data_refs,
            'S chn.',
            # 100,
        )
        # move frame_settings_data_sampref
        self.move_to_col(
            self.ui.frame_settings_data_sampref,
            self.ui.treeWidget_settings_data_refs,
            'S ref.',
            # 100,
        )
        self.ui.comboBox_settings_data_samprefsource.currentIndexChanged.connect(self.update_widget)
        self.ui.lineEdit_settings_data_sampidx.textChanged.connect(self.update_widget)
        self.ui.lineEdit_settings_data_samprefidx.textChanged.connect(self.update_widget)

        # NOTE: following two only emitted when value manually edited (activated)
        self.ui.comboBox_settings_data_samprefsource.activated.connect(self.save_data_saver_sampref)
        self.ui.comboBox_settings_data_samprefsource.activated.connect(self.set_mech_layer_0_indchn_val)
        self.ui.lineEdit_settings_data_sampidx.editingFinished.connect(self.save_data_saver_sampidx)
        self.ui.lineEdit_settings_data_samprefidx.editingFinished.connect(self.save_data_saver_sampref)
        self.ui.lineEdit_settings_data_samprefidx.editingFinished.connect(self.set_mech_layer_0_indchn_val)

        # move lineEdit_settings_data_refidx
        self.move_to_col(
            self.ui.lineEdit_settings_data_refidx,
            self.ui.treeWidget_settings_data_refs,
            'R chn.',
            # 100,
        )
        # move frame_settings_data_refref
        self.move_to_col(
            self.ui.frame_settings_data_refref,
            self.ui.treeWidget_settings_data_refs,
            'R ref.',
            # 100,
        )

        self.ui.comboBox_settings_data_refrefsource.currentIndexChanged.connect(self.update_widget)
        self.ui.lineEdit_settings_data_refidx.textChanged.connect(self.update_widget)
        self.ui.lineEdit_settings_data_refrefidx.textChanged.connect(self.update_widget)

        # NOTE: following two only emitted when value manually edited (activated)
        self.ui.comboBox_settings_data_refrefsource.activated.connect(self.save_data_saver_refref)
        self.ui.comboBox_settings_data_refrefsource.activated.connect(self.set_mech_layer_0_indchn_val)
        self.ui.lineEdit_settings_data_refidx.editingFinished.connect(self.save_data_saver_refidx)
        self.ui.lineEdit_settings_data_refrefidx.editingFinished.connect(self.save_data_saver_refref)
        self.ui.lineEdit_settings_data_refrefidx.editingFinished.connect(self.set_mech_layer_0_indchn_val)

        # move frame_settings_data_tempref
        self.move_to_col(
            self.ui.frame_settings_data_tempref,
            self.ui.treeWidget_settings_data_refs,
            'Channels',
            # 100,
        )
        # load opts to comboBox_settings_data_ref_fitttype
        self.build_comboBox(self.ui.comboBox_settings_data_ref_fitttype, 'ref_interp_opts')
        self.ui.comboBox_settings_data_ref_fitttype.currentIndexChanged.connect(self.update_widget)
        # initially hiden
        self.ui.comboBox_settings_data_ref_fitttype.hide()
        self.ui.comboBox_settings_data_ref_fitttype.currentIndexChanged.connect(self.on_ref_mode_changed)
        # comboBox_settings_data_ref_crystmode
        self.build_comboBox(self.ui.comboBox_settings_data_ref_crystmode, 'ref_crystal_opts')
        self.ui.comboBox_settings_data_ref_crystmode.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_data_ref_crystmode.currentIndexChanged.connect(self.on_ref_mode_changed)

        # comboBox_settings_data_ref_tempmode
        self.build_comboBox(self.ui.comboBox_settings_data_ref_tempmode, 'ref_mode_opts')
        self.ui.comboBox_settings_data_ref_tempmode.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_data_ref_tempmode.currentIndexChanged.connect(self.on_ref_mode_changed)


       # set treeWidget_settings_data_refs expanded
        self.ui.treeWidget_settings_data_refs.expandToDepth(0)

        # spinBox_settings_data_marknpts
        # self.ui.spinBox_settings_data_marknpts.valueChanged.connect(self.update_widget)
        # pushButton_settings_data_marknpts
        self.ui.pushButton_settings_data_marknpts.clicked.connect(self.marknpts)
        # comboBox_settings_data_marknptschn
        self.build_comboBox(self.ui.comboBox_settings_data_marknptschn, 'channel_opts')
        # self.ui.comboBox_settings_data_marknptschn.currentIndexChanged.connect(self.update_widget)

        #endregion


        #region settings_mechanics
        #########
        self.ui.tabWidget_mechanics_chn.currentChanged.connect(self.on_mech_chn_changed)

        self.ui.checkBox_settings_mech_liveupdate.toggled.connect(self.update_widget)

        # actionfilm_construction_mode_switch
        self.ui.stackedWidget_settings_mechanics_modeswitch.addAction(self.ui.actionfilm_construction_mode_switch)
        self.ui.actionfilm_construction_mode_switch.triggered.connect(self.film_construction_mode_switch)

        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'checkBox_nhplot' + harm).toggled.connect(self.update_widget)

        # set max value of nhcalc_n<n>
        self.ui.spinBox_settings_mechanics_nhcalc_n1.setMaximum(self.settings['max_harmonic'])
        self.ui.spinBox_settings_mechanics_nhcalc_n1.setMaximum(self.settings['max_harmonic'])
        self.ui.spinBox_settings_mechanics_nhcalc_n3.setMaximum(self.settings['max_harmonic'])
        self.ui.spinBox_settings_mechanics_nhcalc_n1.valueChanged.connect(self.update_widget)
        self.ui.spinBox_settings_mechanics_nhcalc_n2.valueChanged.connect(self.update_widget)
        self.ui.spinBox_settings_mechanics_nhcalc_n3.valueChanged.connect(self.update_widget)

        self.ui.pushButton_settings_mechanics_clrallprops.clicked.connect(self.mech_clear)



        self.ui.checkBox_settings_mechanics_witherror.toggled.connect(self.update_widget)

        # spinBox_mech_expertmode_layernum
        self.ui.spinBox_mech_expertmode_layernum.setMinimum(config_default['min_mech_layers'])
        self.ui.spinBox_mech_expertmode_layernum.setMaximum(config_default['max_mech_layers'])
        self.ui.spinBox_mech_expertmode_layernum.valueChanged.connect(self.update_mechchnwidget)
        self.ui.spinBox_mech_expertmode_layernum.valueChanged.connect(self.build_mech_layers)

        self.ui.spinBox_mech_expertmode_layernum.setValue(self.settings.get('spinBox_mech_expertmode_layernum', 0))

        # comboBox_settings_mechanics_calctype
        self.build_comboBox(self.ui.comboBox_settings_mechanics_calctype, 'calctype_opts')
        self.ui.comboBox_settings_mechanics_calctype.currentIndexChanged.connect(self.update_widget)

        # doubleSpinBox_settings_mechanics_bulklimit
        self.ui.doubleSpinBox_settings_mechanics_bulklimit.setMinimum(config_default['mech_bulklimit']['min'])
        self.ui.doubleSpinBox_settings_mechanics_bulklimit.setMaximum(config_default['mech_bulklimit']['max'])
        self.ui.doubleSpinBox_settings_mechanics_bulklimit.setSingleStep(config_default['mech_bulklimit']['step'])
        self.ui.doubleSpinBox_settings_mechanics_bulklimit.valueChanged.connect(self.update_widget)

        self.ui.radioButton_settings_mech_refto_air.toggled.connect(self.update_widget)
        self.ui.radioButton_settings_mech_refto_overlayer.toggled.connect(self.update_widget)
                
        # hide tableWidget_settings_mechanics_errortab
        self.ui.tableWidget_settings_mechanics_errortab.hide()
        # hide tableWidget_settings_mechanics_contoursettings
        self.ui.tableWidget_settings_mechanics_contoursettings.hide()
        # hide groupBox_settings_mechanics_simulator
        self.ui.groupBox_settings_mechanics_simulator.hide()

        self.ui.comboBox_settings_mechanics_selectmodel.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_mechanics_selectmodel.currentIndexChanged.connect(self.set_mechmodel_widgets)
        self.build_comboBox(self.ui.comboBox_settings_mechanics_selectmodel, 'qcm_model_opts')
        # initiate data
        # self.settings['comboBox_settings_mechanics_selectmodel'] = self.ui.comboBox_settings_mechanics_selectmodel.itemData(self.ui.comboBox_settings_mechanics_selectmodel.currentIndex())

        #### following widgets are not saved in self.settings
        # label_settings_mechanics_model_overlayer
        # comboBox_settings_mechanics_model_overlayer/samp_chn
        # lineEdit_settings_mechanics_model_overlayer/samp _idx
        # comboBox_settings_mechanics_model_overlayer/samp _chn
        self.build_comboBox(self.ui.comboBox_settings_mechanics_model_overlayer_chn, 'ref_channel_opts')
        # comboBox_settings_mechanics_model_samplayer_chn
        self.build_comboBox(self.ui.comboBox_settings_mechanics_model_samplayer_chn, 'ref_channel_opts')

        # groupBox_settings_mechanics_contour
        self.ui.groupBox_settings_mechanics_contour.toggled['bool'].connect(self.mech_splitter_vis) # TODO change the name

        # comboBox_settings_mechanics_contourdata
        self.build_comboBox(self.ui.comboBox_settings_mechanics_contourdata, 'contour_data_opts')
        self.ui.comboBox_settings_mechanics_contourdata.currentIndexChanged.connect(self.update_widget)

        # comboBox_settings_mechanics_contourtype
        self.build_comboBox(self.ui.comboBox_settings_mechanics_contourtype, 'contour_type_opts')
        self.ui.comboBox_settings_mechanics_contourtype.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_mechanics_contourtype.currentIndexChanged.connect(self.make_contours)

        # comboBox_settings_mechanics_contourcmap
        self.build_comboBox(self.ui.comboBox_settings_mechanics_contourcmap, 'contour_cmap_opts')
        self.ui.comboBox_settings_mechanics_contourcmap.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_settings_mechanics_contourcmap.currentIndexChanged.connect(self.make_contours)

        # tableWidget_settings_mechanics_contoursettings
        self.add_table_headers(
            'tableWidget_settings_mechanics_contoursettings',
            config_default['mech_contour_lim_tab_vheaders'],
            config_default['mech_contour_lim_tab_hheaders'],
        )
        self.ui.tableWidget_settings_mechanics_contoursettings.itemChanged.connect(self.on_changed_mech_contour_lim_tab)

        self.ui.pushButton_settings_mechanics_contourplot.clicked.connect(self.add_data_to_contour)

        #endregion

        #region spectra_show
        # add figure mpl_sp[n] into frame_sp[n]
        for harm in self.all_harm_list(as_str=True):
            # add first ax
            setattr(
                self.ui, 'mpl_sp' + harm,
                MatplotlibWidget(
                    parent=getattr(self.ui, 'frame_sp' + harm),
                    axtype='sp',
                    showtoolbar=False,
                )
            )
            # getattr(self.ui, 'mpl_sp' + harm).fig.text(0.01, 0.98, harm, va='top',ha='left') # option: weight='bold'
            getattr(self.ui, 'mpl_sp' + harm).update_sp_text_harm(harm)
            # set mpl_sp<n> border
            getattr(self.ui, 'mpl_sp' + harm).setStyleSheet(
                "border: 0;"
            )
            getattr(self.ui, 'mpl_sp' + harm).setContentsMargins(0, 0, 0, 0)
            getattr(self.ui, 'frame_sp' + harm).setLayout(
                self.set_frame_layout(
                    getattr(self.ui, 'mpl_sp' + harm)
                )
            )


        #endregion


        #region spectra_fit
        # add figure mpl_spectra_fit_polar into frame_spectra_fit_polar
        self.ui.mpl_spectra_fit_polar = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit_polar,
            axtype='sp_polar',
            showtoolbar=('Save',),
            )
        self.ui.frame_spectra_fit_polar.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit_polar))

        # add figure mpl_spectra_fit into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit,
            axtype='sp_fit',
            showtoolbar=('Save',),
            # showtoolbar=False,
            )
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))
        # connect signal
        self.ui.mpl_spectra_fit.ax[0].cidx = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('xlim_changed', self.on_fit_lims_change)
        self.ui.mpl_spectra_fit.ax[0].cidy = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('ylim_changed', self.on_fit_lims_change)

        # disconnect signal while dragging
        # self.ui.mpl_spectra_fit.canvas.mpl_connect('button_press_event', self.spectra_fit_axesevent_disconnect)
        # # reconnect signal after dragging (mouse release)
        # self.ui.mpl_spectra_fit.canvas.mpl_connect('button_release_event', self.spectra_fit_axesevent_connect)

        #
        self.ui.pushButton_manual_refit.clicked.connect(self.init_manual_refit)
        # hide widget for manual refit
        self.hide_widgets('manual_refit_enable_disable_list')


        self.ui.horizontalSlider_spectra_fit_spanctrl.valueChanged.connect(self.on_changed_slider_spanctrl)
        self.ui.horizontalSlider_spectra_fit_spanctrl.sliderReleased.connect(self.on_released_slider_spanctrl)
        self.ui.horizontalSlider_spectra_fit_spanctrl.actionTriggered .connect(self.on_acctiontriggered_slider_spanctrl)

        # pushButton_spectra_fit_refresh
        self.ui.pushButton_spectra_fit_refresh.clicked.connect(self.on_clicked_pushButton_spectra_fit_refresh)
        self.ui.pushButton_spectra_fit_showall.clicked.connect(self.on_clicked_pushButton_spectra_fit_showall)
        self.ui.pushButton_spectra_fit_fit.clicked.connect(self.on_clicked_pushButton_spectra_fit_fit)

        #endregion


        #region spectra_mechanics

        self.ui.spinBox_spectra_mechanics_currid.valueChanged.connect(self.on_changed_spinBox_spectra_mechanics_currid)

        self.ui.pushButton_spectra_mechanics_refreshtable.clicked.connect(self.update_spectra_mechanics_table)

        # tableWidget_spectra_mechanics_table
        # make dict for horizontal headers
        mech_table_hnames = {}
        for i in self.all_harm_list():
            mech_table_hnames[int((i-1)/2)] = 'n'+str(i)

        self.add_table_headers(
            'tableWidget_spectra_mechanics_table',
            config_default['mech_table_rowheaders'],
            mech_table_hnames,
        )

        #endregion

        #region data
        # add mpl_legend into frame_legend
        self.ui.mpl_legend = MatplotlibWidget(
            parent=self.ui.frame_legend,
            axtype='legend',
            showtoolbar=False,
            )
        self.ui.mpl_legend.setStyleSheet("background: transparent;")
        self.ui.frame_legend.setLayout(self.set_frame_layout(self.ui.mpl_legend))
        # change frame_legend height
        mpl_legend_p = self.ui.mpl_legend.leg.get_window_extent()
        self.ui.frame_legend.setFixedHeight((mpl_legend_p.p1[1]-mpl_legend_p.p0[1]))
        # self.ui.frame_legend.adjustSize()
        #endregion

        #region data_data

        # add figure mpl_plt1 into frame_spactra_fit
        self.ui.mpl_plt1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit,
            axtype='data',
            # ylabel=r'$\Delta f/n$ (Hz)',
            )
        self.ui.frame_plt1.setLayout(self.set_frame_layout(self.ui.mpl_plt1))


        # add figure mpl_plt2 into frame_spactra_fit
        self.ui.mpl_plt2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit,
            axtype='data',
            # ylabel=r'$\Delta \Gamma$ (Hz)',
            )
        self.ui.frame_plt2.setLayout(self.set_frame_layout(self.ui.mpl_plt2))

        # selector menu
        self.ui.mpl_plt1.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.mpl_plt1.canvas.customContextMenuRequested.connect(lambda position, mpl=self.ui.mpl_plt1, plt_str='plt1': self.mpl_data_open_custom_menu(position, mpl, plt_str))

        self.ui.mpl_plt2.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.mpl_plt2.canvas.customContextMenuRequested.connect(lambda position, mpl=self.ui.mpl_plt2, plt_str='plt2': self.mpl_data_open_custom_menu(position, mpl, plt_str))

        self.ui.radioButton_data_showall.toggled['bool'].connect(self.update_widget)
        self.ui.radioButton_data_showall.clicked.connect(self.clr_mpl_l12)
        self.ui.radioButton_data_showall.clicked.connect(self.update_mpl_plt12)
        self.ui.radioButton_data_showmarked.toggled['bool'].connect(self.update_widget)
        self.ui.radioButton_data_showmarked.toggled['bool'].connect(self.set_mpl_lm_style) # when toggled clicked, this toggled too.
        self.ui.radioButton_data_showmarked.clicked.connect(self.clr_mpl_l12)
        self.ui.radioButton_data_showmarked.clicked.connect(self.update_mpl_plt12)

        # set signals to update plot 1 & 2 options
        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'checkBox_plt1_h' + harm).stateChanged.connect(self.update_widget)
            getattr(self.ui, 'checkBox_plt1_h' + harm).stateChanged.connect(self.update_mpl_plt1)
            getattr(self.ui, 'checkBox_plt1_h' + harm).stateChanged.connect(self.clr_mpl_harm)

            getattr(self.ui, 'checkBox_plt2_h' + harm).stateChanged.connect(self.update_widget)
            getattr(self.ui, 'checkBox_plt2_h' + harm).stateChanged.connect(self.update_mpl_plt2)
            getattr(self.ui, 'checkBox_plt2_h' + harm).stateChanged.connect(self.clr_mpl_harm)

        # set signals to update plot 1 options
        self.ui.comboBox_plt1_optsy.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_plt1_optsy.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_plt1_optsy.currentIndexChanged.connect(self.update_mpl_plt1)
        self.ui.comboBox_plt1_optsx.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_plt1_optsx.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_plt1_optsx.currentIndexChanged.connect(self.update_mpl_plt1)

        self.ui.radioButton_plt1_ref.toggled.connect(self.update_widget)
        self.ui.radioButton_plt1_ref.toggled.connect(self.ui.mpl_plt1.clr_all_lines)
        self.ui.radioButton_plt1_ref.clicked.connect(self.update_mpl_plt1)
        self.ui.radioButton_plt1_samp.toggled.connect(self.update_widget)
        self.ui.radioButton_plt1_samp.toggled.connect(self.ui.mpl_plt1.clr_all_lines)
        self.ui.radioButton_plt1_samp.clicked.connect(self.update_mpl_plt1)

        # set signals to update plot 2 options
        self.ui.comboBox_plt2_optsy.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_plt2_optsy.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_plt2_optsy.currentIndexChanged.connect(self.update_mpl_plt2)
        self.ui.comboBox_plt2_optsx.currentIndexChanged.connect(self.update_widget)
        self.ui.comboBox_plt2_optsx.currentIndexChanged.connect(self.update_data_axis)
        self.ui.comboBox_plt2_optsx.currentIndexChanged.connect(self.update_mpl_plt2)

        self.ui.radioButton_plt2_ref.toggled.connect(self.update_widget)
        self.ui.radioButton_plt2_ref.toggled.connect(self.ui.mpl_plt2.clr_all_lines)
        self.ui.radioButton_plt2_ref.clicked.connect(self.update_mpl_plt2)
        self.ui.radioButton_plt2_samp.toggled.connect(self.update_widget)
        self.ui.radioButton_plt2_samp.toggled.connect(self.ui.mpl_plt2.clr_all_lines)
        self.ui.radioButton_plt2_samp.clicked.connect(self.update_mpl_plt2)

        #endregion


        #region data_mechanics

        # add figure mpl_contour1 into frame_spectra_mechanics_contour1
        self.ui.mpl_contour1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour1,
            axtype='contour'
            )
        self.ui.frame_spectra_mechanics_contour1.setLayout(self.set_frame_layout(self.ui.mpl_contour1))

        # add figure mpl_contour2 into frame_spectra_mechanics_contour2
        self.ui.mpl_contour2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour2,
            axtype='contour',
            )
        self.ui.frame_spectra_mechanics_contour2.setLayout(self.set_frame_layout(self.ui.mpl_contour2))

        self.ui.pushButton_spectra_mechanics_clear.clicked.connect(self.del_prop_plot)
        #endregion


        #region status bar

        #### add widgets to status bar. from left to right
        # move progressBar_status_interval_time to statusbar
        self.ui.progressBar_status_interval_time.setAlignment(Qt.AlignCenter)
        self.ui.statusbar.addPermanentWidget(self.ui.progressBar_status_interval_time)
        # move label_status_pts to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_pts)
        # move pushButton_status_reftype to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.pushButton_status_reftype)
         # move pushButton_status_signal_ch to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.pushButton_status_signal_ch)
       # move pushButton_status_temp_sensor to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.pushButton_status_temp_sensor)
        # move label_status_f0RNG to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_f0RNG)

        #endregion


        #region action group

        # add menu to toolbutton

        # toolButton_settings_data_regenerate_from_raw
        # create menu: menu_settings_data_refit
        self.ui.menu_settings_data_refit = QMenu(self.ui.toolButton_settings_data_regenerate_from_raw)
        self.ui.menu_settings_data_refit.addAction(self.ui.actionRegenerate_allsamp)
        self.ui.actionRegenerate_allsamp.triggered.connect(lambda: self.regenerate_from_raw(chn_name='samp', mode='all'))
        # self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_markedsamp)
        # self.ui.actionFit_markedsamp.triggered.connect(lambda: self.regenerate_from_raw(chn_name='samp', mode='marked'))
        self.ui.menu_settings_data_refit.addAction(self.ui.actionRegenerate_allref)
        self.ui.actionRegenerate_allref.triggered.connect(lambda: self.regenerate_from_raw(chn_name='ref', mode='all'))
        # self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_markedref)
        # self.ui.actionFit_markedref.triggered.connect(lambda: self.regenerate_from_raw(chn_name='ref', mode='marked'))
        # self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_selected)
        # self.ui.actionFit_all.triggered.connect(self.)
        # add menu to toolbutton
        self.ui.toolButton_settings_data_regenerate_from_raw.setMenu(self.ui.menu_settings_data_refit)

        # toolButton_settings_mechanics_solve
        # create menu: menu_settings_mechanics_solve
        self.ui.menu_settings_mechanics_solve = QMenu(self.ui.toolButton_settings_mechanics_solve)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_all)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_marked)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_new)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_test)
        self.ui.actionSolve_all.triggered.connect(self.mech_solve_all)
        self.ui.actionSolve_marked.triggered.connect(self.mech_solve_marked)
        self.ui.actionSolve_new.triggered.connect(self.mech_solve_new)
        self.ui.actionSolve_test.triggered.connect(self.mech_solve_test)
        # add menu to toolbutton
        self.ui.toolButton_settings_mechanics_solve.setMenu(self.ui.menu_settings_mechanics_solve)

        # toolButton_spectra_mechanics_plotrows
        self.ui.actionRows_Time.triggered.connect(self.mechanics_plot_r_time)
        self.ui.actionRows_Temp.triggered.connect(self.mechanics_plot_r_temp)
        self.ui.actionRows_Index.triggered.connect(self.mechanics_plot_r_idx)
        self.ui.actionRow_s1_Row_s2.triggered.connect(self.mechanics_plot_r1_r2)
        self.ui.actionRow_s2_Row_s1.triggered.connect(self.mechanics_plot_r2_r1)
        # create menu: menu_spectra_mechanics_plotrows
        self.ui.menu_spectra_mechanics_plotrows = QMenu(self.ui.toolButton_spectra_mechanics_plotrows)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRows_Time)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRows_Temp)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRows_Index)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRow_s1_Row_s2)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRow_s2_Row_s1)
        # add menu to toolbutton
        self.ui.toolButton_spectra_mechanics_plotrows.setMenu(self.ui.menu_spectra_mechanics_plotrows)


        # set QAction
        self.ui.actionLoad_Settings.triggered.connect(self.on_triggered_load_settings)
        self.ui.actionExport_Settings.triggered.connect(self.on_triggered_export_settings)
        self.ui.actionLoad_Exp.triggered.connect(self.on_triggered_load_exp)
        self.ui.actionNew_Exp.triggered.connect(self.on_triggered_new_exp)
        self.ui.actionSave.triggered.connect(self.on_triggered_actionSave)
        self.ui.actionSave_As.triggered.connect(self.on_triggered_actionSave_As)
        self.ui.actionExport.triggered.connect(self.on_triggered_actionExport)
        self.ui.actionReset.triggered.connect(self.on_triggered_actionReset)
        self.ui.actionClear_All.triggered.connect(self.on_triggered_actionClear_All)
        self.ui.actionOpen_MyVNA.triggered.connect(self.on_triggered_actionOpen_MyVNA)
        # import QCM-D
        self.ui.actionImport_QCM_D.triggered.connect(self.on_triggered_actionImport_QCM_D)
        # import QCM-Z
        self.ui.actionImport_QCM_Z.triggered.connect(self.on_triggered_actionImport_QCM_Z)
        # about QCM_py
        self.ui.actionAbout_QCM_py.triggered.connect(self.msg_about)


        #endregion


        #region ###### add Matplotlib figures in to frames ##########

        # # create an empty figure and move its toolbar to TopToolBarArea of main window
        # self.ui.mpl_dummy_fig = MatplotlibWidget()
        # self.addToolBar(Qt.TopToolBarArea, self.ui.mpl_dummy_fig.toolbar)
        # self.ui.mpl_dummy_fig.hide() # hide the figure

        #endregion



    #region #########  functions ##############

    
    def get_mp_cores(self):
        ''' return the number of cores to use for multiprocessing
        '''
        core_count = multiprocessing.cpu_count()
        # core_count = len(os.sched_getaffinity(0)) # number of usable cpus
        if core_count is None:
            return 1
        core_config = config_default['multiprocessing_cores']

        if core_config == 0: # use all cores
            core_to_use =  core_count
        elif core_config > 0: # use setting cores
            core_to_use =  min(core_config, core_count)
        elif core_config < 0: # subtract the number
            core_to_use =  max(1, core_count + core_config)
        # print('use {} cores'.format(core_to_use))

        return core_to_use


    def link_tab_page(self, tab_idx):
        self.UITab = tab_idx
        if tab_idx in [0, 2]: # link settings_control to spectra_show and data_data
            self.ui.stackedWidget_spectra.setCurrentIndex(0)
            self.ui.stackedWidget_data.setCurrentIndex(0)
        elif tab_idx in [1]: # link settings_settings and settings_data to spectra_fit
            self.ui.stackedWidget_spectra.setCurrentIndex(1)
            self.ui.stackedWidget_data.setCurrentIndex(0)
        elif tab_idx in [3]: # link settings_mechanics to spectra_mechanics and data_mechanics
            self.ui.stackedWidget_spectra.setCurrentIndex(2)
            self.ui.stackedWidget_data.setCurrentIndex(1)


    def move_to_col(self, obj, parent, row_text, width=[], col=1):
        if width: # set width of obj
            obj.setMaximumWidth(width)
        # find item with row_text
        item = self.find_text_item(parent, row_text)
        # insert the combobox in to the 2nd column of row_text
        parent.setItemWidget(item, col, obj)

    def find_text_item(self, parent, text):
        '''
        find item with 'text' in widgets e.g.: treeWidget, tableWidget
        return a item
        Make sure the text is unique in the widget
        if not, return None
        '''
        item = parent.findItems(text, Qt.MatchExactly | Qt.MatchRecursive, 0)
        if len(item) == 1:
            item = item[0]
        else:
            item = None
        return item


    def set_frame_layout(self, widget):
        '''set a dense layout for frame with a single widget'''
        vbox = QGridLayout()
        vbox.setContentsMargins(0, 0, 0, 0) # set layout margins (left, top, right, bottom)
        vbox.addWidget(widget)
        return vbox


    ########## action functions ##############
    # @pyqtSlot['bool']
    def on_clicked_pushButton_runstop(self, checked):
        if checked:
            # turn off manual refit mode
            self.set_manual_refit_mode(val=False)

            # check recording chns
            if not self.settings['checkBox_activechn_samp'] and not self.settings['checkBox_activechn_ref']:
                self.ui.pushButton_runstop.setChecked(False)
                return

            # if no checked harmonic, stop
            harm_list = self.get_all_checked_harms()
            if not harm_list:
                self.ui.pushButton_runstop.setChecked(False)
                print('No harmonic is checked for recording!')
                # TODO update statusbar
                return
            # check filename if avaialbe
            if not self.data_saver.path: # no filename
                if self.tempPath: # new file name is set
                    path = self.tempPath
                else: # no file name is set. save data to a temp file
                    path = os.path.abspath(os.path.join(config_default['unsaved_path'], datetime.datetime.now().strftime(config_default['unsaved_filename']) + '.h5'))
                    # display path in lineEdit_datafilestr
                    self.set_filename(fileName=path)
                self.data_saver.init_file(
                    path=path,
                    settings=self.settings,
                    t0=self.settings['dateTimeEdit_reftime']
                ) # save to unsaved folder
                # update exp_ref in UI
                self.load_refsource()
                self.update_refsource()

            # this part is auto reset reftime to current time
            # it only works when dateTimeEdit_reftime & pushButton_resetreftime is hidden
            if not self.ui.pushButton_resetreftime.isVisible() and not self.ui.dateTimeEdit_reftime.isVisible() and self.ui.pushButton_resetreftime.isEnabled() and self.ui.dateTimeEdit_reftime.isEnabled():
                self.reset_reftime() # use current time as t0

            # disable features
            self.disable_widgets(
                'pushButton_runstop_disable_list'
            )

            # cmd diary?

            # test scheduler? start/end increasement

            # start the timer
            self.timer.start(0)

            self.ui.pushButton_runstop.setText('STOP')
        else:
            # set text on button for waitiong
            self.ui.pushButton_runstop.setText('FINISHING...')
            # stop running timer and/or test
            # logger.info(self.timer.isActive()) 
            self.timer.stop()
            # logger.info(self.timer.isActive()) 

            # stop bartimer
            self.bartimer.stop()
            # reset progressbar
            self.set_progressbar(val=0, text='')

            # # wait for data_collection fun finish (self.idle == True)
            # while self.idle == False:
            #     loop = QEventLoop()
            #     QTimer.singleShot(1000, loop.quit)
            #     loop.exec_()
            #     logger.info('looping') 

            # write dfs and settings to file
            if self.idle == True: # Timer stopped while timeout func is not running (test stopped while waiting)
                self.process_saving_when_stop()
                logger.info('data saved while waiting') 


    def process_saving_when_stop(self):
        '''
        process saving fitted data when test is stopped
        '''
        # save data
        self.data_saver.save_data()
        # write UI information to file
        self.data_saver.save_data_settings(settings=self.settings) # TODO add exp_ref

        self.counter = 0 # reset counter

        logger.info('data saver samp') 
        logger.info(self.data_saver.samp) 

        # enable features
        self.enable_widgets(
            'pushButton_runstop_enable_list'
        )

        #
        self.ui.pushButton_runstop.setText('START RECORD')


    # @pyqtSlot()
    def reset_reftime(self):
        '''
        set time in dateTimeEdit_reftime
        '''
        # use qt use python deal with datetime. But show the time with QdatetimeEdit
        self.ui.dateTimeEdit_reftime.setDateTime(datetime.datetime.now())


    def on_dateTimeChanged_dateTimeEdit_reftime(self, datetime):
        '''
        get time in dateTimeEdit_reftime and save it to self.settings
        '''
        self.settings['dateTimeEdit_reftime'] = self.ui.dateTimeEdit_reftime.dateTime().toPyDateTime().strftime(config_default['time_str_format'])
        logger.info(self.settings['dateTimeEdit_reftime']) 
        self.ui.label_settings_data_t0.setText(self.settings['dateTimeEdit_reftime'][:-3]) # [:-3] remove the extra 000 at the end
        self.data_saver.set_t0(t0=self.settings['dateTimeEdit_reftime'])


    def on_dateTimeChanged_dateTimeEdit_t0shifted(self, datetime):
        '''
        get time in dateTimeEdit_settings_data_t0shifted
        and save it to self.settings and data_saver
        '''
        self.settings['dateTimeEdit_settings_data_t0shifted'] = self.ui.dateTimeEdit_settings_data_t0shifted.dateTime().toPyDateTime().strftime(config_default['time_str_format'])
        logger.info(self.settings['dateTimeEdit_settings_data_t0shifted']) 

        self.data_saver.set_t0(t0_shifted=self.settings['dateTimeEdit_settings_data_t0shifted'])


    def reset_shiftedt0(self):
        '''
        reset shiftedt0 to t0
        '''
        self.ui.dateTimeEdit_settings_data_t0shifted.setDateTime(datetime.datetime.strptime(self.settings['dateTimeEdit_reftime'], config_default['time_str_format']))


    def save_data_saver_sampidx(self):
        '''
        set data_saver.exp_ref['samp_ref'][2]
        '''
        self.save_data_saver_chn_idx('samp')


    def save_data_saver_refidx(self):
        '''
        set data_saver.exp_ref['ref_ref'][2]
        '''
        self.save_data_saver_chn_idx('ref')


    def save_data_saver_chn_idx(self, chn_name):
        '''
        save chn_idx to data_saver.exp_ref[chn_name + '_ref'][2]
        '''
        chn_idx_str = self.settings['lineEdit_settings_data_'+ chn_name + 'idx']
        chn_idx = self.data_saver.get_idx(chn_name).values.tolist() # use list of index for comperison
        # convert chn_idx/ref_idx from str to a list of int
        chn_idx = UIModules.index_from_str(chn_idx_str, chn_idx, join_segs=False)

        # save to data_saver
        self.data_saver.set_chn_idx(chn_name, chn_idx)


    def save_data_saver_sampref(self):
        '''
        set the data_saver.exp_ref['samp_ref']
        '''
        self.save_data_saver_refsource('samp')


    def save_data_saver_refref(self):
        '''
        set the data_saver.exp_ref['ref_ref']
        '''
        self.save_data_saver_refsource('ref')


    def save_data_saver_refsource(self, chn_name): 
        '''
        set the data_saver.exp_ref[chn_name]
        '''
        logger.info('save_data_saver_refsource') 
        logger.info('chn_name %s', chn_name) 
        ref_source = self.settings['comboBox_settings_data_'+ chn_name + 'refsource']
        ref_idx_str = self.settings['lineEdit_settings_data_'+ chn_name + 'refidx']
        logger.info('ref_source %s', ref_source) 
        logger.info('ref_idx_str: %s %s', ref_idx_str, type(ref_idx_str))

        # chn_queue_list = list(self.data_saver.get_queue_id(ref_source).tolist()) # list of available index in the target chn
        ref_chn_idx = self.data_saver.get_idx(ref_source).values.tolist() # use list of index for comperison
        logger.info('ref_chn_idx: %s type %s', ref_chn_idx, type(ref_chn_idx))

        # convert ref_idx from str to a list of int

        ref_idx = UIModules.index_from_str(ref_idx_str, ref_chn_idx, join_segs=False)
        logger.info('ref_idx: %s type %s', ref_idx, type(ref_idx))

        # check if the ref chn is empty, then ref to chn itself
        if len(ref_chn_idx) == 0: # no data in the setting ref chn
            logger.info('refsource is empty.')
            # set ref to chn itself
            ref_source = chn_name
            self.settings['comboBox_settings_data_' + chn_name + 'refsource'] = chn_name

        if (not ref_idx) and ((list(self.data_saver.get_queue_id('samp')) !=  list(self.data_saver.get_queue_id('ref')))): # samp and ref were not collected together
            ref_idx = [0]
            # getattr(self.ui, 'lineEdit_settings_data_'+ chn_name + 'refidx').setText('[0]')
            self.settings['lineEdit_settings_data_'+ chn_name + 'refidx'] = '[0]'


        self.update_refsource()

        # # save to data_saver
        # self.data_saver.exp_ref[chn_name + '_ref'][0] = ref_source
        # self.data_saver.exp_ref[chn_name + '_ref'][1] = ref_idx

        # save to data_saver and
        # update and set reference
        self.data_saver.set_ref_set(chn_name, ref_source, ref_idx, df=None) # TODO add df if ref_source == exp

        # refresh mpl_plt<n>
        self.update_mpl_plt12()


    def recalc_refs(self):
        '''
        recalculate delf and delg by reference set saved in data_saver
        '''
        self.data_saver.calc_fg_ref('samp', mark=False) # False or True??
        self.data_saver.calc_fg_ref('ref', mark=False) # False or True??


    def on_triggered_actionOpen_MyVNA(self):
        '''
        open myVNA.exe
        '''
        if UIModules.system_check() != 'win32': # not windows
            return

        myvna_path = self.settings.get('vna_path', '')
        logger.info('myvna_path: %s\n', myvna_path) 
        if myvna_path and os.path.exists(myvna_path): # user defined myVNA.exe path exists and correct
            logger.info('vna_path in self.settings') 
            pass
        else: # use default path list
            logger.info('vna_path try config_default') 
            for myvna_path in config_default['vna_path']:
                if os.path.exists(myvna_path):
                    logger.info('vna_path in config_default') 
                    break
                else:
                    logger.info('vna_path not found') 
                    myvna_path = ''

        logger.info('myvna_path: %s\n', myvna_path) 
        if myvna_path:
            logger.info('vna_path to open exe') 
            subprocess.call(myvna_path) # open myVNA
        else:
            logger.info('vna_path msg box') 
            process = self.process_messagebox(
                text='Failed to open myVNA.exe',
                message=['Cannot find myVNA.exe in: \n{}\nPlease add the path for "vna_path" in "settings_default.json"!'.format('\n'.join(config_default['vna_path'])),
                'The format of the path should like this:',
                r'"C:\\Program Files (x86)\\G8KBB\\myVNA\\myVNA.exe"'
                ],
                opts=False,
                forcepop=True,
            )


    def on_triggered_actionImport_QCM_D(self):
        '''
        import QCM-D data for calculation
        '''
        process = self.process_messagebox(message=['Load QCM-D data!'])

        if not process:
            return

        fileName = self.openFileNameDialog(title='Choose an existing file to append', filetype=config_default['external_qcm_datafiletype']) # !! add path of last opened folder

        if fileName:
            self.data_saver.import_qcm_with_other_format('qcmd', fileName, config_default, settings=self.settings)

            self.data_saver.saveflg = True

            self.on_triggered_actionReset(settings=self.data_saver.settings)

            self.disable_widgets(
                'pushButton_appendfile_disable_list',
            )

            logger.info('path: %s', self.data_saver.path)
            # change the displayed file directory in lineEdit_datafilestr and save it to data_saver
            self.set_filename(os.path.splitext(fileName)[0]+'.h5')
            

    def on_triggered_actionImport_QCM_Z(self):
        '''
        import QCM-Z data for calculation
        '''
        process = self.process_messagebox(message=['Load QCM-Z data!'])

        if not process:
            return

        fileName = self.openFileNameDialog(title='Choose an existing file to append', filetype=config_default['external_qcm_datafiletype']) # !! add path of last opened folder

        if fileName:
            self.data_saver.import_qcm_with_other_format('qcmz', fileName, config_default, settings=self.settings)


    def msg_about(self):
        '''
        This function opens a message box to display the version information
        '''
        msg_text = []
        msg_text.append('Version: {}'.format(_version.__version__))
        msg_text.append('Authors: {}'.format(' ,'.join(_version.__authors__)))
        msg_text.append('Contact: {}'.format(_version.__contact__))
        msg_text.append('Copyright: {}'.format(_version.__copyright__))
        msg_text.append("Source: <a href='{0}'>{0}</a>".format(_version.__source__))
        msg_text.append("Report issues: <a href='{0}'>{0}</a>".format(_version.__report__))
        msg_text.append('License: {}'.format(_version.__license__))
        msg_text.append('Date: {}'.format(_version.__date__))

        buttons = QMessageBox.Ok

        msg = QMessageBox()
        msg.setTextFormat(Qt.RichText)
        # msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('About ' + _version.__projectname__)
        msg.setText('<b>{} {}<\b>'.format(_version.__projectname__, _version.__version__))
        msg.setInformativeText('<P>'.join(msg_text))
        msg.setStandardButtons(buttons)
        msg.exec_()


    def set_recording_time(self):
        '''
        the idea here is try to set record interval
        if record interval is changed, set scan interval
        '''
        # get text
        record_interval = self.settings['spinBox_recordinterval']
        refresh_resolution = self.settings['spinBox_refreshresolution']
        scaninterval = self.settings['spinBox_scaninterval']
        
        # sender
        if not self.sender():
            sender = None
        else:
            sender = self.sender().objectName()
        #convert to flot
        
        if sender == 'spinBox_recordinterval': 
            # change scan interval
            self.ui.spinBox_scaninterval.setValue(int(record_interval / refresh_resolution))
        elif sender in ['spinBox_refreshresolution', 'spinBox_scaninterval', None]:
            # change record interval
            self.ui.spinBox_recordinterval.setValue(int(scaninterval * refresh_resolution))


    ## functions for open and save file
    def openFileNameDialog(self, title, path='', filetype=config_default['default_datafiletype']):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, title, path, filetype, options=options)
        if fileName:
            logger.info(fileName)
        else:
            fileName = ''
        return fileName

    # def openFileNamesDialog(self, title, path=''):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     files, _ = QFileDialog.getOpenFileNames(self,title, "","All Files (*);;Python Files (*.py)", options=options)
    #     if files:
    #         logger.info(files) 


    def saveFileDialog(self, title, path='', filetype=config_default['default_datafiletype']):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,title, os.path.splitext(path)[0], filetype, options=options)
        if fileName:
            logger.info(fileName)
        else:
            fileName = ''
        return fileName


    def on_triggered_new_exp(self):
        process = self.process_messagebox(message=['Create a new experiment!'])

        if not process:
            return

        fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        if fileName:
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(fileName)
            self.tempPath = fileName #
            # reset dateTimeEdit_reftime
            self.reset_reftime()
            # set enable
            self.enable_widgets(
                'pushButton_newfile_enable_list',
            )


    def on_triggered_load_exp(self):

        process = self.process_messagebox(message=['Load new experiment data!'])

        if not process:
            return

        fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
        if fileName:
            # load UI settings
            self.data_saver.load_file(fileName) # load factors from file to data_saver
            if not self.data_saver.settings: # failed to load file
                process = self.process_messagebox(
                    text='Failed to load File',
                    message=['Failed to load file:\n{}\nPlease check if the right file is selected!'.format(fileName)],
                    opts=False,
                    forcepop=True,
                )
            else:
                self.on_triggered_actionReset(settings=self.data_saver.settings)

                self.disable_widgets(
                    'pushButton_appendfile_disable_list',
                )

                # change the displayed file directory in lineEdit_datafilestr and save it to data_saver
                self.set_filename(fileName)


    # open folder in explorer
    # methods for different OS could be added
    def on_clicked_pushButton_gotofolder(self):
        file_path = self.ui.lineEdit_datafilestr.text() #TODO replace with reading from settings dict
        path = os.path.abspath(os.path.join(file_path, os.pardir)) # get the folder of the file
        UIModules.open_file(path)

    #
    def on_triggered_load_settings(self):

        process = self.process_messagebox(message=['Load settings from other file!'])

        if not process:
            return

        fileName = self.openFileNameDialog('Choose a file to load its settings', path=self.data_saver.path, filetype=config_default['default_settings_load_filetype']) # TODO add path of last opened folder

        if fileName:

            # load settings from file
            name, ext = os.path.splitext(fileName)
            if ext == '.h5':
                settings = self.data_saver.load_settings(path=fileName)
            elif ext == '.json':
                with open(fileName, 'r') as f:
                    settings = json.load(f)
            else:
                settings = None
            # reset default settings
            # replase keys in self.settings with those in settings_default
            if not settings:
                process = self.process_messagebox(
                    text='Failed to load Settings',
                    message=['Failed to load settings:\n{}\nPlease check if the right file is selected!'.format(fileName)],
                    opts=False,
                    forcepop=True,
                )
                print('File with wrong fromat!')
                return
            else:
                # remove some k
                for key, val in settings.items():
                    self.settings[key] = val

            # reload widgets' setup
            self.load_settings()


    def on_triggered_export_settings(self):
        process = self.process_messagebox(message=['Export settings to a file!'])
        if not process:
            return

        fileName = self.saveFileDialog('Choose a file to save settings', path=self.data_saver.path, filetype=config_default['default_settings_export_filetype']) # TODO add path of last opened folder

        if fileName:
            # load settings from file
            name, ext = os.path.splitext(fileName)
            if ext == '.json':
                with open(fileName, 'w') as f:
                    settings = self.settings.copy()
                    # remove some keys dependent on each test
                    settings.pop('dateTimeEdit_reftime', None) # time reference
                    settings.pop('dateTimeEdit_settings_data_t0shifted', None) # time shift
                    # settings.pop('lineEdit_datafilestr', None) # file path (it is not in settings)

                    line = json.dumps(settings, indent=4) + "\n"
                    f.write(line)
                print('Settings were exported as json file.')
                #TODO statusbar


    def on_triggered_actionSave(self):
        '''
        save current data to file if file has been opened
        '''
        # turn off manual refit mode
        self.set_manual_refit_mode(val=False)

        if self.data_saver.path: # there is file
            self.data_saver.save_data_settings(settings=self.settings)
            print('Data has been saved to file!')
        elif (not self.data_saver.path) & len(self.tempPath)>0: # name given but file not been created (no data)
            print('No data collected!')
        else:
            print('No file information!')


    def on_triggered_actionSave_As(self):
        ''' save current data to a new file  '''

        # turn off manual refit mode
        self.set_manual_refit_mode(val=False)

        # export data to a selected form
        fileName = self.saveFileDialog(title='Choose a new file', filetype=config_default['default_datafiletype'], path=self.data_saver.path) # !! add path of last opened folder
        # codes for data exporting
        if fileName:
            if self.data_saver.path: # there is file

                # copy file
                try:
                    shutil.copyfile(self.data_saver.path, fileName)
                except Exception as e:
                    print('Failed to copy file!')
                    print(e)
                    return
                # change the path in data_saver
                self.data_saver.path = fileName
                # save modification to new file
                self.data_saver.save_data_settings(settings=self.settings)


    def on_triggered_actionExport(self):
        ''' export data to a selected format '''
        process = self.process_messagebox(message=['Export data to a selected format!'])
        if not process:
            return

        fileName = self.saveFileDialog(title='Choose a file and data type', filetype=config_default['export_datafiletype'], path=self.data_saver.path) # !! add path of last opened folder
        # codes for data exporting
        if fileName:
            print('Exporting data ...')
            self.data_saver.data_exporter(fileName) # do the export
            print('Data is exported.')
        

    def process_messagebox(self, text='Your selection was paused!', message=[], opts=True, forcepop=False):
        '''
        check is the experiment is ongoing (self.timer.isActive()) and if data is saved (self.data_saver.saveflg)
        and pop up a messageBox to ask if process
        message: list of strings
        forcepop: True, the message will popup anyway

        return process: True/False for checking
        '''

        process = True

        if self.timer.isActive() or (self.data_saver.saveflg == False) or forcepop:
            if self.data_saver.saveflg == False:
                message.append('There is data unsaved!')
            if self.timer.isActive():
                message.append('Test is Running! You may stop the test first.')
                buttons = QMessageBox.Ok
            else:
                if not opts:
                    buttons = QMessageBox.Ok
                else:
                    message.append('Do you want to continue anyway?')
                    buttons = QMessageBox.Yes | QMessageBox.Cancel

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(text)
            msg.setInformativeText('\n'.join(message))
            msg.setWindowTitle(_version.__projectname__ + ' Message')
            msg.setStandardButtons(buttons)
            retval = msg.exec_()

            if retval == QMessageBox.Yes:
                if self.timer.isActive():
                    # stop test
                    self.ui.pushButton_runstop.setChecked(False)

                process = True
            else:
                process = False

        if process:
            # turn off manual refit mode
            self.set_manual_refit_mode(val=False)

        return process


    def on_triggered_actionReset(self, settings=None):
        """
        reset MainWindow
        if settings is given, it will load the given settings (load settings)
        """

        process = self.process_messagebox()

        if not process:
            return

        # # turn off manual refit mode
        # self.set_manual_refit_mode(val=False)

        # clear spectra_fit items
        self.clr_spectra_fit()

        # clear data table on mechanics tab
        self.ui.tableWidget_spectra_mechanics_table.clearContents()

        # set widgets enabled by using the disabled list
        self.enable_widgets(
            'pushButton_runstop_disable_list',
            'pushButton_appendfile_disable_list',
        )

        # reset default settings
        # replase keys in self.settings with those in settings_default
        if not settings:
            for key, val in settings_default.items():
                self.settings[key] = val
        else:
            for key, val in settings.items():
                self.settings[key] = val
        self.peak_tracker = PeakTracker.PeakTracker(max_harm=self.settings['max_harmonic'])
        self.vna_tracker = VNATracker()

        if not settings: # reset UI
            self.data_saver = DataSaver.DataSaver(ver=_version.__version__, settings=self.settings)
            # enable widgets
            self.enable_widgets(
                'pushButton_runstop_disable_list',
                'pushButton_appendfile_disable_list',
            )

        # reload widgets' setup
        self.load_settings()

        # clear fileName
        self.set_filename()

        # reset  status pts
        self.set_status_pts()


    def on_triggered_actionClear_All(self):
        '''
        clear all data
        '''
        if not self.data_saver.path: # no data
            return

        process = self.process_messagebox(message=['All data in the file will be deleted!'], forcepop=True)

        if not process:
            return

        logger.info(self.data_saver.path) 
        # re-initiate file
        self.data_saver.init_file(self.data_saver.path, settings=self.settings, t0=self.settings['dateTimeEdit_reftime'])

        # enable widgets
        self.enable_widgets(
            'pushButton_runstop_disable_list',
            'pushButton_appendfile_disable_list',
        )

        # clear spectra_fit items
        self.clr_spectra_fit()

        # in most condition, you do not want to clear the sample description
        # # clear plainTextEdit
        # self.ui.plainTextEdit_settings_sampledescription.clear()


    def on_triggered_del_menu(self, chn_name, plt_harms, sel_idx_dict, mode, marks):
        '''
        this function delete the raw data by given variables
        
        '''

        # define the alert by mode (str)
        alert_dic = {
            # mode: alert string
            'all': 'all showing data',
            'marked': 'showing marked data',
            'selpts': 'selected data',
            'selidx': 'selected indices of all showing harmonics',
            'selharm': 'all data of selected harmonics',
        }

        if not self.data_saver.path: # no data
            return

        process = self.process_messagebox(message=['Raw data of \n{}\nin the file will be PERMANENTLY DELETED!'.format(alert_dic.get(mode, None))], forcepop=True)

        if not process:
            return

        self.data_saver.selector_del_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, mode, marks))


    def clr_spectra_fit(self):
        # delete prop plot
        self.del_prop_plot()
        # clear all mpl objects
        self.clear_all_mpl()
        # clear plainTextEdit
        self.ui.plainTextEdit_spectra_fit_result.clear()


    def clear_all_mpl(self):
        '''
        clear lines in all mpls
        '''
        # find all mpl objects
        mpl_list = self.findChildren(MatplotlibWidget)
        logger.info(mpl_list) 
        # clear mpl_sp
        for mpl in mpl_list:
            mpl.clr_lines()


    def set_status_pts(self):
        '''
        set status bar label_status_pts
        '''
        # self.ui.label_status_pts.setText(str(self.data_saver.get_npts()))
        logger.info(str(self.data_saver.get_npts())) 
        try:
            # logger.info(10) 
            self.ui.label_status_pts.setText(str(self.data_saver.get_npts()))
            # logger.info(11) 
        except:
            # logger.info(21) 
            self.ui.label_status_pts.setText('pts')
            # logger.info(22) 


    def show_widgets(self, *args):
        '''
        show widgets in given args
        args: list of names
        '''
        logger.info('show') 
        logger.info(args) 
        for name_list in args:
            for name in config_default[name_list]:
                logger.info(name) 
                if (name not in config_default['version_hide_list']) and (name_list != 'version_hide_list'):
                    # getattr(self.ui, name).show()
                    getattr(self.ui, name).setVisible(True)


    def hide_widgets(self, *args):
        '''
        hide widgets in given args
        args: list of names
        '''
        logger.info('hide') 
        logger.info(args) 
        for name_list in args:
            for name in config_default[name_list]:
                logger.info(name) 
                if (name not in config_default['version_hide_list']) or (name_list == 'version_hide_list'):
                    # getattr(self.ui, name).hide()
                    getattr(self.ui, name).setVisible(False)


    def enable_widgets(self, *args):
        '''
        enable/ widgets in given args
        args: list of names
        '''
        logger.info(args) 
        for name_list in args:
            for name in config_default[name_list]:
                getattr(self.ui, name).setEnabled(True)
                # the following check if is hidden by the version may not be necessary in some case. e.g. hide ref_time widgets need them to be disabled
                # if name not in config_default['version_hide_list'] or name_list == 'version_hide_list':
                #     getattr(self.ui, name).setEnabled(True)


    def disable_widgets(self, *args):
        '''
        disable widgets in given args
        args: list of names
        '''
        logger.info(args) 
        for name_list in args:
            for name in config_default[name_list]:
                getattr(self.ui, name).setEnabled(False)
                # the following check if is hidden by the version may not be necessary in some case. e.g. hide ref_time widgets need them to be disabled
                # if name not in config_default['version_hide_list'] or name_list == 'version_hide_list':
                #     getattr(self.ui, name).setEnabled(False)


    def mech_splitter_vis(self, signal):
        '''
        '''
        # print(signal)

        # set visibility with hide_widget hide_widgets & show_widgets
        # done in Qt Designer

        # set handle of splitter_spectra_mechanics idx = 0 to hide and idx = 1 (index of layout_mech_table) to show
        pass


    def set_filename(self, fileName=''):
        '''
        set self.data_saver.path and lineEdit_datafilestr
        '''
        self.data_saver.path = fileName
        self.tempPath = '' # init class file path information
        self.ui.lineEdit_datafilestr.setText(fileName)


    def add_table_headers(self, tablename, vnames, hnames):
        '''
        find vnames/hnames iterms as dict
        add to an empty table self.ui.<tablename>
        '''
        # get bable object
        table = getattr(self.ui, tablename)
        # get n of rows
        nrows = len(vnames)
        # get n of columns
        ncols = len(hnames)

        _translate = QCoreApplication.translate
        
        # set number of coulmns in table
        table.setColumnCount(ncols)
        table.setRowCount(nrows)

        ## add  headers to table
        ## add vertival headers to table
        for r, (key, val) in enumerate(vnames.items()):
            table.setVerticalHeaderItem(r, QTableWidgetItem())
            table.verticalHeaderItem(r).setText(_translate('MainWindow', val))

        # horizontal headers
        for c, (key, val) in enumerate(hnames.items()):
            table.setHorizontalHeaderItem(c, QTableWidgetItem())
            table.horizontalHeaderItem(c).setText(_translate('MainWindow', val))


    def on_acctiontriggered_slider_spanctrl(self, value):
        '''
        disable the actions other than mouse dragging
        '''
        # logger.info(value) 
        if value < 7: # mouse dragging == 7
            # reset slider to 1
            self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)


    def on_changed_slider_spanctrl(self):
        # get slider value
        n = 10 ** (self.ui.horizontalSlider_spectra_fit_spanctrl.value() / 10)
        # format n
        if n >= 1:
            # n = f'{round(n)} *'
            n = '{} *'.format(min(config_default['span_ctrl_steps'], key=lambda x:abs(x-n))) # python < 3.5
        else:
            # n = f'1/{round(1/n)} *'
            n = '1/{} *'.format(min(config_default['span_ctrl_steps'], key=lambda x:abs(x-1/n))) # python < 3.5
        # set label_spectra_fit_zoomtimes value
        self.ui.label_spectra_fit_zoomtimes.setText(str(n))


    def on_released_slider_spanctrl(self):

        # get slider value
        n = 10 ** (self.ui.horizontalSlider_spectra_fit_spanctrl.value() / 10)
        # format n
        if n >= 1:
            n = min(config_default['span_ctrl_steps'], key=lambda x:abs(x-n))
        else:
            n = 1/min(config_default['span_ctrl_steps'], key=lambda x:abs(x-1/n))

        # get f1, f2
        # f1, f2 = self.ui.mpl_spectra_fit.ax[0].get_xlim()
        f1, f2 = self.get_freq_span()
        # convert start/end (f1/f2) to center/span (fc/fs)
        fc, fs = UIModules.converter_startstop_to_centerspan(f1, f2)
        # multiply fs
        fs = fs * n
        # fc/fs back to f1/f2
        f1, f2 = UIModules.converter_centerspan_to_startstop(fc, fs)

        # set lineEdit_scan_harmstart & lineEdit_scan_harmend
        self.ui.lineEdit_scan_harmstart.setText(str(f1)) # in Hz
        self.ui.lineEdit_scan_harmend.setText(str(f2)) # in Hz

        # reset xlim to active on_fit_lims_change
        self.ui.mpl_spectra_fit.ax[0].set_xlim(f1, f2)

        # # update limit of active harmonic
        # self.on_editingfinished_harm_freq()

        # # get new data
        # f, G, B = self.spectra_fit_get_data()

        # # plot
        # self.tab_spectra_fit_update_mpls(f, G, B)

        # reset slider to 1
        self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)


    def span_check(self, harm=None, f1=None, f2=None):
        '''
        check if lower limit ('f1' in Hz) and upper limit ('f2' in Hz) in base freq +/- BW of harmonic 'harm'
        if out of the limit, return the part in the range
        and show alert in statusbar
        NOTE: if f1 and/or f2 isnan, the nan values will returned and leave the nan value to self.set_freq_span to ckeck!
        '''
        if harm is None:
            harm = self.settings_harm
        # get freq_range
        bf1, bf2 = self.settings['freq_range'][harm] # in Hz
        # check f1, and f2

        if f1 and (f1 < bf1 or f1 >= bf2): # f1 out of limt
            f1 = bf1
            #TODO update statusbar 'lower bound out of limit and reseted. (You can increase the range in settings)'
        if f2 and (f2 > bf2 or f2 <= bf1): # f2 out of limt
            f2 = bf2
            #TODO update statusbar 'upper bond out of limit and reseted. (You can increase the range in settings)'
        if f1 and f2 and (f1 >= f2):
            f2 = bf2

        return [f1, f2]


    def get_spectraTab_mode(self):
        '''
        get the current UI condition from attributes and
        set the mode for spectra_fit
        '''
        mode = None   # None/center/refit

        if self.idle == True: # no test is running
            if self.UITab == 1: # setting
                mode = 'center'
            elif self.UITab == 2: # Data
                mode = 'refit'
            else:
                mode = None
        else: # test is running
            if self.reading == True: # vna and/or temperature sensor is reading data
                if self.UITab == 2: # Data
                    mode  = 'refit'
                else:
                    mode  = None
            else: # is waiting for next reading
                if self.UITab == 1: # setting
                    mode = 'center'
                elif self.UITab == 2: # Data
                    mode = 'refit'
                else:
                    mode = None

        if self.ui.tabWidget_settings_settings_samprefchn.currentIndex() > 1:
            mode = 'refit'
        return mode


    def spectra_fit_get_data(self):
        '''
        get data for mpl_spectra_fit by spectraTab_mode and
        return f, G, B
        '''
        f = None
        G = None
        B = None
        logger.info('self.get_spectraTab_mode() %s', self.get_spectraTab_mode())
        if self.get_spectraTab_mode() == 'center': # for peak centering
            if not self.vna:
                return f, G, B
            # get harmonic from self.settings_harm
            harm = self.settings_harm
            chn = self.settings_chn['chn']
            logger.info(type(chn)) 
            chn_name = self.settings_chn['name']

            with self.vna: # use get_vna_data_no_with which doesn't have with statement and could keep the vna attributes
                f, G, B = self.get_vna_data_no_with(harm=harm, chn_name=chn_name)

        elif self.get_spectraTab_mode() == 'refit': # for refitting
            # get

            # get raw of active queue_id from data_saver
            f, G, B = self.get_active_raw()
            # get the vna reset flag
            freq_span = self.get_freq_span(harm=self.active['harm'], chn_name=self.active['chn_name'])
            
            logger.info('data_span: %s %s', f[0], f[-1])
            logger.info('freq_span: %s', freq_span) 
            logger.info('self.active: %s', self.active) 

            idx = np.where((freq_span[0] <= f) & (f <= freq_span[1]))
            f, G, B = f[idx], G[idx], B[idx]
        else:
            print('Change Tab to Settings or Data to active the function.')

        return f, G, B


    def get_vna_data(self, harm=None, chn_name=None):
        '''
        get data from vna use given channel(int) and harmonic (int)
        return f, G, B
        '''
        f = None
        G = None
        B = None

        if harm is None:
            harm = self.settings_harm
        if chn_name is None:
            chn_name = self.settings_chn['name']

        # get the vna reset flag
        freq_span = self.get_freq_span(harm=harm, chn_name=chn_name)
        steps = int(self.get_harmdata('lineEdit_scan_harmsteps', harm=harm, chn_name=chn_name))
        setflg = self.vna_tracker.set_check(f=freq_span, steps=steps, chn=self.get_chn_by_name(chn_name))
        logger.info(setflg) 

        logger.info(self.vna) 
        with self.vna:
            logger.info(self.vna) 
            logger.info('vna._naverage: %s', self.vna._naverage) 
            ret = self.vna.set_vna(setflg)
            if ret == 0:
                ret, f, G, B = self.vna.single_scan()
                return f, G, B
            else:
                print('There is an error while setting VNA!')
        return f, G, B


    def get_vna_data_no_with(self, harm=None, chn_name=None):
        '''
        NOTE: no with condition used. It can be used for
        continous reading data from different harms and chns.
        You need to add with condition out of it by yourself.

        get data from vna use given channel(int) and harmonic (int)
        return f, G, B
        '''
        f = None
        G = None
        B = None

        if harm is None:
            harm = self.settings_harm
        if chn_name is None:
            chn_name = self.settings_chn['name']

        # get the vna reset flag
        freq_span = self.get_freq_span(harm=harm, chn_name=chn_name)
        steps = int(self.get_harmdata('lineEdit_scan_harmsteps', harm=harm, chn_name=chn_name))
        setflg = self.vna_tracker.set_check(f=freq_span, steps=steps, chn=self.get_chn_by_name(chn_name))
        logger.info(setflg) 
        ret = self.vna.set_vna(setflg)
        logger.info('self.vna._nstep: %s', self.vna._nsteps) 
        if ret == 0:
            ret, f, G, B = self.vna.single_scan()
        else:
            print('There is an error while setting VNA!')
        return f, G, B


    def tab_spectra_fit_update_mpls(self, f, G, B):
        ''' update mpl_spectra_fit and mpl_spectra_fit_polar '''
        if f is None or G is None or B is None:
            logger.warning('None data.')
            return
        ## disconnect axes event
        self.mpl_disconnect_cid(self.ui.mpl_spectra_fit)

        self.ui.mpl_spectra_fit.update_data({'ln': 'lG', 'x': f, 'y': G})
        self.ui.mpl_spectra_fit.update_data({'ln': 'lB', 'x': f, 'y': B})

        # constrain xlim
        logger.info(f) 
        logger.info(type(f)) 
        if (f is not None) and (len(f) > 0 ) and (f[0] != f[-1]): # f is available
            self.ui.mpl_spectra_fit.ax[0].set_xlim(f[0], f[-1])
            self.ui.mpl_spectra_fit.ax[1].set_xlim(f[0], f[-1])
            self.ui.mpl_spectra_fit.ax[0].set_ylim(min(G)-0.05*(max(G)-min(G)), max(G)+0.05*(max(G)-min(G)))
            self.ui.mpl_spectra_fit.ax[1].set_ylim(min(B)-0.05*(max(B)-min(B)), max(B)+0.05*(max(B)-min(B)))
        elif f is None or (not f.any()): # vna error or f is all 0s
            self.ui.mpl_spectra_fit.ax[0].autoscale()
            self.ui.mpl_spectra_fit.ax[1].autoscale()


        ## connect axes event
        self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)

        self.ui.mpl_spectra_fit.canvas.draw()

        self.ui.mpl_spectra_fit_polar.update_data({'ln': 'l', 'x': G, 'y': B})

        # set xlabel
        # self.mpl_set_faxis(self.ui.mpl_spectra_fit.ax[0])

        # update lineedit_fit_span
        self.update_lineedit_fit_span(f)


    def on_clicked_pushButton_spectra_fit_refresh(self):
        logger.info('vna: %s', self.vna) 
        # get data
        f, G, B = self.spectra_fit_get_data()

        # update raw
        self.tab_spectra_fit_update_mpls(f, G, B)


    def on_clicked_pushButton_spectra_fit_showall(self):
        ''' show whole range of current harmonic'''
        if self.get_spectraTab_mode() == 'center': # for peak centering
            # get harmonic
            harm = self.settings_harm
            # set freq_span[harm] to the maximum range (freq_range[harm])
            self.set_freq_span(self.settings['freq_range'][harm])

        elif self.get_spectraTab_mode() == 'refit': # for peak refitting
            # get raw of active queue_id from data_saver
            f, _, _ = self.get_active_raw()
            self.set_freq_span([f[0], f[-1]])

        ## reset xlim to active on_fit_lims_change, emit scan and updating harmtree
        self.ui.mpl_spectra_fit.ax[0].set_xlim(self.get_freq_span())


    def on_fit_lims_change(self, axes):
        logger.info('on lim changed') 
        axG = self.ui.mpl_spectra_fit.ax[0]

        # logger.info('g: %s', axG.get_contains()) 
        # logger.info('r: %s', axG.contains('button_release_event')) : %s
        # logger.info('p: %s', axG.contains('button_press_event')) 

        # data lims [min, max]
        # df1, df2 = UIModules.datarange(self.ui.mpl_spectra_fit.l['lB'][0].get_xdata())
        # get axes lims
        f1, f2 = axG.get_xlim()
        # check lim with BW
        logger.info('flims: %s, %s', f1, f2)
        f1, f2 = self.span_check(harm=self.settings_harm, f1=f1, f2=f2)
        logger.info('get_navigate_mode(): %s', axG.get_navigate_mode()) 
        logger.info('flims: %s, %s', f1, f2)

        logger.info(axG.get_navigate_mode()) 
        # if axG.get_navigate_mode() == 'PAN': # pan
        #     # set a new x range: combine span of dflims and flims
        #     f1 = min([f1, df1])
        #     f2 = max([f2, df2])
        # elif axG.get_navigate_mode() == 'ZOOM': # zoom
        #     pass
        # else: # axG.get_navigate_mode() == 'None'
        #     pass
        logger.info('f12: %s, %s', f1, f2)

        # set lineEdit_scan_harmstart & lineEdit_scan_harmend
        self.ui.lineEdit_scan_harmstart.setText(str(f1)) # in Hz
        self.ui.lineEdit_scan_harmend.setText(str(f2)) # in Hz

        # update limit of active harmonic
        self.on_editingfinished_harm_freq()

        # get new data
        f, G, B = self.spectra_fit_get_data()

        # plot
        self.tab_spectra_fit_update_mpls(f, G, B)


    def update_lineedit_fit_span(self, f):
        '''
        update lineEdit_spectra_fit_span text
        input
        f: list like data in Hz
        '''
        if f is not None:
            span = max(f) - min(f)

            # update
            self.ui.lineEdit_spectra_fit_span.setText(UIModules.num2str((span / 1000), precision=5)) # in kHz
        else:
            self.ui.lineEdit_spectra_fit_span.setText('')

    # def spectra_fit_axesevent_disconnect(self, event):
    #     logger.info('disconnect') 
    #     self.mpl_disconnect_cid(self.ui.mpl_spectra_fit)

    # def spectra_fit_axesevent_connect(self, event):
    #     logger.info('connect') 
    #     self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)
    #     # since pan changes xlim before button up, change ylim a little to trigger ylim_changed
    #     ax = self.ui.mpl_spectra_fit.ax[0]
    #     logger.info('cn: %s', ax.get_navigate_mode()) 
    #     if ax.get_navigate_mode() == 'PAN':
    #         ylim = ax.get_ylim()
    #         ax.set_ylim(ylim[0], ylim[1] * 1.01)

    def mpl_disconnect_cid(self, mpl, axis='xy'):

        if 'x' in axis:
            mpl.ax[0].callbacks.disconnect(mpl.ax[0].cidx)
        if 'y' in axis:
            mpl.ax[0].callbacks.disconnect(mpl.ax[0].cidy)


    def mpl_connect_cid(self, mpl, fun, axis='xy'):
        '''

        '''
        if 'x' in axis:
            mpl.ax[0].cidx = mpl.ax[0].callbacks.connect('xlim_changed', fun)
        if 'y' in axis:
            mpl.ax[0].cidy = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('ylim_changed', fun)


    def mpl_set_faxis(self, ax):
        '''
        set freq axis tack as: [-1/2*span, 1/2*span] and
        freq axis label as: f (+cnter Hz)

        This can be done by
        ax.xaxis.set_major_locator(ticker.LinearLocator(3))
        in MatplotlibWidget.py module
        '''
        # get xlim
        xlim = ax.get_xlim()
        logger.info(xlim) 
        center = (xlim[0] + xlim[1]) / 2
        span = xlim[1] - xlim[0]

        # # get ticks
        # locs = ax.get_xticks()
        # labels = np.array(locs) - center
        # # set ticks
        # ax.set_xticklabels([str(l) for l in labels])

        # use offset
        # ax.ticklabel_format(useOffset=center, axis='x')

        # manually set
        ax.set_xticks([xlim[0], center, xlim[1]])
        #TODO following line makes the x coordinates fail
        # ax.set_xticklabels([str(-span * 0.5), '0', str(span * 0.5)])
        # set xlabel
        # ax.set_xlabel('f (+{} Hz)'.format(center))


    def mpl_sp_clr_lines_set_label(self, signal):
        '''
        clear mpl_sp<n> when the plot mode changes
        Gp, Gp+Bp, Polor
        '''

        sender = self.sender().objectName()
        logger.info(sender) 
        if (sender == 'radioButton_spectra_showGp') or (sender == 'radioButton_spectra_showBp'):
            xlabel = r'$f$ (Hz)'
            ylabel = r'$G_P$ (mS)'
            y2label = r'$B_P$ (mS)'
        elif sender == 'radioButton_spectra_showpolar':
            xlabel = r'$G_P$ (mS)'
            ylabel = r'$B_P$ (mS)'
            y2label = r''
        else:
            xlabel = r'$f$ (Hz)'
            ylabel = r'$G_P$ (mS)'
            y2label = r'$B_P$ (mS)'

        for harm in self.all_harm_list():
            harm = str(harm)
            # clear lines
            getattr(self.ui, 'mpl_sp' + harm).clr_lines()
            # clear .t['chi']
            getattr(self.ui, 'mpl_sp' + harm).update_sp_text_chi()
            # set labels
            getattr(self.ui, 'mpl_sp' + harm).ax[0].set_xlabel(xlabel)
            getattr(self.ui, 'mpl_sp' + harm).ax[0].set_ylabel(ylabel)
            getattr(self.ui, 'mpl_sp' + harm).ax[1].set_ylabel(y2label)

            getattr(self.ui, 'mpl_sp' + harm).canvas.draw()


    def mpl_sp_clr_chis(self):
        for harm in self.all_harm_list(as_str=True):
            # clear .t['chi']
            getattr(self.ui, 'mpl_sp' + harm).update_sp_text_chi()


    def on_clicked_pushButton_spectra_fit_fit(self):
        '''
        fit Gp, Bp data shown in mpl_spectra_fit ('lG' and 'lB')
        '''
        # get data in tuple (x, y)
        data_lG, data_lB = self.ui.mpl_spectra_fit.get_data(ls=['lG', 'lB'])
        logger.info('len(data_lG): %s', len(data_lG)) 
        logger.info('len(data_lG[0]): %s', len(data_lG[0])) 

        if len(data_lG[0]) == 0: # no data
            logger.info('no data from line') 
            return

        # factor = self.get_harmdata('spinBox_harmfitfactor')

        # get guessed value of cen and wid

        ## fitting peak
        logger.info('main set harm: %s', self.settings_harm) 
        self.peak_tracker.update_input(self.settings_chn['name'], self.settings_harm, harmdata=self.settings['harmdata'], freq_span=self.settings['freq_span'], fGB=[data_lG[0], data_lG[1], data_lB[1]])

        fit_result = self.peak_tracker.peak_fit(self.settings_chn['name'], self.settings_harm, components=True)
        logger.info(fit_result['v_fit']) 
        # logger.info(fit_result['comp_g']) 
        # plot fitted data
        self.ui.mpl_spectra_fit.update_data({'ln': 'lGfit', 'x': data_lG[0], 'y': fit_result['fit_g']}, {'ln': 'lBfit','x': data_lB[0], 'y': fit_result['fit_b']})
        self.ui.mpl_spectra_fit_polar.update_data({'ln': 'lfit', 'x': fit_result['fit_g'], 'y': fit_result['fit_b']})

        # clear l.['temp'][:]
        self.ui.mpl_spectra_fit.del_templines()
        self.ui.mpl_spectra_fit_polar.del_templines()
        # add devided peaks
        logger.info('to see when no peak found') 
        logger.info(fit_result['comp_g']) # to see when no peak found
        self.ui.mpl_spectra_fit.add_temp_lines(self.ui.mpl_spectra_fit.ax[0], xlist=[data_lG[0]] * len(fit_result['comp_g']), ylist=fit_result['comp_g'])
        self.ui.mpl_spectra_fit_polar.add_temp_lines(self.ui.mpl_spectra_fit_polar.ax[0], xlist=fit_result['comp_g'], ylist=fit_result['comp_b'])

        # logger.info('fit_result.comp_g', fit_result['comp_g']) 

        # update lsp
        factor_span = self.peak_tracker.get_output(key='factor_span', chn_name=self.settings_chn['name'], harm=self.settings_harm)
        gc_list = [fit_result['v_fit']['g_c']['value']] * 2 # make its len() == 2

        logger.info(factor_span) 
        logger.info(gc_list) 

        # sp_fit
        self.ui.mpl_spectra_fit.update_data({'ln': 'lsp', 'x': factor_span, 'y': gc_list})

        # sp_polar
        logger.info(len(data_lG[0])) 
        logger.info(factor_span) 
        idx = np.where((data_lG[0] >= factor_span[0]) & (data_lG[0] <= factor_span[1]))[0] # determine the indices by f (data_lG[0])

        logger.info('idx: %s', idx) 

        self.ui.mpl_spectra_fit_polar.update_data({'ln': 'lsp', 'x': fit_result['fit_g'][idx], 'y': fit_result['fit_b'][idx]})

        if self.get_spectraTab_mode() == 'center': # center mode
            # update strk
            cen_trk_freq = fit_result['v_fit']['cen_trk']['value']
            cen_trk_G = self.peak_tracker.get_output(key='gmod', chn_name=self.settings_chn['name'], harm=self.settings_harm).eval(
                self.peak_tracker.get_output(key='params', chn_name=self.settings_chn['name'], harm=self.settings_harm),
                f=cen_trk_freq
            )

            logger.info(cen_trk_freq) 
            logger.info(cen_trk_G) 

            self.ui.mpl_spectra_fit.update_data({'ln': 'strk', 'x': cen_trk_freq, 'y': cen_trk_G})

        # update srec
        cen_rec_freq = fit_result['v_fit']['cen_rec']['value']
        cen_rec_G = self.peak_tracker.get_output(key='gmod', chn_name=self.settings_chn['name'], harm=self.settings_harm).eval(
            self.peak_tracker.get_output(key='params', chn_name=self.settings_chn['name'], harm=self.settings_harm),
            f=cen_rec_freq
        )

        logger.info(cen_rec_freq) 
        logger.info(cen_rec_G) 

        self.ui.mpl_spectra_fit.update_data({'ln': 'srec', 'x': cen_rec_freq, 'y': cen_rec_G})

        #TODO add results to textBrowser_spectra_fit_result

        if self.get_spectraTab_mode() == 'refit': # refit mode
            # save scan data to data_saver
            self.data_saver.update_refit_data(
                self.active['chn_name'],
                self.get_active_queueid_from_l_harm_ind(),
                [self.active['harm']],
                fs=[fit_result['v_fit']['cen_rec']['value']], # fs
                gs=[fit_result['v_fit']['wid_rec']['value']], # gs = half_width
                ps=[fit_result['v_fit']['amp_rec']['value']], 
            )
            # update mpl_plt12
            self.update_mpl_plt12()

        # add results to textBrowser_spectra_fit_result
        self.ui.plainTextEdit_spectra_fit_result.setPlainText(self.peak_tracker.fit_result_report())


    def pick_manual_refit(self):
        '''
        manual refit process after manual refit context menu triggered
        '''

        self.disable_widgets('manual_refit_enable_disable_harmtree_list')
        # set pushButton_manual_refit checked
        self.show_widgets('manual_refit_enable_disable_list')
        self.set_manual_refit_mode(val=True)
        # self.ui.pushButton_manual_refit.setChecked(True)
        # self.init_manual_refit()

        # get data from data saver
        f, G, B = self.get_active_raw()

        # update raw
        self.tab_spectra_fit_update_mpls(f, G, B)


    def pick_export_raw(self):
        '''
        export raw data of picked point
        '''
        logger.info('export raw') 
        # make the file name
        name, ext = os.path.splitext(self.data_saver.path)
        logger.info('%s, %s', name, ext) 
        chn_txt = '_S_' if self.active['chn_name'] == 'samp' else '_R_' if self.active['chn_name'] == 'ref' else '_NA_'
        logger.info(chn_txt) 
        queue_id = self.get_active_queueid_from_l_harm_ind()
        path = name + chn_txt + str(queue_id)

        fileName = self.saveFileDialog(
            'Choose a file to save raw data',
            path=path,
            filetype=config_default['export_rawfiletype']
        )
        logger.info(fileName) 
        if fileName:
            self.data_saver.raw_exporter(fileName, self.active['chn_name'], queue_id, self.active['harm'])


    def regenerate_from_raw(self, chn_name='samp', mode='all'):
        '''
        This function is to regenerate data_saver.[chn_name] (df) all or marked data from raw of given chn_name
        '''
        msg = 'This process will delete current data in the selected channel and regenerate it from raw data.\n It may take long time whih is depends on the number of points.'

        logger.info(chn_name) 
        logger.info(mode) 

        if not self.data_saver.path: # no data
            logger.warning('No file opened.')
            return

        process = self.process_messagebox(text='Your selection was paused!', message=[msg], opts=True, forcepop=True)

        if not process:
            return

        # get all available queue_id in chn_name
        chn_queue_list = sorted(self.data_saver.get_chn_queue_list_from_raw(chn_name))

        if not chn_queue_list: # no data
            logger.warning('There is no data in selected channel.')
            return

        # reset chn & chn ref
        setattr(self.data_saver, chn_name, self.data_saver._make_df())

        # remove above queue_id from data_saver.queue_list
        self.data_saver.queue_list = sorted(list(set(self.data_saver.queue_list) - set(chn_queue_list)))

        # auto refit data by iterate all id in chn_queue_list
        for queue_id in chn_queue_list:
            # add queue_id to list
            self.data_saver.queue_list.append(queue_id)
            # get harms
            harms = self.data_saver.get_queue_id_harms_from_raw(chn_name, queue_id) # list of strings
            # add empty data 
            self.data_saver._append_new_queue([chn_name], queue_id=queue_id)
            # get index of queue_id in data
            ind = self.data_saver.get_queue_id(chn_name).index[-1] # the empty just appended should be the last one
            sel_idx_dict = {harm:[ind] for harm in harms}

            self.data_refit(chn_name, sel_idx_dict, regenerate=True)

        logger.warning('Data receating is done.')


    def get_active_queueid_from_l_harm_ind(self):
        '''
        get queue_id from data_saver by
        l_str: str. line 'l' or 'lm'
        harm: str.
        ind: index
        return queue_id
        '''
        if self.active['l_str'] == 'l': # showing all data
            mark=False
        elif self.active['l_str'] == 'lm': # showing marked data
            mark=True
            
        queue_list = self.data_saver.get_marked_harm_queue_id(self.active['chn_name'], self.active['harm'], mark=mark)

        logger.info('queue_list: %s', queue_list) 
        logger.info("self.active['ind']: %s", self.active['ind']) 
        logger.info("self.active['l_str']: %s", self.active['l_str']) 

        # return queue_list.loc[self.active['ind']]
        return queue_list.iloc[self.active['ind']]
        # return queue_list[self.active['ind']]


    def get_active_raw(self):
        '''
        get raw data of active from data_saver
        '''
        queue_id = self.get_active_queueid_from_l_harm_ind()
        f, G, B = self.data_saver.get_raw(self.active['chn_name'], queue_id, self.active['harm'])
        # logger.info('raw', f, G, B) 

        return f, G, B


    def set_manual_refit_mode(self, val=True):
        '''
        set manual refit mode off:
            pushButton_manual_refit.setChecked(val)
            than run self.init_manual_refit()
        val: True/False
        '''
        # turn off manual refit mode
        self.ui.pushButton_manual_refit.setChecked(val)
        # set other items
        self.init_manual_refit()


    def init_manual_refit(self):
        '''
        initiate widgets for manual refit
        '''
        logger.info('refit isChecked %s', self.ui.pushButton_manual_refit.isChecked()) 
        if self.ui.pushButton_manual_refit.isChecked():
            # make a copy of self.freq_span and self.harmdata for refit
            logger.info('copy to active') 
            # if use .copy() the manual refit related self.active[chn_name] needs to be changed!
            # self.settings['freq_span']['refit'] = self.settings['freq_span'][self.active['chn_name']].copy()
            # self.settings['harmdata']['refit'] = self.settings['harmdata'][self.active['chn_name']].copy()
            
            # link (by linking, we can keep the active['chn_name'] the same instead to change it to 'refit')
            self.settings['freq_span']['refit'] = self.settings['freq_span'][self.active['chn_name']]
            self.settings['harmdata']['refit'] = self.settings['harmdata'][self.active['chn_name']]

            # add manual refit tab to tabWidget_settings_settings_samprefchn
            self.add_manual_refit_tab(True)
            logger.info(self.settings_chn) 

            # self.ui.tabWidget_settings_settings_samprefchn.setCurrentIndex(-1) # show manual refit buttons and emit update_settings_chn
            # self.ui.tabWidget_settings_settings_harm.setCurrentIndex((int(self.active['harm'])-1)/2) # set to active harm and emit update_settings_chn
            self.ui.tabWidget_settings_settings_harm.setCurrentIndex(self.ui.tabWidget_settings_settings_harm.indexOf(getattr(self.ui, 'tab_settings_settings_harm'+self.active['harm']))) # set to active harm and emit update_settings_chn

            # # update treeWidget_settings_settings_harmtree
            # self.update_harmonic_tab()

            # change tabWidget_settings to settings tab
            self.ui.tabWidget_settings.setCurrentWidget(self.ui.tab_settings_settings)

        else:
            self.hide_widgets('manual_refit_enable_disable_list')
            # delete the refit tab
            self.enable_widgets('manual_refit_enable_disable_harmtree_list')
            self.add_manual_refit_tab(False)
            # reset index
            self.setvisible_samprefwidgets()

            # clear mpl
            self.ui.mpl_spectra_fit.clr_lines()
            self.ui.mpl_spectra_fit_polar.clr_lines()


    def add_manual_refit_tab(self, signal):
        '''
        add/delete manual refit tab to tabWidget_settings_settings_samprefchn
        self.add_manual_refit_tab(True)
        signal: True, add; False, delete
        '''
        if signal:
                if self.ui.tabWidget_settings_settings_samprefchn.currentIndex() != self.ui.tabWidget_settings_settings_samprefchn.indexOf(self.ui.tab_settings_settings_harmchnrefit): # refit is current tab
                    self.ui.tabWidget_settings_settings_samprefchn.addTab(self.ui.tab_settings_settings_harmchnrefit, 'Refit')
                    self.ui.tabWidget_settings_settings_samprefchn.setCurrentWidget(self.ui.tab_settings_settings_harmchnrefit)
        else:
            self.ui.tabWidget_settings_settings_samprefchn.removeTab(self.ui.tabWidget_settings_settings_samprefchn.indexOf(
                self.ui.tab_settings_settings_harmchnrefit
                )
            )


    ###### data display functions #########
    def get_axis_settings(self, name):
        '''
        get axis settings from treeWidget_settings_settings_plots
        return

        '''
        if name == 'comboBox_timeunit':
            return self.settings.get('comboBox_timeunit', 'm')
        elif name == 'comboBox_tempunit':
            return self.settings.get('comboBox_tempunit', 'C')
        elif name == 'comboBox_xscale':
            return self.settings.get('comboBox_xscale', 'linear')
        elif name == 'comboBox_yscale':
            return self.settings.get('comboBox_yscale', 'linear')
        elif name == 'checkBox_linkx':
            return self.settings.get('checkBox_linkx', True)
        else:
            return None


    def get_plt_opt(self, plt_str):
        '''
        get option for data plotting
        plt_str: 'plt1' or 'plt2'
        return itemdata splited by '_'
        '''
        return [self.settings.get('comboBox_' + plt_str + '_optsy'), self.settings.get('comboBox_' + plt_str + '_optsx')] # use the first one if failed


    def get_plt_harms(self, plt_str):
        '''
        get harmonics to plot
        plt_str: 'plt1' or 'plt2'
        return list of harmonics in strings
        '''
        return [harm for harm in self.all_harm_list(as_str=True) if self.settings.get('checkBox_' + plt_str + '_h' + harm, False)]


    def get_plt_chnname(self, plt_str):
        '''
        get channel name to plot
        plt_str: 'plt1' or 'plt2'
        return a str ('samp' or 'ref')
        '''
        if self.settings.get('radioButton_' + plt_str + '_samp'):
            return 'samp'
        elif self.settings.get('radioButton_' + plt_str + '_ref'):
            return 'ref'
        else:
            return 'samp'


    def update_mpl_plt12(self):
        '''
        update mpl_plt1 and mpl_plt2
        '''
        self.update_mpl_dataplt(plt_str='plt1')
        self.update_mpl_dataplt(plt_str='plt2')


    def update_mpl_plt1(self):
        '''
        update mpl_plt1
        '''
        self.update_mpl_dataplt(plt_str='plt1')


    def update_mpl_plt2(self):
        '''
        update mpl_plt2
        '''
        self.update_mpl_dataplt(plt_str='plt2')


    def update_mpl_dataplt(self, plt_str='none'):
        '''
        update mpl_<plt_str> by the UI settings
        plt_str: str of 'plt1' or 'plt2'
        '''

        logger.info('showall: %s', self.settings['radioButton_data_showall']) 
        logger.info('showmarked: %s', self.settings['radioButton_data_showmarked']) 

        if (plt_str != 'plt1') and (plt_str != 'plt2'): # n is not in the UI
            # do nothing
            return

        if not self.data_saver.mode: # no data
            return

        # get plt opts
        plt_opt = self.get_plt_opt(plt_str) # split str to [y, x]
        logger.info('opt: %s', plt_opt) 
        if plt_opt[0] == 'none':
            # no data need to be plotted
            return
        # get checked harmonics
        plt_harms = self.get_plt_harms(plt_str)
        logger.info('plt_harms: %s', plt_harms) 
        if not plt_harms: # no harmonic to plot
            return

        # get plot channel
        plt_chnname = self.get_plt_chnname(plt_str)

        # get timeunit
        timeuint = self.settings['comboBox_timeunit']
        # get tempunit
        tempunit = self.settings['comboBox_tempunit']
        logger.info(timeuint) 
        logger.info(tempunit) 

        # axis scale will be auto changed when comboBox_plt<n>_opts changed. We don't need to get it here

        # from tabWidget_settings
        if self.show_marked_data(): # show marked data only
            # mark = True
            if self.data_saver.with_marks(plt_chnname):
                line_group = 'lm'
                mark = True
            else:
                line_group = 'l'
                mark = False
        else: # show all data
            mark = False
            line_group = 'l'

        data_list = []
        for plt_harm in plt_harms:
            # get y data
            harm_ydata = self.get_harm_data_by_typestr(plt_opt[0], plt_chnname, plt_harm, mark=mark, unit_t=timeuint, unit_temp=tempunit)

            # get x data. normally t
            harm_xdata = self.get_harm_data_by_typestr(plt_opt[1], plt_chnname, plt_harm, mark=mark, unit_t=timeuint, unit_temp=tempunit)
            data_list.append({'ln': line_group + plt_harm, 'x': harm_xdata, 'y': harm_ydata})     

            if config_default['show_marked_when_all']:
                ## display marked data (solid) along with all data (open) (can be removed if don't like)
                if self.settings['radioButton_data_showall']:
                    if self.data_saver.with_marks(plt_chnname):
                        mark_list = self.data_saver.get_harm_marks(plt_chnname, plt_harm) == 1
                        # logger.info('mark_list', mark_list) 
                    else:
                        mark_list = []
                    if isinstance(mark_list, pd.Series):
                        data_list.append({'ln': 'lm'+plt_harm, 'x': harm_xdata[mark_list], 'y': harm_ydata[mark_list]})

        ##  this block uses full set of harms and separates harms by plt_harms
        ''' 
        # get y data
        ydata = self.get_data_by_typestr(plt_opt[0], plt_chnname, mark=mark, unit_t=timeuint, unit_temp=tempunit)

        # get x data. normally t
        xdata = self.get_data_by_typestr(plt_opt[1], plt_chnname, mark=mark, unit_t=timeuint, unit_temp=tempunit)

        logger.info('------xdata--------') 
        # logger.info(xdata) 
        logger.info('-------------------') 
        logger.info('------ydata--------') 
        # logger.info(ydata) 
        logger.info('-------------------') 

        # prepare data for plotting
        data_list = self.prepare_harm_data_for_mpl_update(plt_chnname, plt_harms, line_group, xdata, ydata, show_marked_when_all=True)
        # logger.info('data_list\n', data_list) 
        '''

        # clear .lt lines
        getattr(self.ui, 'mpl_' + plt_str).clr_lines(l_list=['lt'])
        # update mpl_<plt_str>
        getattr(self.ui, 'mpl_' + plt_str).update_data(*data_list)

        # # get keys of harms don't want to plot
        # clr_list = ['l'+harm for harm in self.all_harm_list(as_str=True) if not self.settings.get('checkBox_' + plt_str + '_h' + harm, False)]
        # # clear harmonics don't plot
        # getattr(self.ui, 'mpl_' + plt_str).clr_lines(clr_list)


    def prepare_harm_data_for_mpl_update(self, plt_chnname, plt_harms, line_group, xdata, ydata, show_marked_when_all=True):
        # MAY NOT using
        '''
        devide xdata/ydata by harmonics and return a list of tuples for data_saver.update_data
        This is for data update (also live update). So, keep it simple and fast
        the function for prop plot update can be found separately
        '''
        data_list = []

        if show_marked_when_all:
            mark_df = self.data_saver.get_list_column_to_columns_marked_rows(plt_chnname, 'marks', mark=False, dropnanmarkrow=False, deltaval=False, norm=False)
        for harm in plt_harms: # selected
            harm = str(harm)
            # set xdata for harm
            logger.info(xdata.shape) 
            if len(xdata.shape) == 1: # one column e.g.: tuple (1,) is series
                harm_xdata = xdata
            else: # multiple columns
                harm_xdata = xdata.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series

            # set ydata for harm
            if len(ydata.shape) == 1: # series
                harm_ydata = ydata
            else: # multiple columns
                harm_ydata = ydata.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series

            data_list.append({'ln': line_group+harm, 'x': harm_xdata, 'y': harm_ydata})

            if show_marked_when_all:
                ## display marked data (solid) along with all data (open) (can be removed if don't like)
                if self.settings['radioButton_data_showall']:
                    if self.data_saver.with_marks(plt_chnname):
                        mark_list = mark_df['mark'+harm] == 1
                        # logger.info('mark_list', mark_list) 
                    else:
                        mark_list = []
                    if isinstance(mark_list, pd.Series):
                        data_list.append({'ln': 'lm'+harm, 'x': harm_xdata[mark_list], 'y': harm_ydata[mark_list]})
        return data_list


    def prepare_harm_data_for_mpl_prop_update(self, plt_chnname, plt_harms, line_group, xdata, ydata, xerr=None, yerr=None, show_marked_when_all=True):
        '''
        devide xdata/ydata by harmonics and return a list of tuples for mpl.update_data
        '''
        data_list = []

        if show_marked_when_all:
            mark_df = self.data_saver.get_list_column_to_columns_marked_rows(plt_chnname, 'marks', mark=False, dropnanmarkrow=False, deltaval=False, norm=False)
        for harm in plt_harms: # selected
            harm = str(harm)
            # set xdata for harm
            logger.info(xdata.shape) 
            if len(xdata.shape) == 1: # one column e.g.: tuple (1,) is series
                harm_xdata = xdata
            else: # multiple columns
                harm_xdata = xdata.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series
            if xerr is not None: # there is err data
                if len(xdata.shape) == 1: # one column e.g.: tuple (1,) is series
                    harm_xerr = xerr
                else: # multiple columns
                    harm_xerr = xerr.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series
            # set ydata for harm
            if len(ydata.shape) == 1: # series
                harm_ydata = ydata
            else: # multiple columns
                harm_ydata = ydata.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series
            if yerr is not None: # there is err data
                if len(ydata.shape) == 1: # series
                    harm_yerr = yerr
                else: # multiple columns
                    harm_yerr = yerr.filter(regex=r'\D{}$'.format(harm), axis=1).squeeze(axis=1) # convert to series

            if (xerr is not None) or (yerr is not None): # exist error
                # change None to np.nan
                if xerr is None:
                    harm_xerr = np.nan
                if yerr is None:
                    harm_yerr = np.nan
                data_list.append({'ln': line_group+harm, 'x': harm_xdata, 'y': harm_ydata, 'xerr': harm_xerr, 'yerr': harm_yerr})
            else: # no error
                data_list.append({'ln': line_group+harm, 'x': harm_xdata, 'y': harm_ydata})


            if show_marked_when_all:
                ## display marked data (solid) along with all data (open) (can be removed if don't like)
                if self.settings['radioButton_data_showall']:
                    if self.data_saver.with_marks(plt_chnname):
                        mark_list = mark_df['mark'+harm] == 1
                        # logger.info('mark_list', mark_list) 
                    else:
                        mark_list = []
                    if isinstance(mark_list, pd.Series):
                        if (xerr is not None) or (yerr is not None): # exist error
                            data_list.append({'ln': line_group+'m'+harm, 'x': harm_xdata[mark_list], 'y': harm_ydata[mark_list],'xerr': harm_xerr[mark_list], 'yerr': harm_yerr[mark_list]})
                        else:
                            data_list.append({'ln': line_group+'m'+harm, 'x': harm_xdata[mark_list], 'y': harm_ydata[mark_list]})
        return data_list


    def get_harm_data_by_typestr(self, typestr, chn_name, harm, mark=False, unit_t=None, unit_temp=None):
        '''
        get data of a single harmonics from data_saver by given type (str) and harm (str)
        str: 'df' ('delf_exps'), 'dfn', 'mdf', 'mdfn', 'dg' ('delg_exps'), 'dgn', 'f', 'g', 'p', 'temp', 't'
        return: harm data
        '''

        logger.info('typestr: %s', typestr) 
        if typestr in ['df', 'delf_exps']: # get delf
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=True, norm=False, mark=mark)
        elif 'mdf' == typestr: # get delf
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=True, norm=False, mark=mark)
            data = self.data_saver.minus_columns(data)
        elif typestr in ['dg', 'delg_exps']: # get delg
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=True, norm=False, mark=mark)
        elif 'dfn' == typestr: # get delfn
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=True, norm=True, mark=mark)
        elif 'mdfn' == typestr: # get delfn
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=True, norm=True, mark=mark)
            data = self.data_saver.minus_columns(data)
        elif 'dgn' == typestr: # get delgn
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=True, norm=True, mark=mark)
        elif 'dD' == typestr: # get delta D
            # get delg first
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=True, norm=False, mark=mark)
            f1 = self.data_saver.get_harm_marked_f1(chn_name, harm, mark=mark)
            # convert delg to delD
            data = self.data_saver.convert_gamma_to_D(data, f1, harm)
            # convert unit
            data = data * 1e6
        elif 'dsm' == typestr: # get Sauerbrey mass
            # get delg first
            delf = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=True, norm=False, mark=mark)
            f1 = self.data_saver.get_harm_marked_f1(chn_name, harm, mark=mark)
            Zq = self.qcm.Zq
            # calculate Sauerbrey mass
            data = self.data_saver.sauerbreym(harm, delf, f1, Zq)
            # convert unit
            data = data * 1000
        elif 'f' == typestr: # get f
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'fs', deltaval=False, norm=False, mark=mark)
        elif 'g' == typestr: # get g
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=False, norm=False, mark=mark)
        elif 'p' == typestr: # get p
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'ps', deltaval=False, norm=False, mark=mark)
        elif 'gp' == typestr: # get gp
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'ps', deltaval=False, norm=False, mark=mark) * self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=False, norm=False, mark=mark)
        elif 'D' == typestr: # get delta D
            # get delg first
            data = self.data_saver.get_marked_harm_col_from_list_column(chn_name, harm, 'gs', deltaval=False, norm=False, mark=mark)
            f1 = self.data_saver.get_f1(chn_name)
            # convert delg to delD
            data = self.data_saver.convert_gamma_to_D(data, f1, harm)
            # convert unit
            data = data * 1e6
        elif 't' == typestr: # get t
            data = self.data_saver.get_marked_harm_t(chn_name, harm, mark=mark, unit=unit_t)
        elif 'temp' == typestr: # get temp
            data = self.data_saver.get_marked_harm_temp(chn_name, harm, mark=mark, unit=unit_temp)
        elif 'id' == typestr: # get queue_id
            data = self.data_saver.get_marked_harm_queue_id(chn_name, harm, mark=mark)
        elif 'idx' == typestr: # get indices
            data = self.data_saver.get_marked_harm_idx(chn_name, harm, mark=mark)

        return data


    def get_data_by_typestr(self, typestr, chn_name, mark=False, idx=[], unit_t=None, unit_temp=None):
        '''
        get data of all harmonics from data_saver by given type (str)
        str: 'df', 'dfn', 'mdf', 'mdfn', 'dg', 'dgn', 'f', 'g', 'temp', 't'
        return: data
        '''

        logger.info(typestr) 
        if typestr in ['df', 'delf_exps']: # get delf
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'fs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=False)
        elif 'mdf' == typestr: # get delf
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'fs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=False)
            data = self.data_saver.minus_columns(data)
        elif typestr in ['dg', 'delg_exps']: # get delg
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'gs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=False)
        elif 'dfn' == typestr: # get delfn
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'fs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=True)
        elif 'mdfn' == typestr: # get delfn
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'fs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=True)
            data = self.data_saver.minus_columns(data)
        elif 'dgn' == typestr: # get delgn
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'gs', mark=mark, dropnanmarkrow=False, deltaval=True, norm=True)
        elif 'f' == typestr: # get f
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'fs', mark=mark, dropnanmarkrow=False, deltaval=False, norm=False)
        elif 'g' == typestr: # get g
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'gs', mark=mark, dropnanmarkrow=False, deltaval=False, norm=False)
        elif 'p' == typestr: # get p
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'ps', mark=mark, dropnanmarkrow=False, deltaval=False, norm=False)
        elif 'gp' == typestr: # get p
            data = self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'ps', mark=mark, dropnanmarkrow=False, deltaval=False, norm=False) * self.data_saver.get_list_column_to_columns_marked_rows(chn_name, 'gs', mark=mark, dropnanmarkrow=False, deltaval=False, norm=False)
        elif 't' == typestr: # get t
            data = self.data_saver.get_t_marked_rows(chn_name, dropnanmarkrow=False, unit=unit_t)
        elif 'temp' == typestr: # get temp
            data = self.data_saver.get_temp_by_uint_marked_rows(chn_name, dropnanmarkrow=False, unit=unit_temp)
        elif 'id' == typestr: # get queue_id
            data = self.data_saver.get_queue_id_marked_rows(chn_name, dropnanmarkrow=False)
        elif 'idx' == typestr: # get indices
            data = self.data_saver.get_idx_marked_rows(chn_name, dropnanmarkrow=False)

        return data


    def update_data_axis(self, signal):

        sender_name = self.sender().objectName()
        logger.info(sender_name) 

        # check which plot to update
        if ('plt1' in sender_name) or ('plt2' in sender_name):# signal sent from one of the plots
            plt_str = sender_name.split('_')[1] # plt1 or plt2

            # plot option str in list [y, x]
            plt_opt = self.get_plt_opt(plt_str)
            logger.info(plt_opt) 

            if 't' in plt_opt: # there is time axis in the plot
                self.update_time_unit(plt_str, plt_opt)

            if 'temp' in plt_opt: # htere is temp axis in the plot
                self.update_temp_unit(plt_str, plt_opt)

            if plt_opt[0] not in ['t', 'temp']: # other type in y-axis w/o changing the unit
                ylabel = config_default['data_plt_axis_label'].get(plt_opt[0], 'label error')
                # set y labels
                getattr(self.ui, 'mpl_' + plt_str).ax[0].set_ylabel(ylabel)
                getattr(self.ui, 'mpl_' + plt_str).canvas.draw()

            if plt_opt[1] not in ['t', 'temp']: # other type in x-axis w/o changing the unit
                xlabel = config_default['data_plt_axis_label'].get(plt_opt[1], 'label error')
                # set x labels
                getattr(self.ui, 'mpl_' + plt_str).ax[0].set_xlabel(xlabel)
                getattr(self.ui, 'mpl_' + plt_str).canvas.draw()

        if sender_name == 'comboBox_timeunit': # update both axises of mpl_plt1 & mpl_plt2
            for plt_str in ['plt1', 'plt2']:
                # plot option str in list [y, x]
                plt_opt = self.get_plt_opt(plt_str)
                if 't' in plt_opt: # there is time axis in the plot (plt_opt[0] == 't' or plt_opt[1] == 't')
                    self.update_time_unit(plt_str, plt_opt)
                    # getattr(self.ui, 'mpl_' + plt_str).canvas.draw()


        if sender_name == 'comboBox_tempunit': # update both axises of mpl_plt1 & mpl_plt2
            for plt_str in ['plt1', 'plt2']:
                # plot option str in list [y, x]
                plt_opt = self.get_plt_opt(plt_str)
                if 'temp' in plt_opt: # there is temp axis in the plot
                    self.update_temp_unit(plt_str, plt_opt)
                    # getattr(self.ui, 'mpl_' + plt_str).canvas.draw()

        if sender_name == 'comboBox_xscale': # update both axises of mpl_plt1 & mpl_plt2
            for plt_str in ['plt1', 'plt2']:
                # plot option str in list [y, x]
                getattr(self.ui, 'mpl_' + plt_str).ax[0].set_xscale(self.sender().itemData(signal))
                getattr(self.ui, 'mpl_' + plt_str).canvas.draw()

        if sender_name == 'comboBox_yscale': # update both axises of mpl_plt1 & mpl_plt2
            for plt_str in ['plt1', 'plt2']:
                # plot option str in list [y, x]
                getattr(self.ui, 'mpl_' + plt_str).ax[0].set_yscale(self.sender().itemData(signal))
                getattr(self.ui, 'mpl_' + plt_str).canvas.draw()

        if sender_name == 'checkBox_linkx': # link x axis of mpl_plt1 & mpl_plt2
            if signal:
                self.ui.mpl_plt1.ax[0].get_shared_x_axes().join(
                    self.ui.mpl_plt1.ax[0],
                    self.ui.mpl_plt2.ax[0]
                )
            else:
                self.ui.mpl_plt1.ax[0].get_shared_x_axes().remove(
                    self.ui.mpl_plt2.ax[0]
                )

            self.ui.mpl_plt1.canvas.draw()
            self.ui.mpl_plt2.canvas.draw()

    # def set_plt2_on_plt1_xlim_change(self):
    #     # get mpl_plt1 xlims
    #     xlim = self.ui.mpl_plt1.ax[0].get_xlim()
    #     # set mpl_plt2 xlim
    #     self.ui.mpl_plt2.ax[0].set_xlim(xlim)

    # def set_plt1_on_plt2_xlim_change(self):
    #     # get mpl_plt2 xlims
    #     xlim = self.ui.mpl_plt2.ax[0].get_xlim()
    #     # set mpl_plt1 xlim
    #     self.ui.mpl_plt1.ax[0].set_xlim(xlim)


    def show_marked_data(self):
        '''
        check radiobuttons and return mode to display data
        '''
        if self.settings['radioButton_data_showall']: # show all data
            mark = False
        elif self.settings['radioButton_data_showmarked']: # show marked data only
            mark = True

        return mark


    def update_time_unit(self, plt_str, plt_opt):
        '''
        update time unit in mpl_<plt_str> x/y label
        plt_str: 'plt1' or 'plt2'
        plt_opt: list of plot type str [y, x]
        NOTE: check if time axis in plot_opt before sending to this function
        '''
        logger.info(plt_str) 
        logger.info(plt_opt) 
        if 't' not in plt_opt:
            return

        if 't' == plt_opt[0]: # is y axis
            ylabel = config_default['data_plt_axis_label'].get(plt_opt[0], 'label error')
            logger.info(ylabel) 
            ylabel = self.time_str_unit_replace(ylabel)
            logger.info(ylabel) 
            getattr(self.ui, 'mpl_' + plt_str).ax[0].set_ylabel(ylabel)
            logger.info(getattr(self.ui, 'mpl_' + plt_str).ax[0].get_ylabel()) 
        if 't' == plt_opt[1]: # is x axis
            xlabel = config_default['data_plt_axis_label'].get(plt_opt[1], 'label error')
            xlabel = self.time_str_unit_replace(xlabel)
            getattr(self.ui, 'mpl_' + plt_str).ax[0].set_xlabel(xlabel)
            logger.info(getattr(self.ui, 'mpl_' + plt_str).ax[0].get_xlabel()) 
            logger.info(xlabel) 
        getattr(self.ui, 'mpl_' + plt_str).canvas.draw()


    def update_temp_unit(self, plt_str, plt_opt):
        '''
        update temp unit in mpl_<plt_str> x/y label
        plt_str: 'plt1' or 'plt2'
        plt_opt: list of plot type str [y, x]
        '''
        logger.info(plt_str) 
        logger.info(plt_opt) 
        if 'temp' not in plt_opt:
            return
        # idx_temp, = [i for i in range(len(plt_opt)) if plt_opt[i] == 'temp']
        # logger.info(idx_temp) 

        if 'temp' == plt_opt[0]: # is y axis
            ylabel = config_default['data_plt_axis_label'].get(plt_opt[0], 'label error')
            ylabel = self.temp_str_unit_replace(ylabel)
            getattr(self.ui, 'mpl_' + plt_str).ax[0].set_ylabel(ylabel)
            logger.info(ylabel) 
        if 'temp' == plt_opt[1]: # is x axis
            xlabel = config_default['data_plt_axis_label'].get(plt_opt[1], 'label error')
            xlabel = self.temp_str_unit_replace(xlabel)
            getattr(self.ui, 'mpl_' + plt_str).ax[0].set_xlabel(xlabel)
            logger.info(xlabel) 
        getattr(self.ui, 'mpl_' + plt_str).canvas.draw()


    def time_str_unit_replace(self, time_str):
        '''
        replace '<unit>' in time_str and
        return time_str with uint set in UI
        '''
        timeunit = self.get_axis_settings('comboBox_timeunit')
        logger.info(timeunit) 
        if timeunit == 's':
            timeunit = r's'
        elif timeunit == 'm':
            timeunit = r'min'
        elif timeunit == 'h':
            timeunit = r'h'
        elif timeunit == 'd':
            timeunit = r'day'
        return time_str.replace('<unit>', timeunit)


    def temp_str_unit_replace(self, temp_str):
        '''
        replace '<unit>' in temp_str and
        return temp_str with uint set in UI
        '''
        tempunit = self.get_axis_settings('comboBox_tempunit')
        if tempunit == 'C':
            tempunit = r'$\degree$C'
        elif tempunit == 'K':
            tempunit = r'K'
        elif tempunit == 'F':
            tempunit = r'$\degree$F'
        logger.info(tempunit) 

        return temp_str.replace('<unit>', tempunit)


    def clr_mpl_harm(self):
        '''
        clear 'l' and 'lm' lines of harm (str) in mpl_<plt_str>
        '''
        sender = self.sender().objectName()
        logger.info(sender) 
        str_list = sender.split('_')
        logger.info(str_list) 
        logger.info(self.settings[sender]) 

        if not self.settings[sender]: # unchecked
            self.clr_mpl_l(str_list[1], line_group_list=['l', 'lm'], harm_list=[sender[-1]]) # sender[-1] is the harm from checkBox_plt<n>_h<harm>


    def set_mpl_lm_style(self, signal):
        line_list = ['lm'+harm for harm in self.all_harm_list(as_str=True)]
        if signal:
            self.ui.mpl_plt1.change_style(line_list, linestyle='-')
            self.ui.mpl_plt2.change_style(line_list, linestyle='-')
        else:
            self.ui.mpl_plt1.change_style(line_list, linestyle='none')
            self.ui.mpl_plt2.change_style(line_list, linestyle='none')


    def clr_mpl_l12(self):
        # self.clr_mpl_l('plt1')
        # self.clr_mpl_l('plt2')
        self.ui.mpl_plt1.clr_lines()
        self.ui.mpl_plt2.clr_lines()


    def clr_mpl_l(self, plt_str, line_group_list=['l'], harm_list=[]):
        '''
        clear .l['l<n>'] in mpl_<plt_str>
        '''
        if not harm_list:
            harm_list = self.all_harm_list(as_str=True)
        for line_group in line_group_list:
            # get keys of harms don't want to plot
            clr_list = [line_group+harm for harm in harm_list]
            # clear harmonics don't plot
            getattr(self.ui, 'mpl_' + plt_str).clr_lines(clr_list)


    def mpl_data_open_custom_menu(self, position, mpl, plt_str):
        '''
        check which menu to open: mpl_data_open_selector_menu or mpl_data_pen_picker_menu
        '''
        logger.info('customMenu') 
        logger.info(position) 
        logger.info(mpl) 
        logger.info(plt_str) 

        if not self.data_saver.path:
            return

        if mpl.sel_mode == 'selector':
            self.mpl_data_open_selector_menu(position, mpl, plt_str)
        elif mpl.sel_mode == 'picker':
            self.mpl_data_open_picker_menu(position, mpl, plt_str)
        # else:
        #     self.mpl_data_open_selector_menu(position, mpl, plt_str)

        # # update display
        # mpl.canvas.draw()
        logger.info('this should run after contextmenu') 
        self.update_mpl_plt12()


    def mpl_data_open_selector_menu(self, position, mpl, plt_str):
        '''
        function to execute the selector custom context menu for selector
        '''
        logger.info('selector') 
        logger.info(position) 
        logger.info(mpl) 
        logger.info(plt_str) 

        # get .l['ls<n>'] data
        # dict for storing the selected indices
        sel_idx_dict = {}
        selflg = False # flag for if sel_data_dict is empty
        plt_harms = self.get_plt_harms(plt_str) # get checked harmonics
        for harm in plt_harms:
            harm = str(harm)
            logger.info('harm: %s', harm) 
            # print(mpl.get_data(ls=['ls'+harm]))
            harm_sel_data, = mpl.get_data(ls=['ls'+harm]) # (xdata, ydata)
            logger.info(harm_sel_data) 
            logger.info(harm_sel_data[0]) 
            if isinstance(harm_sel_data[0], pd.Series) and harm_sel_data[0].shape[0] > 0: # data is not empty
                harm_sel_idx = list(harm_sel_data[0].index) # get indices from xdata
                logger.info(harm_sel_idx) 
                sel_idx_dict[harm] = harm_sel_idx
                selflg = True
        logger.info(sel_idx_dict) 
        # if no selected data return
        if not selflg:
            # pass
            return

        logger.info('selflg: %s', selflg) 

        # get channel name
        chn_name = self.get_plt_chnname(plt_str)
        marks = self.data_saver.get_marks(chn_name, tocolumns=True) # df of boolean shows if has data

        # create contextMenu
        selmenu = QMenu('selmenu', self)

        menuMark = QMenu('Mark', self)
        actionMark_all = QAction('Mark all showing data', self)
        actionMark_all.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'all', marks), 1))
        if selflg:
            actionMark_selpts = QAction('Mark selected points', self)
            actionMark_selpts.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selpts', marks), 1))
            actionMark_selidx = QAction('Mark selected indices', self)
            actionMark_selidx.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selidx', marks), 1))
            actionMark_selharm = QAction('Mark selected harmonics', self)
            actionMark_selharm.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selharm', marks), 1))

        menuMark.addAction(actionMark_all)
        if selflg:
            menuMark.addAction(actionMark_selpts)
            menuMark.addAction(actionMark_selidx)
            menuMark.addAction(actionMark_selharm)

        menuUnmark = QMenu('Unmark', self)
        actionUnmark_all = QAction('Unmark all showing data', self)
        actionUnmark_all.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'all', marks), 0))
        if selflg:
            actionUnmark_selpts = QAction('Unmark selected points', self)
            actionUnmark_selpts.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selpts', marks), 0))
            actionUnmark_selidx = QAction('Unmark selected indices', self)
            actionUnmark_selidx.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selidx', marks), 0))
            actionUnmark_selharm = QAction('Unmark selected harmonics', self)
            actionUnmark_selharm.triggered.connect(lambda: self.data_saver.selector_mark_sel(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selharm', marks), 0))

        menuUnmark.addAction(actionUnmark_all)
        if selflg:
            menuUnmark.addAction(actionUnmark_selpts)
            menuUnmark.addAction(actionUnmark_selidx)
            menuUnmark.addAction(actionUnmark_selharm)

        menuDel = QMenu('Delete', self)
        actionDel_all = QAction('Delete all showing data', self)
        actionDel_all.triggered.connect(lambda: self.on_triggered_del_menu(chn_name, plt_harms, sel_idx_dict, 'all', marks))
        if selflg:
            actionDel_selpts = QAction('Delete selected points', self)
            actionDel_selpts.triggered.connect(lambda: self.on_triggered_del_menu(chn_name, plt_harms, sel_idx_dict, 'selpts', marks))
            actionDel_selidx = QAction('Delete selected indices', self)
            actionDel_selidx.triggered.connect(lambda: self.on_triggered_del_menu(chn_name, plt_harms, sel_idx_dict, 'selidx', marks))
            actionDel_selharm = QAction('Delete selected harmonics', self)
            actionDel_selharm.triggered.connect(lambda: self.on_triggered_del_menu(chn_name, plt_harms, sel_idx_dict, 'selharm', marks))

        menuDel.addAction(actionDel_all)
        if selflg:
            menuDel.addAction(actionDel_selpts)
            menuDel.addAction(actionDel_selidx)
            menuDel.addAction(actionDel_selharm)

        menuRefit = QMenu('Refit', self)
        actionRefit_all = QAction('Refit all showing data', self)
        actionRefit_all.triggered.connect(lambda: self.data_refit(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'all', marks)))
        if selflg:
            actionRefit_selpts = QAction('Refit selected points', self)
            actionRefit_selpts.triggered.connect(lambda: self.data_refit(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selpts', marks)))
            actionRefit_selidx = QAction('Refit selected indices', self)
            actionRefit_selidx.triggered.connect(lambda: self.data_refit(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selidx', marks)))
            actionRefit_selharm = QAction('Refit selected harmonics', self)
            actionRefit_selharm.triggered.connect(lambda: self.data_refit(chn_name, UIModules.sel_ind_dict(plt_harms, sel_idx_dict, 'selharm', marks)))

        menuRefit.addAction(actionRefit_all)
        if selflg:
            menuRefit.addAction(actionRefit_selpts)
            menuRefit.addAction(actionRefit_selidx)
            menuRefit.addAction(actionRefit_selharm)

        selmenu.addMenu(menuMark)
        selmenu.addMenu(menuUnmark)
        selmenu.addMenu(menuDel)
        selmenu.addMenu(menuRefit)

        #else, find out the indices and do mark/unmark/delete
        selmenu.exec_(mpl.canvas.mapToGlobal(position))


    def mpl_data_open_picker_menu(self, position, mpl, plt_str):
        '''
        function to execute the picker custom context menu for selector
        '''
        logger.info('picker customMenu') 
        logger.info(position) 
        logger.info(mpl) 
        logger.info(plt_str) 

        # get .l['lp'] data
        pk_data, = mpl.get_data(ls=['lp']) # (xdata, ydata)
        logger.info(pk_data) 
        logger.info(pk_data[0]) 
        logger.info(type(pk_data)) 
        logger.info(type(pk_data[0])) 

        if isinstance(pk_data[0], (float, int, np.int64)): # data is not empty (float for values, int for index and queue_id)
            label = mpl.l['lp'][0].get_label()
            line, ind = label.split('_')
            l, harm = line[:-1], line[-1]
            logger.info('label: %s', label) 
            logger.info(line) 
            logger.info(l) 
            logger.info(harm) 
            logger.info(ind) 

            self.active['chn_name'] = self.get_plt_chnname(plt_str)
            self.active['harm'] = harm
            self.active['plt_str'] = plt_str
            self.active['l_str'] = l
            self.active['ind'] = int(ind)

            logger.info(self.active) 

            # get channel name
            chn_name = self.get_plt_chnname(plt_str)

            # create contextMenu
            pkmenu = QMenu('pkmenu', self)

            actionManual_fit = QAction('Manual fit', self)
            actionManual_fit.triggered.connect(self.pick_manual_refit)

            actionExport_raw = QAction('Export raw data', self)
            actionExport_raw.triggered.connect(self.pick_export_raw)

            pkmenu.addAction(actionManual_fit)
            pkmenu.addAction(actionExport_raw)

            pkmenu.exec_(mpl.canvas.mapToGlobal(position))

        else:
            # nothing to do
            pass


    def marknpts(self):
        '''
        mark n points by the settings
        There are two ways:
        NOTE: This function doesn't clear previous marks
        '''

        if not self.data_saver.path:
            logger.warning('No data available!')
            return

        item_marknptschn = self.ui.comboBox_settings_data_marknptschn
        chn_name = item_marknptschn.itemData(item_marknptschn.currentIndex())
        # get settings (the widgets don't saved in self.settings)
        npt = self.ui.spinBox_settings_data_marknpts.value()
        marklinear = self.ui.radioButton_settings_data_marklinear.isChecked()
        marklog = self.ui.radioButton_settings_data_marklog.isChecked()

        if npt == 0: # no point to mark
            return
        
        # get channel indices
        chn_idx = self.data_saver.get_idx(chn_name).values # use list of index for comperison (ndarray)
        # get channel t
        chn_t = self.data_saver.get_t_s(chn_name).values # time in second (ndarray)
        
        if marklinear:
            mark_t = np.linspace(chn_t[0], chn_t[-1], num=npt)
        elif marklog:
            mark_t = np.logspace(np.log10(chn_t[0]), np.log10(chn_t[-1]), num=npt)
        else:
            return

        # find indices of t close to mark_t
        mark_idx = np.zeros(mark_t.shape, dtype='int')
        for i, t in enumerate(mark_t):
            mark_idx[i] = np.argmin(np.abs(chn_t - t))
        
        # remove duplicate
        mark_idx = np.unique(mark_idx)
        
        # apply mark_idx to chn_idx in case chn_idx not continous
        mark_idx = list(chn_idx[mark_idx]) # ndarray to list

        # make the sel_idx_dict
        sel_idx_dict = {}
        for harm in self.all_harm_list(as_str=True):
            sel_idx_dict[harm] = mark_idx
        
        self.data_saver.selector_mark_sel(chn_name, sel_idx_dict, 1)

        # plot data
        self.update_mpl_plt12()
        
        










        



    #### mech related funcs ###
    def film_construction_mode_switch(self):
        '''
        switch the mode
        '''
        logger.info('triggrued') 
        currInd = self.ui.stackedWidget_settings_mechanics_modeswitch.currentIndex()
        count = self.ui.stackedWidget_settings_mechanics_modeswitch.count()
        # increase currentIndex by (currInd+1)%count
        self.ui.stackedWidget_settings_mechanics_modeswitch.setCurrentIndex((currInd + 1) % count)


    def build_mech_layers(self):
        '''
        insert/delete a gridlayout with items for layers between bulk and electrode
        nlayers is from a spinBox which limits its value e.g. [0, 5]

        ------------
        radio button (nth layer) | comobx (source) | lineEdit (value)
        ...
        radio button (electrod)  | comobx (source) | lineEdit (value)
        '''
        start_row = 1 # the first row to insert

        # check previous number of layer by check radioButton_mech_expertmode_calc_0 row number
        # number of rows
        rowcount = self.ui.gridLayout_mech_expertmode_layers.rowCount()
        logger.info('rowcount: %s', rowcount) 
        logger.info( self.get_mechchndata('radioButton_mech_expertmode_calc_0')) 
        logger.info(self.settings['mechchndata']) 
        if (self.get_mechchndata('radioButton_mech_expertmode_calc_0', mech_chn='samp') is not None) or (self.get_mechchndata('radioButton_mech_expertmode_calc_0', mech_chn='ref') is not None):
            bottom_row = self.ui.gridLayout_mech_expertmode_layers.getItemPosition(self.ui.gridLayout_mech_expertmode_layers.indexOf(self.ui.radioButton_mech_expertmode_calc_0))[0]
            rowcount = bottom_row + 1
            # since the rowcount will not decrease

        pre_nlayers = rowcount - start_row
        logger.info('pre_nlayers: %s', pre_nlayers) 
        nlayers = self.get_mechchndata('spinBox_mech_expertmode_layernum') # get changed number of layers after update in self.settings
        nlayers += 1 # add 0 layer
        logger.info('nlayers: %s', nlayers) 

        del_nlayers = nlayers - pre_nlayers
        logger.info('del_nlayers: %s', del_nlayers) 
        if pre_nlayers == nlayers:
            # no changes
            return
        elif pre_nlayers < nlayers: # add new layers
            # add layers above previous layer
            logger.info('add') 
            for i in range(pre_nlayers, nlayers):
                logger.info(i) 

                ## create wedgits
                # radiobutton radioButton_mech_expertmode_calc_
                setattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i), QRadioButton(self.ui.stackedWidgetPage_mech_expertmode))
                getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).setObjectName("radioButton_mech_expertmode_calc_"+str(i))
                self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)), nlayers-i, 0, 1, 1)
                if i == 0:
                    getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).setText(QCoreApplication.translate('MainWindow', 'Electrode'))
                    getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).setCheckable(False)

                else:
                    getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).setText(QCoreApplication.translate('MainWindow', 'layer '+str(i)))

                # combobox comboBox_mech_expertmode_source_
                setattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i), QComboBox(self.ui.stackedWidgetPage_mech_expertmode))
                sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).sizePolicy().hasHeightForWidth())
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).setSizePolicy(sizePolicy)
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).setCurrentText("")
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).setObjectName("comboBox_mech_expertmode_source_"+str(i))
                self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)), nlayers-i, 1, 1, 1)

                # combobox comboBox_mech_expertmode_indchn_
                setattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i),  QComboBox(self.ui.stackedWidgetPage_mech_expertmode))
                getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)).setObjectName('comboBox_mech_expertmode_indchn_'+str(i))
                self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)), nlayers-i, 2, 1, 1)

                # lineEdit lineEdit_mech_expertmode_value_
                setattr(self.ui, 'lineEdit_mech_expertmode_value_'+str(i),QLineEdit(self.ui.stackedWidgetPage_mech_expertmode))
                # getattr(self.ui, 'lineEdit_mech_expertmode_value_'+str(i)).setReadOnly(True)
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+str(i)).setObjectName("lineEdit_mech_expertmode_value_"+str(i))
                #change the background
                # getattr(self.ui, 'lineEdit_mech_expertmode_value_' + str(i)).setStyleSheet(
                    # "QLineEdit { background: transparent; }"
                # )

                ## set signals. reverse the widgets secquence
                # linedeit: signal
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+str(i)).textChanged.connect(self.update_mechchnwidget)
                
                # combobox: add signal/slot
                getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)).currentIndexChanged.connect(self.on_mech_layer_indchn_changed)
                getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)).currentIndexChanged.connect(self.update_mechchnwidget)

                # combobox: add signal/slot
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).currentIndexChanged.connect(self.on_mech_layer_source_changed)
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).currentIndexChanged.connect(self.update_mechchnwidget)

                # radiobutton: signal
                getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).toggled.connect(self.on_mech_layer_calc_changed)
                getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).toggled.connect(self.update_mechchnwidget)

                ## initiate values. reverse the widgets secquence
                # combobox: build with opts
                self.build_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)), 'ref_channel_opts')
                # combobox: build with opts
                self.build_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)), 'qcm_layer_source_opts')

        elif pre_nlayers > nlayers: # delete extra layers
            # delete layers from row 1 (leave bulk (0))
            logger.info('delete') 
            for i in range(nlayers, pre_nlayers):
                logger.info(i) 
                # radiobutton
                getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)).deleteLater()
                # combobox
                getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)).deleteLater()
                # combobox
                getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+str(i)).deleteLater()
                # lineedit
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+str(i)).deleteLater()
                # deleting the whole row will move previous layers to (row - 1)

                # pop the data from self.settings
                self.pop_mechchndata('radioButton_mech_expertmode_calc_'+str(i))
                self.pop_mechchndata('comboBox_mech_expertmode_source_'+str(i))
                self.pop_mechchndata('comboBox_mech_expertmode_indchn_'+str(i))
                self.pop_mechchndata('lineEdit_mech_expertmode_value_'+str(i))
        else:
            pass
        # move previous layers to (current row + del_nlayers)
        for i in range(nlayers):
            # bottom_row-i+del_nlayers
            # bottom_row-i: current row
            self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'radioButton_mech_expertmode_calc_'+str(i)), (rowcount-1-i+del_nlayers), 0, 1, 1)
            self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'comboBox_mech_expertmode_source_'+str(i)), (rowcount-1-i+del_nlayers), 1, 1, 1)
            self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'comboBox_mech_expertmode_indchn_' + str(i)), (rowcount-1-i+del_nlayers), 2, 1, 1)
            self.ui.gridLayout_mech_expertmode_layers.addWidget(getattr(self.ui, 'lineEdit_mech_expertmode_value_' + str(i)), (rowcount-1-i+del_nlayers), 3, 1, 1)

        # load data 
        self.load_film_layers_widgets()

    
        logger.info('rowcount %s', self.ui.gridLayout_mech_expertmode_layers.rowCount()) 


    def on_mech_chn_changed(self):
        '''
        update self.mech_chn
        set film lsyers widgets
        '''
        idx = self.ui.tabWidget_mechanics_chn.currentIndex()
        logger.info(idx) 
        if idx == 0: # samp
            self.mech_chn = 'samp'
        elif idx == 1: # ref
            self.mech_chn = 'ref'
        
        self.set_film_layers_widgets()

        # set comboBox_settings_mechanics_model_samplayer_chn
        self.load_comboBox(self.ui.comboBox_settings_mechanics_model_samplayer_chn, val=self.mech_chn)


    def update_mechchnwidget(self, signal):
        '''
        update widgets in stackedWidgetPage_mech_expertmode
        which will change with self.mech_chn
        '''
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        logger.info('mech_chn widget update: %s %s', self.sender().objectName(), signal)
        # mech_chn = self.mech_chn

        if isinstance(self.sender(), QLineEdit):
                self.set_mechchndata(self.sender().objectName(), signal)
                # try:
                #     self.set_mechchndata(self.sender().objectName(), float(signal))
                # except:
                #     self.set_mechchndata(self.sender().objectName(), 0)
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.set_mechchndata(self.sender().objectName(), signal)
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.set_mechchndata(self.sender().objectName(), signal)
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.set_mechchndata(self.sender().objectName(), value)
        # if the sender of the signal isA QSpinBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), (QSpinBox, QDoubleSpinBox)):
            self.set_mechchndata(self.sender().objectName(), signal)


    def get_mechchndata(self, objname, mech_chn=None):
        '''
        get data with given objname in
        stackedWidgetPage_mech_expertmode
        '''
        if mech_chn is None: # use harmonic displayed in UI
            mech_chn = self.mech_chn

        return self.settings['mechchndata'][mech_chn].get(objname, None)


    def set_mechchndata(self, objname, val, mech_chn=None):
        '''
        set data with given objname in
        stackedWidgetPage_mech_expertmode
        '''
        if mech_chn is None: # use harmonic displayed in UI
            mech_chn = self.mech_chn

        self.settings['mechchndata'][mech_chn][objname] = val


    def pop_mechchndata(self, objname, mech_chn=None):
        '''
        set data with given objname in
        stackedWidgetPage_mech_expertmode
        '''
        if mech_chn is None: # use harmonic displayed in UI
            mech_chn = self.mech_chn
        logger.info(self.settings['mechchndata']) 
        if self.get_mechchndata(objname, mech_chn=mech_chn) is not None: # exists
            self.settings['mechchndata'][mech_chn].pop(objname)


    def on_mech_layer_calc_changed(self, signal):
        '''
        change the slected layer comboBox_mech_expertmode_indchn_<n> by self.mech_chn
        '''
        sender_name = self.sender().objectName()
        layer_num = sender_name.split('_')[-1] # str

        if signal: # is checked
            # set mechchandata
            self.set_mechchndata('comboBox_mech_expertmode_indchn_' + layer_num, val=self.mech_chn)
            # reload the comboBox
            self.load_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_indchn_' + layer_num), mech_chn=self.mech_chn)


    def on_mech_layer_indchn_changed(self, signal):
        '''
        change the slected layer comboBox_mech_expertmode_indchn_<n> to self.mech_chn if radioButton_mech_expertmode_calc_<n> is checked
        '''
        sender_name = self.sender().objectName()
        layer_num = sender_name.split('_')[-1] # str

        iscalc = self.get_mechchndata('radioButton_mech_expertmode_calc_' + layer_num)

        indchn = self.sender().itemData(signal)

        if iscalc and (indchn != self.mech_chn): # is checked
            # set mechchandata
            self.set_mechchndata('comboBox_mech_expertmode_indchn_' + layer_num, val=self.mech_chn)
            # reload the comboBox
            self.load_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_indchn_' + layer_num), mech_chn=self.mech_chn)


    def on_mech_layer_source_changed(self, signal):
        '''
        hide/show comboBox_mech_expertmode_indchn_<n> by comboBox_mech_expertmode_source_<n> value
        '''
        sender_name = self.sender().objectName()

        sender_val = self.sender().itemData(signal) # str

        layer_num = sender_name.split('_')[-1] # str

        if sender_val == 'ind': # use index 
            # show comboBox_mech_expertmode_indchn_<n>
            getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+layer_num).setVisible(True)

            # set comboBox_mech_expertmode_indchn_<n> format
            if layer_num == '0': # electrode layer and there is data. This the the bare value in air use the same in data chn_name ref
                #TODO This layer should be set at the same time data reference chn is set
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+layer_num).setText('[]')
            else: # film layers
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+layer_num).setText('[]')
        else: # use other form
            # hide comboBox_mech_expertmode_indchn_<n>
            getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+layer_num).setVisible(False)

            # set lineEdit_mech_expertmode_value_<n>
            if sender_val == 'prop': # use property
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+layer_num).setText("{0}''grho': {grho}, 'phi': {phi}, drho': {drho}, 'n': {n}{1}".format('{', '}', **QCM.prop_default['air']))
            elif sender_val == 'name': # use name
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+layer_num).setText("air")
            elif sender_val == 'fg': # use freq and gamma value
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+layer_num).setText("{'delf': [], 'delg': []}")


    def make_film_layers_dict(self):
        '''
        return a dict with film layers construction 
        from currenent widgets' value or mechchndata (now we are using widgets' value)
        '''
        prefix = {
            'calc': 'radioButton_mech_expertmode_calc_', 
            'source': 'comboBox_mech_expertmode_source_', 
            'indchn': 'comboBox_mech_expertmode_indchn_',
            'val': 'lineEdit_mech_expertmode_value_',
        }
        # get number of layers 
        n_layers = int(self.get_mechchndata('spinBox_mech_expertmode_layernum'))
        n_layers += 1 # add electrode layer

        logger.info(n_layers) 

        film_dict = {}

        for n in range(n_layers): # all layers
            logger.info(n) 
            n = str(n) # convert to string for storing as json, which does not support int key
            film_dict[n] = {key: self.get_mechchndata(pre_name+n) for key, pre_name in prefix.items()}

        logger.info(film_dict) 

        return film_dict


    def set_film_layers_widgets(self):
        '''
        set widgets related film_layers construction from self.settings['mechchndata']
        '''
        logger.info('mech_ch %s', self.mech_chn)  
        logger.info("self.settings['mechchndata']") 
        logger.info(self.settings['mechchndata']) 
        # get number of layers 
        layernum = self.get_mechchndata('spinBox_mech_expertmode_layernum')
        logger.info('layernum: %s', layernum) 

        if layernum is None: # no data saved
            logger.info('layernum is None') 
            # use default layernum
            layernum = int(self.settings['spinBox_mech_expertmode_layernum'])
            # generate layers
            self.ui.spinBox_mech_expertmode_layernum.setValue(layernum) # use default value
            self.set_mechchndata('spinBox_mech_expertmode_layernum', layernum)
            logger.info(self.settings['mechchndata']) 
        else:
            # generate layers
            self.ui.spinBox_mech_expertmode_layernum.setValue(int(layernum))

        self.load_film_layers_widgets() # load values
        

    def load_film_layers_widgets(self):
        '''
        load values in mechchndata to widgets
        if None
        save to mechndata
        '''
        layernum = int(self.get_mechchndata('spinBox_mech_expertmode_layernum'))
        mech_chn = self.mech_chn
        for n in range(layernum+1): # all layers
            logger.info(n) 
            n = str(n) # convert to string for storing as json, which does not support int key

            if (self.get_mechchndata('radioButton_mech_expertmode_calc_'+n) is None): # no data saved for this layer or new layer
                # save current value to self.settings
                self.set_mechchndata('lineEdit_mech_expertmode_value_'+n, getattr(self.ui, 'lineEdit_mech_expertmode_value_'+n).text())
                # save current value to self.settings
                self.set_mechchndata('comboBox_mech_expertmode_indchn_'+n, getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+n).itemData(getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+n).currentIndex()))

                # save current value to self.settings
                self.set_mechchndata('comboBox_mech_expertmode_source_'+n, getattr(self.ui, 'comboBox_mech_expertmode_source_'+n).itemData(getattr(self.ui, 'comboBox_mech_expertmode_source_'+n).currentIndex()))

                # save current value to self.settings
                self.set_mechchndata('radioButton_mech_expertmode_calc_'+n, getattr(self.ui, 'radioButton_mech_expertmode_calc_'+n).isChecked())
            else: # data in mechchndata
                # the order of setting values should not change
                getattr(self.ui, 'radioButton_mech_expertmode_calc_'+n).setChecked(self.get_mechchndata('radioButton_mech_expertmode_calc_'+n))
                self.load_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_source_'+n), val=self.get_mechchndata('comboBox_mech_expertmode_source_'+n), mech_chn=mech_chn)
                self.load_comboBox(getattr(self.ui, 'comboBox_mech_expertmode_indchn_'+n), val=self.get_mechchndata('comboBox_mech_expertmode_indchn_'+n), mech_chn=mech_chn)
                getattr(self.ui, 'lineEdit_mech_expertmode_value_'+n).setText(str(self.get_mechchndata('lineEdit_mech_expertmode_value_'+n)))


    def mech_solve_chn(self, chn_name=None, chn_queue_ids=None, chn_idx=None):
        '''
        send the data to qcm module to solve in secquence by queue_ids and
        save the returned mechanic data to data_saver
        chn_queue_ids: all available queue_ids for solving. it can be all or marked queue_ids
        chn_idx: all available indices for solving. it can be all or marked indices
        '''
        t0 = time.time()

        if not self.data_saver.path:
            print('No data available!')
            return

        # set f1 to qcm module (leave this to qcm modue for each single solution)

        self.qcm.refh = self.settings['spinBox_settings_mechanics_nhcalc_n3'] # use the dissipatione harmonic as reference
        refh = self.qcm.refh # reference harmonic

        # set film reference
        # logger.info('radioButton_settings_mech_refto_air: %s', self.settings['radioButton_settings_mech_refto_air'])
        # logger.info('radioButton_settings_mech_refto_overlayer: %s', self.settings['radioButton_settings_mech_refto_overlayer'])
        if self.settings['radioButton_settings_mech_refto_air'] == 1: 
            self.qcm.refto = 0
            logger.info('set qcm.refto to 0')
        elif self.settings['radioButton_settings_mech_refto_overlayer'] == 1:
            self.qcm.refto = 1
            logger.info('set qcm.refto to 1')

        # logger.info('refh', refh) 

        # get nhcalc
        nhcalc = self.gen_nhcalc_str()

        nhcalc_list = self.gen_nhcalc_list()

        calctype = self.settings['comboBox_settings_mechanics_calctype']

        layernum = self.get_mechchndata('spinBox_mech_expertmode_layernum', mech_chn=chn_name)

        bulklimit = self.settings['doubleSpinBox_settings_mechanics_bulklimit'] # the rd limit for bulk calculation

        # logger.info('nhcalc_list', nhcalc_list) 

        # check which is using: model or layers
        if self.ui.stackedWidget_settings_mechanics_modeswitch.currentIndex() == 0: # model mode
            # convert model to mechchndata
            film_dict = self.make_film_dict_by_mechmodel_widgets()
        else: # layers mode
            ## work with layers first. get film_dict from mechchndata
            film_dict = self.make_film_layers_dict()

        # {'0': {'calc': False, 'source': 'ind', 'indchn': 'ref', 'val': '[2]'}, '1': {'calc': True, 'source': 'ind', 'indchn': 'samp', 'val': '[]'}}

        # 1. find the layer w/ calc = True
        calc_num = None
        for n, dic in film_dict.items(): # n is str
            if dic['calc']:
                calc_num = int(n)
                break
            # dic is kept for following lines
        
        if calc_num is None:
            print('Stoped! No layer is marked to calculate')
            return

        # 2. check chn_name
        if chn_name is None:
            chn_name = self.mech_chn

        # NOTE: by default dic['indchn'] == chn_name
        if dic['indchn'] != chn_name: 
            print('sample was set to different channel to mech_chn!')
            chn_name = dic['indchn']
            print('set chn_name to indchn!')

        if chn_queue_ids is None: # if no queue_id given, use all in the channel
            chn_queue_ids = self.data_saver.get_queue_id_marked_rows(chn_name, dropnanmarkrow=False)

        if chn_idx is None: # if no idx given, use all in the channel
            chn_idx = list(chn_queue_ids.index) # all available indics

        # initialize idx for solving
        idx = chn_idx.copy()      
        idx_joined = chn_idx.copy()
        queue_ids = chn_queue_ids

        # 2. get qcm data (columns=['queue_id', 't', 'temp', 'marks', 'fstars', 'fs', 'gs', 'delfstars', 'delfs', 'delgs', 'f0stars', 'f0s', 'g0s'])
        # 'delf', 'delgs' may not necessary
        qcm_df = self.data_saver.df_qcm(chn_name) 

        qcm_df_calc = qcm_df.loc[idx] # df of calc layer

        # 3. layer calc's source and index
        if dic['source'] == 'ind': # dic is the layer to calc (is where the above loop break)
            # use ind to get queue_id
            calc_idx_str = dic['val']
            calc_idx = UIModules.index_from_str(calc_idx_str, chn_idx, join_segs=False) # overwrite idx with given index
            calc_idx_joined = UIModules.index_from_str(calc_idx_str, chn_idx, join_segs=True) # overwrite idx with given index
            if calc_idx_joined:
                idx = calc_idx
                idx_joined = calc_idx_joined
                queue_ids = chn_queue_ids[idx_joined] # overwrite queue_id with queue_id calculated with given idx
                qcm_df_calc = qcm_df.loc[idx_joined] # df of calc layer
            else: # idx_joined = []
                # logger.info('idx_joined is empty') 
                pass
            # logger.info('idx', idx) 
            # logger.info('idx_joined', idx_joined) 
        elif dic['source'] == 'prop_guess':
            # set given prop 'prop_guess'
            film_dict[n]['prop_guess'] = (dic['val'])
        elif dic['source'] == 'prop':
            # set given prop 'prop'
            film_dict[n]['prop'] = (dic['val'])
        # elif dic['source'] == 'fg':
        #     # use f/g to calc prop_guess
        #     #TODO
        #     pass
        elif dic['source'] == 'name':
            # get prop_guess from qcm
            film_dict[n]['prop_guess'] = self.qcm.get_prop_by_name(dic['val'])
        else: 
            print('source not defined!')
            return

        print('Calculating {} with {} cores ...'.format(nhcalc, self.get_mp_cores()))

        # 4. iterate all layers to get props
        # prop_dict = {}
        prop_dict = {ind: {int(n): {'calc': film_dict[n]['calc']} for n in film_dict.keys()} for ind in idx_joined} # initiate dict for storing the prop 
        # to iterate the dict sorted w/o using orderedDict
        for n in sorted([int(n) for n in film_dict.keys()]):
            dic = film_dict[str(n)]
            if dic['calc']: # layer to calc: we only update prop used for guess or limit prop calculation
                if dic['source'] in ['prop', 'prop_guess']:
                    # TODO how to define prop and prop_guess.e.g.: if grho, phi, drho are all given, it is prop_guess. otherwise, is prop (limitation)
                    # set given prop 'prop_guess'
                    for ind in idx_joined:
                         prop_dict[ind][n].update(**eval(dic['val']))
                        #  prop_dict[ind][n]['calc'] = dic['calc']

                elif dic['source'] == 'name':
                    # get prop_guess from qcm
                    # since all give, it should be considered as prop_guess
                    for ind in idx_joined:
                        prop_dict[ind][n].update(**self.qcm.get_prop_by_name(dic['val']))
                        #  prop_dict[ind][n]['calc'] = dic['calc']
                # if dic['source'] == 'fg': # giving f/g 
                #     # this can be used for simulater
                #     fg = dic['val'] # {'f': [], 'g': []}
                #     fstar = {int(i*2+1): delf + 1j*delg for i, delf, delg in enumerate(zip(fg['delf'], fg['delg']))}
                #     prop_dict[ind][n]['fstar'] = fstar
            else: # known layer
                ## get prop for this layer
                if dic['source'] == 'prop':
                    # set given prop, it should not be 'prop_guess'
                    for ind in idx_joined:
                        prop_dict[ind][n].update(**eval(dic['val']))
                        #  prop_dict[ind][n]['calc'] = dic['calc']
                # elif dic['source'] == 'fg':
                #     # use f/g to calc prop_guess
                #     #TODO
                #     pass

                elif dic['source'] == 'name':
                    # get prop_guess from qcm
                    # logger.info('dic', dic) 
                    # logger.info('dic[val]', dic['val']) 
                    # logger.info('get prop by name:', self.qcm.get_prop_by_name(dic['val'])) 
                    for ind in idx_joined:
                        prop_dict[ind][n].update(**self.qcm.get_prop_by_name(dic['val']))

                elif dic['source'] == 'ind':
                    # calc prop for this layer
                    idx_str = dic['val']
                    layer_chn = dic['indchn']
                    layer_queue_ids = self.data_saver.get_queue_id_marked_rows(layer_chn, dropnanmarkrow=False)
                    layer_chn_idx = list(layer_queue_ids.index)
                    idx_layer = UIModules.index_from_str(idx_str, layer_chn_idx, join_segs=False)
                    idx_layer_joined = UIModules.index_from_str(idx_str, layer_chn_idx, join_segs=True)
                    # logger.info('idx_layer_joined', idx_layer_joined) 
                    if idx_layer_joined:
                        queue_ids_layer = layer_queue_ids[idx_layer_joined]

                    # create qcm_df
                    qcm_df_layer_chn = self.data_saver.df_qcm(layer_chn)
                    qcm_df_layer = qcm_df_layer_chn.loc[idx_layer_joined] # df of current layer
                    # create qcm_df by interpolation
                    qcm_df_layer = self.data_saver.shape_qcmdf_b_to_a(qcm_df_calc, qcm_df_layer, idx, idx_layer)
                    # get values for each
                    # logger.info('qcm_df_layer', qcm_df_layer) 

                    nh = QCM.nhcalc2nh(nhcalc)

                    if n == 0: # electrode layer
                        for ind in idx_joined:
                            electrode = QCM.prop_default['electrode']
                            prop_dict[ind][n].update(**electrode)
                    else: # upper layers
                        if self.get_mp_cores() > 1:
                            ## multiprocessing function
                            logger.info('use %s cores for layer prop calc.', self.get_mp_cores())
                            items = [
                                (
                                    ind,
                                    qcm_df_layer.loc[[ind], :].copy(), # as a dataframe
                                    nh,
                                    calctype,
                                    bulklimit, 
                                    refh,
                                    True, # brief_report. we don't need the details of the upper layer calculation
                                    self.qcm.solve_single_queue_to_prop,
                                ) 
                                for ind in idx_joined
                            ]

                            with multiprocessing.Pool(self.get_mp_cores()) as p:
                                layern_prop_list = p.starmap(mp_solve_single_queue_to_prop, items)
                            p.close()
                            p.join()

                            for ind, ind_dic in layern_prop_list:
                                prop_dict[ind][n].update(**ind_dic)
                        else: # use loop
                            for ind in idx_joined:
                                qcm_queue = qcm_df_layer.loc[[ind], :].copy() # as a dataframe
                                # get prop
                                drho, grho_refh, phi, dlam_refh, err = self.qcm.solve_single_queue_to_prop(nh, qcm_queue, calctype=calctype, bulklimit=bulklimit)

                                prop_dict[ind][n].update(drho=drho, grho=grho_refh, phi=phi, n=refh)
                else: 
                    print('source not defined!')

        # logger.info('prop_dict') 
        # logger.info(prop_dict) 
        
        # 5. do calc with each nhcalc
        mech_df = self.data_saver.update_mech_df_shape(chn_name, nhcalc) # this also update in data_saver

        # logger.info(mech_df) # mech_df from data_saver is all nan (passed)
        
        # if live update is not needed, use QCM.analyze to replace. the codes should be the same
        nh = QCM.nhcalc2nh(nhcalc)
        # TODO: mp
        if self.get_mp_cores() > 1:
            # proce mp
            logger.info('use %s core(s) for film prop calc.', self.get_mp_cores())
            with multiprocessing.Pool(self.get_mp_cores()) as p:
                # mp_cal_film(ind, qcm_queue, mech_queue, prop_dict_ind, nh, calctype, bulklimit,  all_nhcaclc_harm_not_na, solve_single_queue)
                items = [
                    (
                        ind, 
                        qcm_df.loc[[ind], :].copy(), 
                        mech_df.loc[[ind], :].copy(), 
                        prop_dict[ind],
                        nh,
                        calctype,
                        bulklimit,
                        self.qcm 
                    ) 
                    for ind in idx_joined
                    ]
                mech_queue_list = p.starmap(mp_solve_single_queue, items)
                p.close()
                p.join()

            print('mp done. ({})'.format(time.time()-t0))

        else: # use loop
            logger.info('use loop (1 core) for film prop calc.')
            mech_queue_list = []
            for ind in idx_joined: # iterate all ids
                # logger.info('ind', ind) 
            # logger.info('ind', ind) 
                # logger.info('ind', ind) 
                # qcm data of queue_id
                qcm_queue = qcm_df.loc[[ind], :].copy() # as a dataframe
                # mechanic data of queue_id
                mech_queue = mech_df.loc[[ind], :].copy()  # as a dataframe  
                # !! The copy here will not work, since mech_df contains object and the data change to mech_queue will be updated in mech_df 
                # create a dump df which is a copy of mech_df.loc[[ind], :]
                # mech_queue = pd.DataFrame.from_dict(mech_df.loc[[ind], :].to_dict())
                mech_queue['queue_id'] = mech_queue['queue_id'].astype('int')

                # obtain the solution for the properties
                if self.qcm.all_nhcaclc_harm_not_na(nh, qcm_queue):
                    # solve a single queue
                    mech_queue = self.qcm.solve_single_queue(nh, qcm_queue, mech_queue, calctype=calctype, film=prop_dict[ind], bulklimit=bulklimit)

                    # save back to mech_df
                    mech_queue.index = [ind] # not necessary
                    # mech_df.update(mech_queue)
                    # mech_df['queue_id'] = mech_df['queue_id'].astype('int')
                    # self.data_saver.update_mech_queue(chn_name, nhcalc, mech_queue) # update to mech_df in data_saver
                    
                    if self.settings['checkBox_settings_mech_liveupdate']: # live update
                        # update tableWidget_spectra_mechanics_table
                        self.ui.spinBox_spectra_mechanics_currid.setValue(ind)

                else:
                    # since the df already initialized with nan values, nothing to do here
                    pass
                mech_queue_list.append(mech_queue)

        # update all data together
        mech_df_new = pd.concat(mech_queue_list)
        logger.info(mech_df_new.head())
        self.data_saver.update_mech_df(chn_name, nhcalc, mech_df_new)

        # skip liveupdate for mp
        # update table with the index of the last None value in mech_queue_list
        last_non_none_ind = next(len(mech_queue_list) - i for i, q in enumerate(reversed(mech_queue_list), 1) if q is not None)

        if self.ui.spinBox_spectra_mechanics_currid.value() != last_non_none_ind: # id is not current, set id to the last one. The table should be updated once the id changed.
            self.ui.spinBox_spectra_mechanics_currid.setValue(idx_joined[-1])
        else:
            # id did not change, force to update table
            self.update_spectra_mechanics_table()

        print('{} calculation finished.'.format(nhcalc))

        # # save back to data_saver
        # self.data_saver.update_mech_df_in_prop(chn_name, nhcalc, refh, mech_df)

        print('Time spent: {} s'.format(time.time()-t0))
        

    def backup_mech_solve_chn(self, chn_name, queue_ids):
        '''
        NOT USING
        send the data to qcm module to solve in secquence by queue_ids and
        save the returned mechanic data to data_saver
        '''

        if not self.data_saver.path:
            print('No data available!')
            return

        # set f1 to qcm module (leave this to qcm modue for each single solution)

        self.qcm.refh = int(self.settings['comboBox_settings_mechanics_refG']) # use user defined refernece harmonic
        refh = self.qcm.refh # reference harmonic

        logger.info('refh: %s', refh) 
        # get nhcalc
        nhcalc_list = self.gen_nhcalc_list()
        # get qcm data (columns=['queue_id', 't', 'temp', 'marks', 'fstars', 'fs', 'gs', 'delfstars', 'delfs', 'delgs', 'f0stars', 'f0s', 'g0s'])
        # 'delf', 'delgs' may not necessary
        qcm_df = self.data_saver.df_qcm(chn_name)

        # do calc with each nhcalc
        for nhcalc in nhcalc_list:
            mech_df = self.data_saver.update_mech_df_shape(chn_name, nhcalc) # this also update in data_saver

            logger.info(mech_df) # mech_df from data_saver is all nan (passed)

            # if live update is not needed, use QCM.analyze to replace. the codes should be the same
            nh = QCM.nhcalc2nh(nhcalc)
            for queue_id in queue_ids: # iterate all ids
                logger.info('queue_id: %s', queue_id) 
                # logger.info('qcm_df: %s', qcm_df) 
                logger.info(type(qcm_df)) 
                # queue index
                idx = qcm_df[qcm_df.queue_id == queue_id].index.astype(int)[0]
                # idx = qcm_df[qcm_df.queue_id == queue_id].index
                logger.info('index: %s', qcm_df.index) 
                logger.info('index: %s', mech_df.index) 
                logger.info('idx: %s', idx) 
                # qcm data of queue_id
                qcm_queue = qcm_df.loc[[idx], :].copy() # as a dataframe
                # mechanic data of queue_id
                mech_queue = mech_df.loc[[idx], :].copy()  # as a dataframe 
                # !! The copy here will not work, since mech_df contains object and the data change to mech_queue will be updated in mech_df 

                # create a dump df which is a copy of mech_df.loc[[idx], :]
                # mech_queue = pd.DataFrame.from_dict(mech_df.loc[[idx], :].to_dict())
                mech_queue['queue_id'] = mech_queue['queue_id'].astype('int')
                # exit(0)

                # obtain the solution for the properties
                if self.qcm.all_nhcaclc_harm_not_na(nh, qcm_queue):
                    # solve a single queue
                    mech_queue = self.qcm.solve_single_queue(nh, qcm_queue, mech_queue)

                    # save back to mech_df
                    mech_queue.index = [idx] # not necessary
                    # mech_df.update(mech_queue)
                    # mech_df['queue_id'] = mech_df['queue_id'].astype('int')
                    self.data_saver.update_mech_queue(chn_name, nhcalc, mech_queue) # update to mech_df in data_saver
                    
                    if self.settings['checkBox_settings_mech_liveupdate']: # live update
                        # update tableWidget_spectra_mechanics_table
                        self.ui.spinBox_spectra_mechanics_currid.setValue(idx)

                else:
                    # since the df already initialized with nan values, nothing to do here
                    pass

            print('{} calculation finished.'.format(nhcalc))

            # # save back to data_saver
            # self.data_saver.update_mech_df_in_prop(chn_name, nhcalc, refh, mech_df)

            if not self.settings['checkBox_settings_mech_liveupdate']: 
                # update table
                self.update_spectra_mechanics_table()


    def mech_clear(self):
        self.data_saver.clr_mech_df_in_prop()


    def mech_solve_test(self):
        '''
        for code testing
        '''
        logger.info('stackedWidget_settings_mechanics_modeswitch: %s', self.ui.stackedWidget_settings_mechanics_modeswitch.currentIndex()) 
        if self.ui.stackedWidget_settings_mechanics_modeswitch.currentIndex() == 0: # model mode
            # convert model to mechchndata
            film_dict = self.make_film_dict_by_mechmodel_widgets()
            # logger.info("self.settings['mechchndata']") 
            # logger.info(self.settings['mechchndata']) 
        else:
            film_dict = self.make_film_layers_dict()
        logger.info(film_dict) 


    def mech_solve_all(self):
        queue_ids = self.data_saver.get_queue_id_marked_rows(self.mech_chn, dropnanmarkrow=False)
        self.mech_solve_chn(self.mech_chn, queue_ids)


    def mech_solve_marked(self):
        queue_ids = self.data_saver.get_queue_id_marked_rows(self.mech_chn, dropnanmarkrow=True)
        self.mech_solve_chn(self.mech_chn, queue_ids)


    def mech_solve_new(self):
        queue_id_diff = self.data_mech_queue_ids_diff()

        if queue_id_diff.empty:
            print('No new data to calculate.')
        else:
            self.mech_solve_chn(self.mech_chn, queue_id_diff)


    def data_mech_queue_ids_diff(self):
        '''
        return the difference between data and mech queue_id
        '''
        data_queue_ids = self.data_saver.get_queue_id_marked_rows(self.mech_chn, dropnanmarkrow=False)
        
        nhcalc = self.gen_nhcalc_str()
        mech_queue_ids = self.data_saver.get_mech_queue_id(self.mech_chn, nhcalc)

        queue_id_diff =  data_queue_ids[data_queue_ids.isin(set(data_queue_ids) - set(mech_queue_ids))]

        return queue_id_diff


    def set_mech_layer_0_indchn_val(self):
        '''
        set layer 0 value of mech_layers
        '''
        # get ref_tempmode
        ref_refmode= self.settings['comboBox_settings_data_ref_tempmode']
        samprefsource = self.settings['comboBox_settings_data_samprefsource']
        samprefidx = self.settings['lineEdit_settings_data_samprefidx']
        refrefsource = self.settings['comboBox_settings_data_refrefsource']
        refrefidx = self.settings['lineEdit_settings_data_refrefidx']

        # set mechchndata
        # const and var are the same for now
        if ref_refmode== 'const':
            self.set_mechchndata('comboBox_mech_expertmode_indchn_0', samprefsource, mech_chn='samp')
            self.set_mechchndata('comboBox_mech_expertmode_indchn_0', refrefsource, mech_chn='ref')
        elif ref_refmode== 'var': # use the same reference
            self.set_mechchndata('comboBox_mech_expertmode_indchn_0', samprefsource, mech_chn='samp')
            self.set_mechchndata('comboBox_mech_expertmode_indchn_0', refrefsource, mech_chn='ref')
        else:
            pass
        
        self.set_mechchndata('lineEdit_mech_expertmode_value_0', samprefidx, mech_chn='samp')
        self.set_mechchndata('lineEdit_mech_expertmode_value_0', refrefidx, mech_chn='ref')
        
        # load film layers data 
        self.load_film_layers_widgets()


    def on_changed_spinBox_spectra_mechanics_currid(self, ind):
        '''
        collect data and send to update_spectra_mechanics_table
        '''
            # updaate in mech table
        self.update_spectra_mechanics_table()


    def make_film_dict_by_mechmodel_widgets(self):
        '''
        this function set mechchdata of 
            spinBox_mech_expertmode_layernum value
            comboBox_mech_expertmode_calc_0/1/2
            comboBox_mech_expertmode_source_0/1/2
            comboBox_mech_expertmode_indchn_0/1/2
        by mechmodel_widgets value
        We keep model and layers mode independent to make sure they don't influence each other
        '''
        layer_name = {
            '1': 'samplayer', 
            '2': 'overlayer',
        }
        # initialize fil_dict
        film_dict = {}

        if not self.settings['comboBox_settings_mechanics_selectmodel']: # in case comboBox_settings_mechanics_selectmodel is empty
            self.settings['comboBox_settings_mechanics_selectmodel'] = self.ui.comboBox_settings_mechanics_selectmodel.itemData(self.ui.comboBox_settings_mechanics_selectmodel.currentIndex())

        model = self.settings['comboBox_settings_mechanics_selectmodel'] # onelayer, bulk, twolayers
        logger.info('model: %s', model) 
        if model == 'onelayer' or model == 'bulk':
            n_layers = 1
        elif model == 'twolayers':
            n_layers = 2        
        else:
            return film_dict
        
        # set layer 1/2
        for n in range(1, n_layers+1):
            n = str(n)
            film_dict[n] = {
                'calc': True if n == '1' else False,
                'source': 'ind',
                'indchn': getattr(self.ui, 'comboBox_settings_mechanics_model_' + layer_name[n] + '_chn').itemData(getattr(self.ui, 'comboBox_settings_mechanics_model_' + layer_name[n] + '_chn').currentIndex()),
                'val': getattr(self.ui, 'lineEdit_settings_mechanics_model_' + layer_name[n] + '_idx').text(),
            }

        film_dict['0'] = {
            'calc': False,
            'source': 'ind',
            'indchn': self.settings['comboBox_settings_data_'+self.mech_chn+'refsource'],
            'val': self.settings['lineEdit_settings_data_'+self.mech_chn+'refidx'],
        }

        return film_dict


    def set_mechmodel_widgets(self):
        '''
        this function set visible & value (to default) of 
            label_settings_mechanics_model_overlayer
            comboBox_settings_mechanics_model_overlayer_chn
            lineEdit_settings_mechanics_model_overlayer_idx
        by comboBox_settings_mechanics_selectmodel value
        and
        '''
        model = self.settings['comboBox_settings_mechanics_selectmodel'] # onelayer, bulk, twolayers
        logger.info('model: %s', model) 
        if model == 'onelayer' or model == 'bulk':

            # show samplayer widgets
            self.show_widgets('mech_model_show_hide_samplayer_list')
            # hide overlayer widgets
            self.hide_widgets('mech_model_show_hide_overlayer_list')
 
        elif model == 'twolayers':
           
            # show samplayer widgets
            self.show_widgets('mech_model_show_hide_samplayer_list')
            # show overlayer widgets
            self.show_widgets('mech_model_show_hide_overlayer_list')

        else:
            # hide all layers
            # hide samplayer widgets
            self.hide_widgets('mech_model_show_hide_samplayer_list')
            # hide overlayer widgets
            self.hide_widgets('mech_model_show_hide_overlayer_list')


    def update_spectra_mechanics_table(self):
        '''
        this function update data in tableWidget_spectra_mechanics_table
        and relative information displaying
        '''
        # clear table
        table = self.ui.tableWidget_spectra_mechanics_table
        table.clearContents()
        
        if not self.data_saver.path: # no data
            return

        ## get variables
        # index
        ind = self.ui.spinBox_spectra_mechanics_currid.value()

        # channel name
        chn_name = self.mech_chn

        # check if index is in the range of df_qcm
        qcm_df = self.data_saver.df_qcm(chn_name)

        if ind not in qcm_df.index:
            ind = qcm_df.index[-1] # set ind no more than qcm_df.idx
            self.ui.spinBox_spectra_mechanics_currid.setValue(ind)
            return

        # nhcalc
        nhcalc = self.gen_nhcalc_str()

        logger.info('ind: %s', ind) 
        logger.info('chn_name: %s', chn_name) 
        # logger.info('refh', refh) 
        logger.info('nhcalc: %s', nhcalc) 

        # check if solution is in data_saver
        mech_key = self.data_saver.get_mech_key(nhcalc)
        if mech_key not in self.data_saver.get_prop_keys(chn_name): # no solution stored of given combination
            logger.warning('Solution of %s does not exist.', mech_key)
            return

        mech_df = self.data_saver.get_mech_df_in_prop(chn_name, nhcalc)

        # logger.info('qcm_df: %s', qcm_df) 
        # logger.info('mech_df: %s', mech_df) 

        # check index range
        if ind not in mech_df.index:
            # ind = mech_df.index[-1] # set ind as the last in mech_df
            logger.warning('%s has not been solved.', ind) 
            return
        else:
            logger.info('ind in range.') 

            nhcalc = self.gen_nhcalc_str()

            logger.info('ind: %s', ind) 
            logger.info('chn_name: %s', chn_name) 
            # logger.info('refh', refh) 
            logger.info('nhcalc: %s', nhcalc) 

            # # check if solution is in data_saver
            # mech_key = self.data_saver.get_mech_key(nhcalc)
            # if mech_key not in self.data_saver.get_prop_keys(chn_name): # no solution stored of given combination
            #     logger.warning('Solution of {} does not exist.'.format(mech_key))
            #     return

            # mech_df = self.data_saver.get_mech_df_in_prop(chn_name, nhcalc)

            # logger.info('qcm_df: %s', qcm_df) 
            # logger.info('mech_df: %s', mech_df) 

            # get queue_id
            # logger.info(qcm_df.queue_id) 
            queue_id = qcm_df.queue_id.loc[ind]

            # qcm data of queue_id
            qcm_queue = qcm_df.loc[[ind], :].copy() # as a dataframe
            # mechanic data of queue_id
            mech_queue = mech_df.loc[[ind], :].copy()  # as a dataframe 
            # logger.info('qcm_queue: %s', qcm_queue) 
            # logger.info('mech_queue: %s', mech_queue) 
            

        if (qcm_queue is None) and (mech_queue is None):
            logger.info('qcm_queue is None: %s; mech_queue is None:  %s', qcm_queue is None, mech_queue is None)
            return

        # convert grho, drho and phi unit in mech_queue
        mech_queue = self.qcm.convert_mech_unit(mech_queue)
        # get n of rows and columns of the table
        tb_rows = table.rowCount()
        tb_cols = table.columnCount()

        # get keys in qcm_queue and mech_queue
        qcm_cols = qcm_queue.columns
        mech_cols = mech_queue.columns

        # update table contents
        for tb_row in range(tb_rows):
            vh = table.verticalHeaderItem(tb_row).text()
            # logger.info('vh: %s', vh) 
            # find corresponding key in qcm_queue or mech_queue
            for key, val in config_default['mech_table_rowheaders'].items():
                if vh == val:
                    df_colname = key
                    # logger.info(key) 
                    if df_colname in qcm_cols:
                        df_queue = qcm_queue
                        # logger.info(qcm_cols) 
                    elif df_colname in mech_cols:
                        df_queue = mech_queue
                        # logger.info(mech_cols) 
                    else:
                        logger.info('did not find %s %s', key, val)
                        df_colname = '' # not in df
                    break
                else:
                    # logger.info(vh.encode('utf-8')) 
                    # logger.info(val.encode('utf-8')) 
                    df_colname = ''
            if df_colname:
                row_data = df_queue[df_colname].iloc[0]
                # logger.info('type(row_data): %s', type(row_data)) 
                # logger.info(df_queue[df_colname]) # (it is a series)
                for tb_col in range(tb_cols): # update the row by columns
                    # logger.info('r,c: %s %s', tb_row, tb_col) 
                    # if df_colname.endswith('s'): # multiple values
                    #     data = df_queue[df_colname].iloc[0][tb_col]
                    # else:
                    #     data = df_queue[df_colname].iloc[0]
                    if isinstance(row_data, list):
                        # logger.info(df_queue[df_colname].shape) 
                        # logger.info(df_queue[df_colname].iloc[0]) 
                        # logger.info(df_queue[df_colname].columns) 
                        data = df_queue[df_colname].iloc[0][tb_col]
                    else:
                        data = row_data
                    # logger.info(data) 

                    # format data 
                    if data is not None: # the following format does not work with None
                        data = format(data, config_default['mech_table_number_format'])

                    tableitem = self.ui.tableWidget_spectra_mechanics_table.item(tb_row, tb_col)
                    if tableitem: # tableitem != 0
                        logger.info('item set') 
                        tableitem.setText(data)
                    else: # item is not set
                        # logger.info('item not set') 
                        self.ui.tableWidget_spectra_mechanics_table.setItem(tb_row, tb_col, QTableWidgetItem(data))
        # logger.info(self.ui.tableWidget_spectra_mechanics_table.item(0,0)) 
        self.ui.tableWidget_spectra_mechanics_table.viewport().update() # TODO update doesn't work. update in UI

        # TODO update contours if checked


    def gen_nhcalc_str(self):
        '''
        generate nhcalc str from relative widgets
        spinBox_settings_mechanics_nhcalc_n1/2/3
        '''
        n1 = self.settings['spinBox_settings_mechanics_nhcalc_n1']
        n2 = self.settings['spinBox_settings_mechanics_nhcalc_n2']
        n3 = self.settings['spinBox_settings_mechanics_nhcalc_n3']
        return ''.join(map(str, [n1, n2, n3]))


    def gen_nhcalc_list(self):
        '''
        make list from nhcalc strs
        '''
        #TODO can be extanded to multiple strings
        return [self.gen_nhcalc_str()]


    def mechanics_plot_r_time(self):
        self.mechanics_plot('t')


    def mechanics_plot_r_temp(self):
        self.mechanics_plot('temp')


    def mechanics_plot_r_idx(self):
        self.mechanics_plot('idx')


    def mechanics_plot_r1_r2(self):
        self.mechanics_plot('r1r2')


    def mechanics_plot_r2_r1(self):
        self.mechanics_plot('r2r1')


    def mechanics_plot(self, plot_type):
        '''
        make plot by plot_type
        variable is given by row selection of tableWidget_spectra_mechanics_table
        '''
        x_list = ['t', 'temp', 'idx'] # common x-axis variables. t, temp is the column name, idx is not.
        logger.info('plot_type %s', plot_type) 

        # get chn_name
        chn_name = self.mech_chn
        logger.info('chn_name %s', chn_name) 

        # get mech_key
        nhcalc = self.gen_nhcalc_str()
        logger.info('nhcalc %s', nhcalc) 

        refh =int(self.settings['spinBox_settings_mechanics_nhcalc_n3']) # refh 
        logger.info('refh %s', refh) 

        mech_key = self.data_saver.get_mech_key(nhcalc)
        logger.info('mech_key %s', mech_key) 
        logger.info('prop_chn_keys: %s', getattr(self.data_saver, chn_name + '_prop').keys()) 

        # check if data exists mech_key
        if mech_key not in getattr(self.data_saver, chn_name + '_prop').keys(): # no corresponding prop data
            print('There is no property data of {}'.format(mech_key))
            return

        if not self.data_mech_queue_ids_diff().empty:
            print('there are new data not calculated. Slove those before plotting.')
            return

        # get harmonics to plot
        plt_harms = [harm for harm in self.all_harm_list(as_str=True) if self.settings.get('checkBox_nhplot' + harm, None)]
        logger.info('plt_harms: %s', plt_harms) 

        if not plt_harms: # no harmonic selected
            return

        # get variables to plot
        if plot_type == 'contourdata':
            varplots = [['phi', 'dlams']]
        else:
            varplot = []
            selrowidx = self.ui.tableWidget_spectra_mechanics_table.selectionModel().selectedRows() # fully selected rows
            for r in selrowidx:
                logger.info(r.row()) 
                vh = self.ui.tableWidget_spectra_mechanics_table.verticalHeaderItem(r.row()).text()
                logger.info(vh) 
                for key, val in config_default['mech_table_rowheaders'].items():
                    if vh == val:
                        varplot.append(key)

            if not varplot: # no variable selected
                return

        # get data mode showall or marked
        # from tabWidget_settings
        idx = []
        prop_plot_list = []
        if (plot_type == 'contourdata'): # contour data 
            prop_plot_list.extend([
                self.ui.mpl_contour1,
                self.ui.mpl_contour2,
            ])

        if (plot_type == 'contourdata') and (self.settings['comboBox_settings_mechanics_contourdata'] == 'w_curr'): # contour data with current data
            prop_plot_list.extend([
                self.ui.mpl_contour1,
                self.ui.mpl_contour2,
            ])
            logger.info('current ind: %s', idx) 
            prop_group = 'p'
            line_group = 'l'
            mark = False

            # get current idx
            # since spinBox_spectra_mechanics_currid is not saved in self.settings we access it directly
            idx.append(self.ui.spinBox_spectra_mechanics_currid.value())

        elif self.show_marked_data() and self.data_saver.with_marks(chn_name): # show marked data only
            prop_group = 'pm'
            line_group = 'l'
            mark = True
        else: # show all data
            prop_group = 'p'
            line_group = 'l'
            mark = False

        # create varplots (list of [var_y, var_x] for plots)
        if plot_type in x_list: # y vs. time/temp/idx
            varplots = [[var, plot_type] for var in varplot]
        elif plot_type in ['r1r2', 'r2r1']:
            if len(varplot) < 2: # not enough variables selected
                print('Not enough rows are selected! Please select 2 rows.')
                return
            elif len(varplot) > 2: # too many variables selected
                print('Too many rows are selected! Please select 2 rows.')
                return

            if plot_type == 'r2r1':
                varplot.reverse() # reverse varplot

            varplots = [varplot] # list of plots to plot

        for var_yx in varplots: # for each plot to plot
            # var_yx[0] as y
            ydata, yerr = self.get_data_from_data_or_prop(chn_name, mech_key, var_yx[0], mark=mark, idx=idx)
            ylabel = self.get_label_replace_refh_unit(var_yx[0], refh)

            # var_yx[1] as x
            xdata, xerr = self.get_data_from_data_or_prop(chn_name, mech_key, var_yx[1],  mark=mark, idx=idx)
            xlabel = self.get_label_replace_refh_unit(var_yx[1], refh)

            ## make the plot
            if plot_type not in ['contourdata']: # not contour data, no plot given
                # create figure
                self.prop_plot_list.append(
                    MatplotlibWidget(
                        parent=self.ui.scrollArea_data_mechanics_plots,
                        axtype='prop',
                        showtoolbar=True,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=mech_key,
                    )
                )
            # check if data is harmonic dependent
            if var_yx[0].endswith('s') or var_yx[1].endswith('s'):
                figharms = plt_harms
            else:
                figharms = [str(refh)]
            # prepare marker data
            prop_list = self.prepare_harm_data_for_mpl_prop_update(chn_name, figharms, prop_group, xdata, ydata, xerr, yerr, show_marked_when_all=False) #

            logger.info('prop_list: %s', prop_list) 

            # calc value from single harm for line_grop
            if '_calc' in var_yx[0] or '_calc' in var_yx[1]: # var in mech_keys_multiple
                line_xdata = xdata
                line_ydata = ydata
                if '_calc' in var_yx[0]:
                    line_ydata, _ = self.get_data_from_data_or_prop(chn_name, mech_key, var_yx[0].replace('calc', 'exp'),  mark=mark, idx=idx)
                if '_calc' in var_yx[1]:
                    line_xdata, _ = self.get_data_from_data_or_prop(chn_name, mech_key, var_yx[1].replace('calc', 'exp'),  mark=mark, idx=idx)
                # prepare line data
                line_list = self.prepare_harm_data_for_mpl_prop_update(chn_name, figharms, line_group, line_xdata, line_ydata, show_marked_when_all=False) # use the same value

            else:
                # line data
                line_list = self.prepare_harm_data_for_mpl_prop_update(chn_name, figharms, line_group, xdata, ydata, show_marked_when_all=False) # use the same value

            logger.info('line_list: %s', line_list) 
            # update data in figure
            if plot_type in ['contourdata']: 
                # we have two plots to update
                for prop_plot in prop_plot_list:
                    prop_plot.update_data(*prop_list, *line_list)
            else:
                if len(self.prop_plot_list) > 0: # there are prop plot
                    self.prop_plot_list[-1].update_data(*prop_list, *line_list)
                    # add to scrollarea
                    self.update_mpl_to_prop_scrollarea()
                else: # there is no prop plot in the list or some thing wrong
                    logger.info(self.prop_plot_list) # to show the structrue for debug
                    pass


    def get_label_replace_refh_unit(self, var, refh):
        '''
        get label from config_default and replace '_refh' and '<unit>' in it
        '''
        label = config_default['data_plt_axis_label'][var]
        if '_refh' in var: # variable referenced to refh
            label = label.replace('{refh}', '{' + str(refh) + '}')
        if var == 't':
            label = self.time_str_unit_replace(label)
        elif var == 'temp':
            label = self.temp_str_unit_replace(label)
        return label


    def get_data_from_data_or_prop(self, chn_name, mech_key, var, mark=False, idx=[]):
        '''
        get data from data_saver.<chn_name> or data_saver.<chn_name + _prop>[mech_key]
        '''
        data, err = None, None # inintiate value

        # get keys
        data_cols = list(getattr(self.data_saver, chn_name).columns)
        prop_cols = list(getattr(self.data_saver, chn_name+'_prop')[mech_key].columns)

        # NOTE: for now, we don't select single index from the data. So by index is only used to prop data set!
        
        if var in data_cols + ['delf_exps', 'delg_exps']: # delg delf is in data, so add them to the check list
            # get timeunit
            timeunit = self.settings['comboBox_timeunit']
            # get tempunit
            tempunit = self.settings['comboBox_tempunit']

            data = self.get_data_by_typestr(var, chn_name, mark=mark, unit_t=timeunit, unit_temp=tempunit)

        elif var in prop_cols: # delg delf is in data, so add them to the check list
            if not idx: # use mark
                data = self.data_saver.get_mech_column_to_columns_marked_rows(chn_name, mech_key, var, mark=mark, dropnanmarkrow=False)
            else: # use index list
                data = self.data_saver.get_mech_column_to_columns_by_idx(chn_name, mech_key, var, idx=idx)
            data = self.qcm.convert_mech_unit(data) # convert unit for plot
            # get yerr
            if self.settings['checkBox_settings_mechanics_witherror'] and (var + '_err' in getattr(self.data_saver, chn_name + '_prop')[mech_key].keys()): # corresponding error exists
                err = self.data_saver.get_mech_column_to_columns_marked_rows(chn_name, mech_key, var + '_err', mark=mark, dropnanmarkrow=False)
                err = self.qcm.convert_mech_unit(err)
            else:
                err = None
        elif var == 'idx': # get index
            data = self.data_saver.get_idx_marked_rows(chn_name, dropnanmarkrow=False)
            err = None

        return data, err


    def update_mpl_to_prop_scrollarea(self):
        '''
        add mpl figure (prop_plot_list[-1]) into scrollArea_data_mechanics_plots
        at the end of its layout
        '''
        n = len(self.prop_plot_list)
        mpl = self.prop_plot_list[-1]

        ncols = config_default['prop_plot_ncols']

        self.ui.gridLayout_propplot.addWidget(mpl, (n-1)//ncols, (n-1)%ncols)
        if (n-1)%ncols == 0: # start a new row # old code: not (n-1)%ncols
            self.ui.gridLayout_propplot.setRowMinimumHeight((n-1)//2, config_default['prop_plot_minmum_row_height'])
        # return
        # self.ui.scrollArea_data_mechanics_plots.setWidget(mpl)
        # self.ui.scrollArea_data_mechanics_plots.show()
        # return
        # layout = self.ui.scrollArea_data_mechanics_plots.layout()
        # layout.insertWidget(layout.count() - 1, mpl)


    def del_prop_plot(self):
        '''
        delete all prop plots in
        '''
        for i in reversed(range(self.ui.gridLayout_propplot.count())):
            item = self.ui.gridLayout_propplot.itemAt(i)
            item.widget().deleteLater()
        for r in range(self.ui.gridLayout_propplot.rowCount()):
            self.ui.gridLayout_propplot.setRowMinimumHeight(r, 0)
        self.prop_plot_list = []


    #### end mech related funcs ###


    #### mech contour funcs ####


    def prep_contour_mesh(self, contour_type=None):
        '''
        create contour mesh for both normf/normg & rh/rd
        mesh1 = {'X': array, 'Y': array, 'Z': array}
        mesh2 = {'X': array, 'Y': array, 'Z': array}
        '''
        # initialize return values
        mesh1, mesh2 = {}, {}

        contour_lim = self.settings['contour_plot_lim_tab']
        contour_array = config_default['contour_array']

        # make meshgrid for contour
        phi = np.linspace(contour_lim['phi']['min'], contour_lim['phi']['max'], contour_array['num']) 
        dlam = np.linspace(contour_lim['dlam']['min'], contour_lim['dlam']['max'], contour_array['num'])

        dlam_i, phi_i = np.meshgrid(dlam, phi * np.pi / 180) # convert phi to rad

        mesh1['X'], mesh1['Y'] = dlam_i, phi_i * 180 / np.pi # convert back to deg
        mesh2['X'], mesh2['Y'] = dlam_i, phi_i * 180 / np.pi # convert back to deg

        # set self.qcm.refh
        self.qcm.refh = self.settings['spinBox_settings_mechanics_nhcalc_n3'] # int

        nhcalc = self.gen_nhcalc_str() # str of harmonics
        nh = QCM.nhcalc2nh(nhcalc) # list of harmonics in int

        if contour_type.lower() == 'normfnormg':
            # calculate the z values and reset things to -1 at dlam=0
            normdelfstar = self.qcm.normdelfstar(self.qcm.refh, dlam_i, phi_i)
            normdelfstar[:,0] = -1

            mesh1['Z'], mesh2['Z'] = np.real(normdelfstar), np.imag(normdelfstar)
            
        elif contour_type.lower() == 'rhrd':
            rh_i = self.qcm.rhcalc(nh, dlam_i, phi_i)
            rd_i = self.qcm.rdcalc(nh, dlam_i, phi_i)
            mesh1['Z'], mesh2['Z'] = rh_i, rd_i
        else: # contour_type not found
            mesh1['Z'], mesh2['Z'] = np.random.rand(*dlam_i.shape), np.random.rand(*dlam_i.shape) # just make some random numbers for fun

        return mesh1, mesh2


    def make_contours(self):
        '''
        plot contours by settings
        '''
        # get factors
        contour_type = self.settings['comboBox_settings_mechanics_contourtype']
        contour_array = config_default['contour_array']
        contour_lim = self.settings['contour_plot_lim_tab']
        # logger.info('contour_array: %s', contour_array) 
        # logger.info('contour_lim: %s', contour_lim) 
        cmap = self.settings['comboBox_settings_mechanics_contourcmap']
        logger.info('cmap %s', cmap)

        if contour_type.lower() == 'normfnormg':
            levels1 = self.make_contour_levels(contour_lim['normf']['min'], contour_lim['normf']['max'], contour_array['levels'])
            levels2 = self.make_contour_levels(contour_lim['normg']['min'], contour_lim['normg']['max'], contour_array['levels'])
            title1 = config_default['contour_title']['normf']
            title2 = config_default['contour_title']['normg']
        elif contour_type.lower() == 'rhrd':
            levels1 = self.make_contour_levels(contour_lim['rh']['min'], contour_lim['rh']['max'], contour_array['levels'])
            levels2 = self.make_contour_levels(contour_lim['rd']['min'], contour_lim['rd']['max'], contour_array['levels'])
            title1 = config_default['contour_title']['rh']
            title2 = config_default['contour_title']['rd']
        else:
            levels1, levels2 = contour_array['levels'], contour_array['levels']

        mesh1, mesh2 = self.prep_contour_mesh(contour_type=contour_type)
        self.ui.mpl_contour1.init_contour(**mesh1, levels=levels1, cmap=cmap, title=title1)
        self.ui.mpl_contour2.init_contour(**mesh2, levels=levels2, cmap=cmap, title=title2)

        # linkxy
        self.ui.mpl_contour1.ax[0].get_shared_x_axes().join(
            self.ui.mpl_contour1.ax[0],
            self.ui.mpl_contour2.ax[0]
        )
        self.ui.mpl_contour1.ax[0].get_shared_y_axes().join(
            self.ui.mpl_contour1.ax[0],
            self.ui.mpl_contour2.ax[0]
        )

        # add data to contour
        # self.add_data_to_contour() # NOTE: comment this and let the data updated only by clicking the button


    def add_data_to_contour(self):
        contour_data = self.settings['comboBox_settings_mechanics_contourdata']
        # clear contour data
        self.clear_all_contour_data()
        # add data to contour
        logger.info('contour_data: %s', contour_data) 
        if contour_data == 'none': # w/o data
            return
        elif contour_data in ['w_curr', 'w_data']: # w/ current data or w/ data (all or marked)
            self.mechanics_plot('contourdata')
        
        self.set_contour_lims()


    def clear_all_contour_data(self):
        '''
        clear contour data but not contour
        '''
        self.ui.mpl_contour1.clr_all_lines()
        self.ui.mpl_contour2.clr_all_lines()

    
    def set_contour_lims(self):
        # set limit by settings
        contour_plot_lim_tab = self.settings['contour_plot_lim_tab']
        dlam_lims = contour_plot_lim_tab['dlam']
        phi_lims = contour_plot_lim_tab['phi']
        self.ui.mpl_contour1.ax[0].set_xlim(dlam_lims['min'], dlam_lims['max'])
        self.ui.mpl_contour1.ax[0].set_ylim(phi_lims['min'], phi_lims['max'])
        self.ui.mpl_contour2.ax[0].set_xlim(dlam_lims['min'], dlam_lims['max'])
        self.ui.mpl_contour2.ax[0].set_ylim(phi_lims['min'], phi_lims['max'])


    def make_contour_levels(self, lmin, lmax, interv):
        '''
        generate an array for contour_levels
        '''
        return np.linspace(lmin, lmax, interv)


    def on_clicked_set_temp_sensor(self, checked):
        # below only runs when vna is available
        if self.vna: # add not for testing code
            if checked: # checkbox is checked
                # if not self.temp_sensor: # tempModule is not initialized
                # get all tempsensor settings
                tempmodule_name = self.settings['comboBox_tempmodule'] # get temp module

                thrmcpltype = self.settings['comboBox_thrmcpltype'] # get thermocouple type
                tempdevice = TempDevices.device_info(self.settings['comboBox_tempdevice']) #get temp device info

                # # check senor availability
                # package_str = config_default['tempmodules_path'][2:].replace('/', '.') + tempmodule_name
                # logger.info(package_str) 
                # import package
                temp_sensor = getattr(TempModules, tempmodule_name)

                try:
                    self.temp_sensor = temp_sensor(
                        tempdevice,
                        config_default['tempdevices_dict'][tempdevice.product_type],
                        thrmcpltype,
                    )
                except Exception as e: # if failed return
                    print(e)
                    #TODO update in statusbar
                    return

                # after tempModule loaded
                # # tempModule should take one arg 'thrmcpltype' and return temperature in C by calling tempModule.get_tempC
                try:
                    curr_temp = self.temp_sensor.get_tempC()

                    # save values to self.settings
                    self.settings['checkBox_control_rectemp'] = True
                    self.settings['checkBox_settings_temp_sensor'] = True
                    # set statusbar pushButton_status_temp_sensor text
                    self.statusbar_temp_update(curr_temp=curr_temp)
                    # disable items to keep the setting
                    self.disable_widgets(
                        'temp_settings_enable_disable_list'
                    )

                except Exception as e: # failed to get temperature from sensor
                    print(e)
                    # uncheck checkBoxes
                    self.ui.checkBox_control_rectemp.setChecked(False)
                    self.ui.checkBox_settings_temp_sensor.setChecked(False)
                    #TODO update in statusbar
            else: # is unchecked

                self.settings['checkBox_control_rectemp'] = False
                self.settings['checkBox_settings_temp_sensor'] = False

                # set statusbar pushButton_status_temp_sensor text
                self.statusbar_temp_update()

                # enable items to keep the setting
                self.enable_widgets(
                    'temp_settings_enable_disable_list'
                )
                # reset self.temp_sensor
                self.temp_sensor = None


            # update checkBox_settings_temp_sensor to self.settings
            # self.update_tempsensor()


    def statusbar_temp_update(self, curr_temp=None):

        # update statusbar temp sensor image
        if self.settings['checkBox_settings_temp_sensor']: # checked
            self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor.svg"))
            try:
            # get temp and change temp unit by self.settings['temp_unit_opts']
                if curr_temp is None:
                    curr_temp = self.temp_sensor.get_tempC()
                logger.info(curr_temp) 
                unit = config_default['temp_unit_opts'].get(self.settings['comboBox_tempunit'])
                self.ui.pushButton_status_temp_sensor.setText('{:.1f} {}'.format(self.data_saver.temp_C_to_unit(curr_temp, unit=unit), unit))
                self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor.svg"))
                self.ui.pushButton_status_temp_sensor.setToolTip('Temp. sensor is on.')
            except:
                #TODO update in statusbar
                pass
        else:
            self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor_off.svg"))
            self.ui.pushButton_status_temp_sensor.setText('')
            self.ui.pushButton_status_temp_sensor.setToolTip('Temp. sensor is off.')



    def on_clicked_checkBox_dynamicfitbyharm(self, value):
        self.ui.checkBox_dynamicfit.setEnabled(not value)


    def on_clicked_checkBox_fitfactorbyharm(self, value):
        self.ui.spinBox_fitfactor.setEnabled(not value)
        self.ui.label_fitfactor.setEnabled(not value)


    def on_ref_mode_changed(self, signal):
        '''
        This function 
        when the reference mode changes to/out of temperature mode
        '''
        logger.info('ref: %s %s', self.sender().objectName(), signal) 
        value = self.sender().itemData(signal)
        parent = self.ui.treeWidget_settings_data_refs

        # change visible of comboBox_settings_data_ref_fitttype by comboBox_settings_data_ref_tempmode value (const/var)
        if self.settings.get('comboBox_settings_data_ref_tempmode', None) == 'const': #constant temp experiment
            self.ui.comboBox_settings_data_ref_fitttype.hide()
        elif self.settings.get('comboBox_settings_data_ref_tempmode', None) == 'var': #variable temp experiment
            self.ui.comboBox_settings_data_ref_fitttype.setVisible(True)

        # save to data_saver
        sender_name = self.sender().objectName()
        if 'cryst' in sender_name:
            self.data_saver.exp_ref['mode']['cryst'] = value
        elif 'temp' in sender_name:
            self.data_saver.exp_ref['mode']['temp'] = value
        elif 'fit' in sender_name:
            self.data_saver.exp_ref['mode']['fit'] = value


    def set_stackedwidget_index(self, stwgt, idx=[], diret=[]):
        '''
        chenge the index of stwgt to given idx (if not [])
        or to the given direction (if diret not [])
          diret=1: index += 1;
          diret=-1: index +=-1
        '''
        # logger.info(self) 
        if idx: # if index is not []
            stwgt.setCurrentIndex(idx) # set index to idx
        elif diret: # if diret is not []
            count = stwgt.count()  # get total pages
            current_index = stwgt.currentIndex()  # get current index
            stwgt.setCurrentIndex((current_index + diret) % count) # increase or decrease index by diret

    # update widget values in settings dict, only works with elements out of settings_settings

    def update_widget(self, signal):
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        logger.info('update: %s %s', self.sender().objectName(), signal) 
        logger.info('type: %s', type(signal)) 
        if isinstance(self.sender(), QLineEdit):
            # self.settings[self.sender().objectName()] = signal
            if UIModules.isint(signal): # is int
                self.settings[self.sender().objectName()] = int(signal)
            elif UIModules.isfloat(signal): # is float
                self.settings[self.sender().objectName()] = float(signal)
            else:
                self.settings[self.sender().objectName()] = signal

            # try:
            #     self.settings[self.sender().objectName()] = float(signal)
            # except:
            #     self.settings[self.sender().objectName()] = signal
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.settings[self.sender().objectName()] = signal
            # self.settings[self.sender().objectName()] = not self.settings[self.sender().objectName()]
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.settings[self.sender().objectName()] = signal
            # self.settings[self.sender().objectName()] = not self.settings[self.sender().objectName()]
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.settings[self.sender().objectName()] = value
            logger.info(self.settings[self.sender().objectName()]) 
        # if the sender of the signal isA QSpinBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), (QSpinBox, QDoubleSpinBox)):
            self.settings[self.sender().objectName()] = signal
        elif isinstance(self.sender(), QTabWidget):
            self.settings[self.sender().objectName()] = signal # index
        elif isinstance(self.sender(), QPlainTextEdit):
            self.settings[self.sender().objectName()] = signal # index
    

    def update_harmwidget(self, signal):
        '''
        update value changes of widgets in treeWidget_settings_settings_harmtree
        to self.settings
        except lineEdit_harmstart & lineEdit_harmend
        '''
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        logger.info('update: %s', signal) 
        harm = self.settings_harm

        if isinstance(self.sender(), QLineEdit):
                try:
                    self.save_harmdata(self.sender().objectName(), float(signal), harm=harm)
                except:
                    self.save_harmdata(self.sender().objectName(), 0, harm=harm)
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.save_harmdata(self.sender().objectName(), signal, harm=harm)
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.save_harmdata(self.sender().objectName(), signal, harm=harm)
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.save_harmdata(self.sender().objectName(), value, harm=harm)
        # if the sender of the signal isA QSpinBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), (QSpinBox, QDoubleSpinBox)):
            self.save_harmdata(self.sender().objectName(), signal, harm=harm)

        # And we need to update harmdata and freq_span to peak_tracker
        self.peak_tracker.update_input(self.settings_chn['name'], harm, harmdata=self.settings['harmdata'], freq_span=self.settings['freq_span'], fGB=None)


    def update_settings_chn(self):
        logger.info('update_settings_chn') 
        if self.sender().objectName() == 'tabWidget_settings_settings_samprefchn': # switched to samp
            idx = self.ui.tabWidget_settings_settings_samprefchn.currentIndex()

            logger.info(idx) 
            logger.info(self.ui.pushButton_manual_refit.isChecked()) 
            logger.info(idx < 2) 
            logger.info(self.ui.pushButton_manual_refit.isChecked() & (idx < 2)) 
            if self.ui.pushButton_manual_refit.isChecked() & (idx < 2): # current idx changed out of refit (2)
                logger.info('samprefchn move out of 2') 
                # disable refit widgets
                self.set_manual_refit_mode(val=False)

            if idx == 0: # swith to samp
                self.settings_chn = {
                    'name': 'samp',
                    'chn': self.settings['comboBox_samp_channel']
                }
            elif idx == 1: # switched to ref
                self.settings_chn = {
                    'name': 'ref',
                    'chn': self.settings['comboBox_ref_channel']
                }
            else: # refit
                logger.info('refit chn') 
                self.settings_chn = {
                    'name': 'refit',
                    'chn': 0, # not available for test
                }

        elif self.sender().objectName() == 'comboBox_ref_channel' or 'comboBox_samp_channel': # define of samp/ref channel(s) changed
            # reset corrresponding ADC
            logger.info(self.settings['comboBox_samp_channel']) 
            logger.info(self.settings['comboBox_ref_channel']) 
            if self.settings_chn['name'] == 'samp':
                self.settings_chn['chn'] = self.settings['comboBox_samp_channel']
            elif self.settings_chn['name'] == 'ref':
                self.settings_chn['chn'] = self.settings['comboBox_ref_channel']
            logger.info(self.settings_chn) 


        # update treeWidget_settings_settings_harmtree
        self.update_harmonic_tab()


    def get_chn_by_name(self, name):
        '''
        get chn (str) by given name (str: 'samp' or 'ref')
        '''
        if name == 'samp':
            return  self.settings['comboBox_samp_channel']
        elif name == 'ref':
            return  self.settings['comboBox_ref_channel']


    def update_harmonic_tab(self):
        #logger.info("update_harmonic_tab was called") 
        harm = str(2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1)
        self.settings_harm = harm

        self.update_frequencies() #update frequency dispaly by harm

        logger.info(self.settings['harmdata'][self.settings_chn['name']][harm]) 

        # update lineEdit_scan_harmsteps
        self.ui.lineEdit_scan_harmsteps.setText(
            str(self.get_harmdata('lineEdit_scan_harmsteps', harm=harm))
        )
        self.load_comboBox(self.ui.comboBox_tracking_method, harm=harm)
        self.load_comboBox(self.ui.comboBox_tracking_condition, harm=harm)

        # update checkBox_harmfit
        self.ui.checkBox_harmfit.setChecked(
            self.get_harmdata('checkBox_harmfit', harm=harm)
        )

        # update spinBox_harmfitfactor
        self.ui.spinBox_harmfitfactor.setValue(
            self.get_harmdata('spinBox_harmfitfactor', harm=harm)
        )

        # update spinBox_peaks_num
        self.ui.spinBox_peaks_num.setValue(
            int(self.get_harmdata('spinBox_peaks_num', harm=harm))
        )

        # update radioButton_peaks_num_max
        self.ui.radioButton_peaks_num_max.setChecked(
            self.get_harmdata('radioButton_peaks_num_max', harm=harm)
        )

        # update radioButton_peaks_num_fixed
        self.ui.radioButton_peaks_num_fixed.setChecked(
            self.get_harmdata('radioButton_peaks_num_fixed', harm=harm)
        )

        # update radioButton_peaks_policy_minf
        self.ui.radioButton_peaks_policy_minf.setChecked(
            self.get_harmdata('radioButton_peaks_policy_minf', harm=harm)
        )

        # update radioButton_peaks_policy_maxamp
        self.ui.radioButton_peaks_policy_maxamp.setChecked(
            self.get_harmdata('radioButton_peaks_policy_maxamp', harm=harm)
        )

        # update checkBox_settings_settings_harmlockphase
        self.ui.checkBox_settings_settings_harmlockphase.setChecked(
            self.get_harmdata('checkBox_settings_settings_harmlockphase', harm=harm)
        )
        # update doubleSpinBox_settings_settings_harmlockphase
        self.ui.doubleSpinBox_settings_settings_harmlockphase.setValue(
            self.get_harmdata('doubleSpinBox_settings_settings_harmlockphase', harm=harm)
        )
        # update spinBox_peaks_policy_peakidx
        self.ui.spinBox_peaks_policy_peakidx.setValue(
            self.get_harmdata('spinBox_peaks_policy_peakidx', harm=harm)
        )

        # update lineEdit_peaks_threshold
        self.ui.lineEdit_peaks_threshold.setText(
            str(self.get_harmdata('lineEdit_peaks_threshold', harm=harm))
        )

        # update lineEdit_peaks_prominence
        self.ui.lineEdit_peaks_prominence.setText(
            str(self.get_harmdata('lineEdit_peaks_prominence', harm=harm))
        )


    def get_harmdata(self, objname, harm=None, chn_name=None):
        '''
        get data with given objname in
        treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        if harm is None: # use harmonic displayed in UI
            harm = self.settings_harm
        if chn_name is None:
            chn_name = self.settings_chn['name']

        try:
            return self.settings['harmdata'][chn_name][str(harm)][objname]
        except:
            self.settings['harmdata'][chn_name][str(harm)][objname] = harm_tree_default[objname]
            logger.warning('%s is not found!\nUse default data', objname)
            return self.settings['harmdata'][chn_name][str(harm)][objname]


    def save_harmdata(self, objname, val, harm=None, chn_name=None):
        '''
        save data with given objname in
        treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        if harm is None: # use harmonic displayed in UI
            harm = self.settings_harm
        else: # use given harmonic. It is useful for mpl_sp<n> getting params
            pass
        if chn_name is None:
            chn_name = self.settings_chn['name']

        try:
            self.settings['harmdata'][chn_name][harm][objname] = val
        except:
            logger.warning('%s is not found!', objname)


    def update_base_freq(self, base_freq_index):
        self.settings['comboBox_base_frequency'] = self.ui.comboBox_base_frequency.itemData(base_freq_index) # in MHz
        logger.info(self.settings['comboBox_base_frequency']) 
        # update freq_range
        self.update_freq_range()
        # check freq_span
        self.check_freq_spans()
        # update freqrency display
        self.update_frequencies()
        # update statusbar
        self.statusbar_f0bw_update()


    def update_range(self, range_index):
        self.settings['comboBox_range'] = self.ui.comboBox_range.itemData(range_index) # in MHz
        logger.info(self.settings['comboBox_range']) 
        # update freq_range
        self.update_freq_range()
        # check freq_span
        self.check_freq_spans()
        # update freqrency display
        self.update_frequencies()
        # update statusbar
        self.statusbar_f0bw_update()


    def statusbar_f0bw_update(self):
        fbase = self.settings['comboBox_base_frequency']
        BW = self.settings['comboBox_range']
        self.ui.label_status_f0RNG.setText('{}\u00B1{} MHz'.format(fbase, BW))
        self.ui.label_status_f0RNG.setToolTip('base frequency = {} MHz; range = \u00B1{} MHz'.format(fbase, BW))


    def update_freq_range(self):
        '''
        update settings['freq_range'] (freq range allowed for scan)
        '''
        fbase = float(self.settings['comboBox_base_frequency']) * 1e6 # in Hz
        BW = float(self.settings['comboBox_range']) * 1e6 # in Hz
        freq_range = {}
        for i in self.all_harm_list():
            freq_range[str(i)] = [i*fbase-BW, i*fbase+BW]
        self.settings['freq_range'] = freq_range
        logger.info('freq_range', self.settings['freq_range']) 


    def get_freq_span(self, harm=None, chn_name=None):
        '''
        return freq_span of given harm and chn_name
        if harm and chn_name not given, use self.settings
        '''
        if harm is None:
            harm = self.settings_harm
        if chn_name is None:
            chn_name = self.settings_chn['name']

        # logger.info('get_freq_span') 
        # logger.info('harm', harm) 
        # logger.info('chn_name', chn_name) 
        # logger.info('freq_span', self.settings['freq_span'][chn_name][harm]) 

        return self.settings['freq_span'][chn_name][harm]


    def set_freq_span(self, span, harm=None, chn_name=None):
        '''
        set freq_span of given harm and chn_name
        if harm and chn_name not given, use self.settings
        span: ndarray of [f1, f2]
        '''
        if harm is None:
            harm = self.settings_harm
        if chn_name is None:
            chn_name = self.settings_chn['name']

        if any(np.isnan(span)): # failed to track peak
            print('failed to set freq_span')
            return # return w/o changing the previous span
        else:
            self.settings['freq_span'][chn_name][harm] = span


    def check_freq_spans(self):
        '''
        check if settings['freq_span'] (freq span for each harmonic) values in the allowed range self.settings['freq_range']
        '''
        if 'freq_span' in self.settings and self.settings['freq_span']:  # if self.settings['freq_span'] exist
            logger.info('##################\n%s', self.settings['freq_span']) 
            freq_span = {'samp': {}, 'ref': {}}
            for harm in self.all_harm_list(as_str=True):
                if harm in self.settings['freq_span']['samp']:
                    freq_span['samp'][harm] = self.span_check(harm, *self.settings['freq_span']['samp'][harm])
                    freq_span['ref'][harm] = self.span_check(harm, *self.settings['freq_span']['ref'][harm])
                else:
                    freq_span['samp'][harm] = self.settings['freq_range'][harm]
                    freq_span['ref'][harm] = self.settings['freq_range'][harm]

            self.settings['freq_span'] = freq_span
        else: # if self.settings['freq_span'] does not exist or is empty
            if 'freq_range' not in self.settings: # check if
                self.update_freq_range() # initiate self.settings['freq_range']
            # set 'freq_span' == 'freq_range
            self.settings['freq_span']['samp'] = self.settings['freq_range']
            self.settings['freq_span']['ref'] = self.settings['freq_range']


    def update_frequencies(self):

        # get display mode (startstop or centerspan)
        disp_mode = self.settings['comboBox_settings_control_dispmode']
        # update lineEdit_startf<n> & lineEdit_endf<n>
        for harm in self.all_harm_list(as_str=True):
            # f1, f2 = self.settings['freq_span']['samp'][harm] # in Hz
            f1, f2 = self.get_freq_span(harm=harm, chn_name='samp') # in Hz
            # f1r, f2r = self.settings['freq_span']['ref'][harm] # in Hz
            f1r, f2r = self.get_freq_span(harm=harm, chn_name='ref') # in Hz
            if disp_mode == 'centerspan':
                # convert f1, f2 from start/stop to center/span
                f1, f2 = UIModules.converter_startstop_to_centerspan(f1, f2)
                f1r, f2r = UIModules.converter_startstop_to_centerspan(f1r, f2r)
            getattr(self.ui, 'lineEdit_startf' + harm).setText(UIModules.num2str(f1, precision=1)) # display as Hz
            getattr(self.ui, 'lineEdit_endf' + harm).setText(UIModules.num2str(f2, precision=1)) # display as Hz
            getattr(self.ui, 'lineEdit_startf' + harm + '_r').setText(UIModules.num2str(f1r, precision=1)) # display as Hz
            getattr(self.ui, 'lineEdit_endf' + harm + '_r').setText(UIModules.num2str(f2r, precision=1)) # display as Hz

        # update start/end in treeWidget_settings_settings_harmtree
        harm = self.settings_harm
        logger.info(harm) 
        f1, f2 = self.get_freq_span()
        # f1, f2 = self.settings['freq_span'][self.settings_chn['name']][harm]
        # Set Start
        self.ui.lineEdit_scan_harmstart.setText(
            UIModules.num2str(f1, precision=1)
        )
        # set End
        self.ui.lineEdit_scan_harmend.setText(
            UIModules.num2str(f2, precision=1)
        )


    def update_freq_display_mode(self, signal):
        ''' update frequency dispaly in settings_control '''
        logger.info(signal) 
        disp_mode = self.settings['comboBox_settings_control_dispmode']
        # disp_mode = self.ui.comboBox_settings_control_dispmode.itemData(signal)

        # set label_settings_control_label1 & label_settings_control_label2
        if disp_mode == 'startstop':
            self.ui.label_settings_control_label1.setText('Start')
            self.ui.label_settings_control_label2.setText('End')
            self.ui.label_settings_control_label1_r.setText('Start')
            self.ui.label_settings_control_label2_r.setText('End')
        elif disp_mode == 'centerspan':
            self.ui.label_settings_control_label1.setText('Center')
            self.ui.label_settings_control_label2.setText('Span')
            self.ui.label_settings_control_label1_r.setText('Center')
            self.ui.label_settings_control_label2_r.setText('Span')

        self.update_frequencies()


    def on_editingfinished_harm_freq(self):
        '''
        update frequency when lineEdit_scan_harmstart or  lineEdit_scan_harmend edited
        '''
        # logger.info(self.sender().objectName()) 
        harmstart = float(self.ui.lineEdit_scan_harmstart.text()) # in Hz
        harmend = float(self.ui.lineEdit_scan_harmend.text()) # in Hz
        harm=self.settings_harm
        logger.info('%s %s %s', harm, harmstart, harmend) 
        f1, f2 = self.span_check(harm=harm, f1=harmstart, f2=harmend)
        logger.info((f1, f2)) 
        self.set_freq_span([f1, f2])
        # self.settings['freq_span'][harm] = [harmstart, harmend] # in Hz
        # self.check_freq_spans()
        self.update_frequencies()


    def update_spanmethod(self, fitmethod_index):
        #NOTUSING
        value = self.ui.comboBox_tracking_method.itemData(fitmethod_index)
        self.save_harmdata('comboBox_tracking_method', value, harm=self.settings_harm)


    def update_spantrack(self, trackmethod_index):
        #NOTUSING
        value = self.ui.comboBox_tracking_condition.itemData(trackmethod_index)
        self.save_harmdata('comboBox_tracking_condition', value, harm=self.settings_harm)


    def setvisible_samprefwidgets(self):
        '''
        set the visibility of sample and reference related widget
        '''
        samp_value = (self.settings['comboBox_samp_channel'] != 'none') and self.settings['checkBox_activechn_samp']
        ref_value = (self.settings['comboBox_ref_channel'] != 'none')and self.settings['checkBox_activechn_ref']
        
        logger.info(samp_value) 
        logger.info(ref_value) 
        self.setvisible_sampwidgets(value=samp_value)
        self.setvisible_refwidgets(value=ref_value)
        # set tabWidget_settings_settings_samprefchn
        if samp_value and ref_value: # both samp and ref channels are selected
            # self.ui.tabWidget_settings_settings_samprefchn.setVisible(True)
            # self.ui.tabWidget_settings_settings_samprefchn.setEnabled(True)
            self.ui.checkBox_activechn_samp.setChecked(True)
            self.ui.checkBox_activechn_ref.setChecked(True)
            pass
        elif not samp_value and not ref_value: # neither of samp or ref channel is selected
            # self.ui.tabWidget_settings_settings_samprefchn.setVisible(False)
            self.ui.checkBox_activechn_samp.setChecked(False)
            self.ui.checkBox_activechn_ref.setChecked(False)
            pass
        else: # one of samp and ref channels is selected
            # self.ui.tabWidget_settings_settings_samprefchn.setVisible(True)
            # self.ui.tabWidget_settings_settings_samprefchn.setEnabled(False)

            # set one as current and check it, uncheck the other
            if samp_value:
                self.ui.tabWidget_settings_settings_samprefchn.setCurrentIndex(0)
                # self.ui.checkBox_activechn_samp.setChecked(True)
                self.ui.checkBox_activechn_ref.setChecked(False)
            else:
                self.ui.tabWidget_settings_settings_samprefchn.setCurrentIndex(1)
                self.ui.checkBox_activechn_samp.setChecked(False)
                # self.ui.checkBox_activechn_ref.setChecked(True)


    def statusbar_signal_chn_update(self):
        '''
        set the pushButton_status_signal_ch
        '''
        samp_value = (self.settings['comboBox_samp_channel'] != 'none') and self.settings['checkBox_activechn_samp']
        ref_value = (self.settings['comboBox_ref_channel'] != 'none')and self.settings['checkBox_activechn_ref']

        logger.info(samp_value) 
        logger.info(ref_value) 

        # set tabWidget_settings_settings_samprefchn
        if samp_value and ref_value: # both samp and ref  (value is '1' or '2')channels are selected
            self.ui.pushButton_status_signal_ch.setIcon(QIcon(":/icon/rc/signal_ch_sr.svg"))
        elif not samp_value and not ref_value: # neither of samp or ref channel is selected
            self.ui.pushButton_status_signal_ch.setIcon(QIcon(":/icon/rc/signal_ch_off.svg"))
        elif samp_value and not ref_value: # only samp selected
            self.ui.pushButton_status_signal_ch.setIcon(QIcon(":/icon/rc/signal_ch_s.svg"))
        elif not samp_value and ref_value: # only ref selected
            self.ui.pushButton_status_signal_ch.setIcon(QIcon(":/icon/rc/signal_ch_r.svg"))
        else: # somthing wrong
            self.ui.pushButton_status_signal_ch.setIcon(QIcon(":/icon/rc/signal_ch_off.svg"))






    def setvisible_sampwidgets(self, value=True):
        '''
        set the visibility of sample related widget
        '''
        self.ui.label_settings_control_samp.setVisible(value)
        self.ui.label_settings_control_label1.setVisible(value)
        self.ui.label_settings_control_label2.setVisible(value)
        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'lineEdit_startf' + harm).setVisible(value)
            getattr(self.ui, 'lineEdit_endf' + harm).setVisible(value)


    def setvisible_refwidgets(self, value=False):
        '''
        set the visibility of reference related widget
        '''
        self.ui.label_settings_control_ref.setVisible(value)
        self.ui.label_settings_control_label1_r.setVisible(value)
        self.ui.label_settings_control_label2_r.setVisible(value)
        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'lineEdit_startf' + harm + '_r').setVisible(value)
            getattr(self.ui, 'lineEdit_endf' + harm + '_r').setVisible(value)


    def check_checked_activechn(self):

        if config_default['activechn_num'] == 1:
            sender_name = self.sender().objectName()
            logger.info(sender_name) 

            # this part is for checking the channels
            if sender_name.startswith('checkBox_') and self.settings[sender_name]: # checkBox_ is checked
                if sender_name == 'checkBox_activechn_samp':
                    self.ui.checkBox_activechn_ref.setChecked(False)
                elif sender_name == 'checkBox_activechn_ref':
                    self.ui.checkBox_activechn_samp.setChecked(False)


    def update_vnachannel(self):
        '''
        update vna channels (sample and reference)
        if ref == sample: sender = 'none'
        '''
        sender_name = self.sender().objectName()
        logger.info(sender_name) 

        samp_channel = self.settings['comboBox_samp_channel']
        ref_channel = self.settings['comboBox_ref_channel']

        # this park sets the other channel to none if conflict found
        # if ref_channel == samp_channel:
        #     if 'samp' in sender_name:
        #         self.settings['comboBox_ref_channel'] = 'none'
        #         self.load_comboBox(self.ui.comboBox_ref_channel)
        #     elif 'ref' in sender_name:
        #         self.settings['comboBox_samp_channel'] = 'none'
        #         self.load_comboBox(self.ui.comboBox_samp_channel)
        #     else:
        #         pass


        # set visibility of samp & ref related widgets 
        self.setvisible_samprefwidgets()

        # set statusbar icon pushButton_status_signal_ch
        self.statusbar_signal_chn_update()


    def update_tempsensor(self, signal):
        # NOTUSING
        logger.info("update_tempsensor was called") 
        self.settings['checkBox_settings_temp_sensor'] = signal
        # self.settings['checkBox_settings_temp_sensor'] = not self.settings['checkBox_settings_temp_sensor']


    def update_tempdevice(self, tempdevice_index):
        value = self.ui.comboBox_tempdevice.itemData(tempdevice_index)
        self.settings['comboBox_tempdevice'] = value
        # update display on label_temp_devthrmcpl
        self.set_label_temp_devthrmcpl()


    def update_thrmcpltype(self, thrmcpltype_index):
        value = self.ui.comboBox_thrmcpltype.itemData(thrmcpltype_index)
        self.settings['comboBox_thrmcpltype'] = value
        # update display on label_temp_devthrmcpl
        self.set_label_temp_devthrmcpl()


    def set_label_temp_devthrmcpl(self):
        '''
        display current selection of temp_sensor & thrmcpl
        in label_temp_devthrmcpl
        '''
        logger.info('%s %s', self.settings['comboBox_tempdevice'], self.settings['comboBox_thrmcpltype']) 
        self.ui.label_temp_devthrmcpl.setText(
            'Dev/Thermocouple: {}/{}'.format(
                self.settings['comboBox_tempdevice'],
                self.settings['comboBox_thrmcpltype']
            )
        )


    def update_timeunit(self, timeunit_index):
        value = self.ui.comboBox_timeunit.itemData(timeunit_index)
        self.settings['comboBox_timeunit'] = value
        #TODO update plt1 and plt2


    def update_tempunit(self, tempunit_index):
        value = self.ui.comboBox_tempunit.itemData(tempunit_index)
        self.settings['comboBox_tempunit'] = value
        #TODO update plt1 and plt2


    def update_timescale(self, timescale_index):
        value = self.ui.comboBox_xscale.itemData(timescale_index)
        self.settings['comboBox_xscale'] = value
        #TODO update plt1 and plt2


    def update_yscale(self, yscale_index):
        value = self.ui.comboBox_yscale.itemData(yscale_index)
        self.settings['comboBox_yscale'] = value
        #TODO update plt1 and plt2


    def update_linkx(self):
       self.settings['checkBox_linkx'] = not self.settings['checkBox_linkx']
        # TODO update plt1 and plt2


    def load_comboBox(self, comboBox, val=None, harm=None, mech_chn=None):
        '''
        load combobox value from self.settings[comboBox.objectName()]
        if harm == None
            set the value of combox from self.settings[comboBox]
        if harm = int
            the combobox is in harmwidget
        '''
        comboBoxName = comboBox.objectName()
        # if config_default[opts]:
        #     for key in config_default[opts].keys():
        #         # TODO look for value from itemdata and loop use for in combox.count()
        #         if harm is None: # not embeded in subdict
        #             if key == self.settings[comboBoxName]:
        #                 comboBox.setCurrentIndex(comboBox.findData(key))
        #                 break
        #         else:
        #             if key == self.get_harmdata(comboBoxName, harm):
        #                 comboBox.setCurrentIndex(comboBox.findData(key))
        #                 break
        if (harm is None) and (mech_chn is None): # normal combobox
            if val is None:
                val = self.settings.get(comboBoxName, None)
            set_ind = comboBox.findData(val)
            logger.info('set_ind: %s', set_ind) 
            if set_ind != -1: # key in list
                comboBox.setCurrentIndex(set_ind)
        elif (harm is not None) and (mech_chn is None): # in harmdata
            logger.info('harmdata') 
            if val is None:
                val = self.get_harmdata(comboBoxName, harm)
            set_ind = comboBox.findData(val)
            if set_ind != -1: # key in list
                logger.info('%s is found', comboBoxName) 
                comboBox.setCurrentIndex(set_ind)
        elif (harm is None) and (mech_chn is not None):
            logger.info('mechchndata') 
            if val is None:
                val = self.get_mechchndata(comboBoxName, mech_chn)
            set_ind = comboBox.findData(val)
            if set_ind != -1: # key in list
                logger.info('%s is found', comboBoxName) 
                comboBox.setCurrentIndex(set_ind)


    def build_comboBox(self, combobox, opts):
        '''
        build comboBox by addItem from opts in config_default[opts]
        '''
        for key, val in config_default[opts].items():
            combobox.addItem(val, userData=key)


    def create_combobox(self, name, contents, box_width, row_text='', parent=''):
        '''
        this function create a combobox object with its name = name, items = contents. and  set it't width.
        And move it to row[0] = row_text in parent
        '''
        # create a combobox object
        setattr(self.ui, name, QComboBox())
        # get the object
        obj_box = getattr(self.ui, name)
        # set objectName
        obj_box.setObjectName(name)
        # set its size adjust policy
        obj_box.SizeAdjustPolicy(QComboBox.AdjustToContents)
        # add items from contents
        if isinstance(contents, list): # if given a list, add only the text
            for val in contents:
                obj_box.addItem(val)
        elif isinstance(contents, dict): # if given a dict, add the text (val) and userData (key)
            for key, val in contents.items():
                obj_box.addItem(val, key)

        # insert to the row of row_text if row_text and parent_name are not empty
        if (row_text and parent):
            self.move_to_col(obj_box, parent, row_text, box_width)


    def update_guichecks(self, checkBox, name_in_settings):
        #NOTUSING
        logger.info("update_guichecks was called") 
        checkBox.setChecked(self.get_harmdata(name_in_settings, harm=self.settings_harm))


    # debug func
    def log_update(self):
        #NOTUSING
        with open('settings.json', 'w') as f:
            line = json.dumps(dict(self.settings), indent=4) + "\n"
            f.write(line)


    def load_normal_widgets(self, name_list):
        '''
        load those widgets don't require special setup
        find the type by widget's name
        '''
        for name in name_list:
            obj = getattr(self.ui, name)
            if name.startswith('lineEdit_'):
                obj.setText(self.settings[name])
            elif name.startswith(('checkBox_', 'radioButton_')):
                obj.setChecked(self.settings.get(name, False))
            elif name.startswith('dateTimeEdit_'):
                obj.setDateTime(datetime.datetime.strptime(self.settings[name], config_default['time_str_format']))
            elif name.startswith(('spinBox_', 'doubleSpinBox_')):
                obj.setValue(self.settings[name])
            elif name.startswith('plainTextEdit_'):
                obj.setPlainText(self.settings[name])
            elif name.startswith('comboBox_'):
                self.load_comboBox(obj)
            elif name.startswith('groupBox_'):
                # NOTE here is only for checked
                obj.setChecked(self.settings[name])


    def load_settings(self):
        '''
        setup the UI with the value from self.settings
        '''

        # load default crystal settings

        # create self.settings['freq_range'].
        # this has to be initated before any
        self.update_freq_range()
        # update self.settings['freq_span']
        self.check_freq_spans()

        ## set default appearence
        # set window title
        self.setWindowTitle(_version.__projectname__ + ' Version ' + _version.__version__ )
        # set window size
        if not self.isMaximized(): # resize window to default if is not maxized
            self.resize(*config_default['window_size'])

        # set deflault displaying of tab_settings
        self.ui.tabWidget_settings.setCurrentIndex(0)
        # set deflault displaying of stackedWidget_spetratop
        self.ui.stackedWidget_spetratop.setCurrentIndex(0)
        # set deflault displaying of stackedWidget_spectra
        self.ui.stackedWidget_spectra.setCurrentIndex(0)
        # set deflault displaying of stackedWidget_data
        self.ui.stackedWidget_data.setCurrentIndex(0)
        # set deflault displaying of tabWidget_settings_settings_harm
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)
        # set deflault displaying of tabWidget_mechanics_chn
        self.ui.tabWidget_mechanics_chn.setCurrentIndex(0)
        # set actived harmonic tab
        # self.settings_harm = 1 #TODO
        # set active_chn
        self.ui.tabWidget_settings_settings_samprefchn.setCurrentIndex(0)
        # set active mech calc method
        self.ui.stackedWidget_settings_mechanics_modeswitch.setCurrentIndex(config_default['mechanics_modeswitch'])
        # set progressbar
        self.set_progressbar(val=0, text='')

        # set lineEdit_datafilestr
        self.set_filename()



        ## following data is read from self.settings
 
        # load display_mode
        self.load_normal_widgets(['comboBox_settings_control_dispmode'])

        # load harm state
        for harm in self.all_harm_list(as_str=True):
            # settings/control/Harmonics
            # self.load_normal_widgets([
            #     'checkBox_harm' + harm,
            #     'checkBox_tree_harm' + harm,
            # ])

            getattr(self.ui, 'checkBox_harm' + harm).setChecked(self.settings.get('checkBox_harm' + harm, False))
            getattr(self.ui, 'checkBox_tree_harm' + harm).setChecked(self.settings.get('checkBox_harm' + harm, False))

        # store t0_shift in a temp variable to prevent it been overwritten while loading reference time
        logger.info(self.settings.keys()) 
        if 'dateTimeEdit_settings_data_t0shifted' in self.settings.keys(): # there is t0_shifted
            temp = self.settings['dateTimeEdit_settings_data_t0shifted']
        else:
            temp = ''

        # load reference time
        if 'dateTimeEdit_reftime' in self.settings.keys(): # reference time has been defined
            logger.info(self.settings['dateTimeEdit_reftime']) 
            logger.info(type(datetime.datetime.strptime(self.settings['dateTimeEdit_reftime'], config_default['time_str_format']))) 
            logger.info(type(datetime.datetime.now())) 
            # exit(0)
            self.load_normal_widgets([
                'dateTimeEdit_reftime',
            ])

        else: # reference time is not defined
            # use current time
            self.reset_reftime()

        # set t0_shifted back
        if temp:
            self.settings['dateTimeEdit_settings_data_t0shifted'] = temp

        self.load_normal_widgets([
            'spinBox_recordinterval', # load default record interval
            'spinBox_refreshresolution', # load default spectra refresh resolution
        ])
        
        # update spinBox_scaninterval
        self.set_recording_time()
        
        self.load_normal_widgets([
            'checkBox_dynamicfit', # load default fitting and display options
            'spinBox_fitfactor', # load default fit factor range
            'checkBox_dynamicfitbyharm', # load default dynamicfitbyharm
            'checkBox_fitfactorbyharm',  # load default fitfactorbyharm
            'plainTextEdit_settings_sampledescription', # plainTextEdit_settings_sampledescription
        ])

        self.load_normal_widgets([
            'checkBox_activechn_samp', 
            'checkBox_activechn_ref', 
        ])

        self.load_normal_widgets([
            'comboBox_settings_settings_analyzer', # load the analyzer selection
            'comboBox_base_frequency', # load this first to create self.settings['freq_range'] & self.settings['freq_span']
            'comboBox_range', 
        ])

        # update statusbar
        self.statusbar_f0bw_update()

        # create self.settings['freq_range'].
        # this has to be initated before any
        self.update_freq_range()
        # update self.settings['freq_span']
        self.check_freq_spans()
        # update frequencies display
        self.update_frequencies()

        self.load_normal_widgets([
            # update crystalcut
            'comboBox_crystalcut', 
            # load default VNA settings
            'comboBox_samp_channel', 
            'comboBox_ref_channel',
        ])

        # show samp & ref related widgets
        self.setvisible_samprefwidgets()
        # set statusbar icon pushButton_status_signal_ch
        self.statusbar_signal_chn_update()

        # set treeWidget_settings_settings_harmtree display
        self.update_harmonic_tab()

        # load default temperature settings
        self.load_normal_widgets([
            'checkBox_settings_temp_sensor',
            'comboBox_tempmodule',
            'comboBox_thrmcpltype',
        ])
        try:
            self.load_comboBox(self.ui.comboBox_tempdevice)
        except:
            logger.warning('No tempdevice found.')
            pass
        # update display on label_temp_devthrmcpl
        self.set_label_temp_devthrmcpl() # this should be after temp_sensor & thrmcpl

        # load default plots settings
        self.load_normal_widgets([
            'comboBox_timeunit',
            'comboBox_tempunit',
            'comboBox_xscale',
            'comboBox_yscale',
            'checkBox_linkx',
        ])

        # set default displaying of spectra show options
        self.load_normal_widgets([
            'radioButton_spectra_showBp',
            'radioButton_spectra_showpolar',
            'checkBox_spectra_showchi',
        ])

        # set data radioButton_data_showall
        self.load_normal_widgets([
            'radioButton_data_showall',
            'radioButton_data_showmarked',
        ])

        self.load_normal_widgets([
            # set default displaying of plot 1 options
            'comboBox_plt1_optsy',
            'comboBox_plt1_optsx',
            # set default displaying of plot 2 options
            'comboBox_plt2_optsy',
            'comboBox_plt2_optsx',
        ])

        # set checkBox_plt<1 and 2>_h<harm>
        for harm in self.all_harm_list(as_str=True):
            self.load_normal_widgets([
                'checkBox_plt1_h' + harm, 
                'checkBox_plt2_h' + harm, 
            ])

        # set radioButton_plt<n>_samp/ref
        self.load_normal_widgets([
            'radioButton_plt1_samp',
            'radioButton_plt1_ref',
            'radioButton_plt2_samp',
            'radioButton_plt2_ref',
        ])

        # load t0_shifted time
        if 'dateTimeEdit_settings_data_t0shifted' in self.settings: # t0_shifted has been defined
            logger.info(self.settings['dateTimeEdit_settings_data_t0shifted']) 
            self.ui.dateTimeEdit_settings_data_t0shifted.setDateTime(datetime.datetime.strptime(self.settings['dateTimeEdit_settings_data_t0shifted'], config_default['time_str_format']))

        else: # t0_shifted is not defined
            # use reference time
            self.ui.dateTimeEdit_settings_data_t0shifted.setDateTime(datetime.datetime.strptime(self.settings['dateTimeEdit_reftime'], config_default['time_str_format']))

        # set widgets to display the channel reference setup
        # the value will be load from data_saver
        self.update_refsource()

        # update mpl_plt<n> at the end
        self.update_mpl_plt12()

        # settings_mechanics
        for harm in self.all_harm_list(as_str=True):
            self.load_normal_widgets([
                'checkBox_nhplot' + harm,
            ])

        self.load_normal_widgets([
            'checkBox_settings_mech_liveupdate',
            'spinBox_settings_mechanics_nhcalc_n1',
            'spinBox_settings_mechanics_nhcalc_n2',
            'spinBox_settings_mechanics_nhcalc_n3',
            'checkBox_settings_mechanics_witherror',
            'comboBox_settings_mechanics_selectmodel',
            'comboBox_settings_mechanics_calctype',
            'doubleSpinBox_settings_mechanics_bulklimit',
            'radioButton_settings_mech_refto_air',
            'radioButton_settings_mech_refto_overlayer',
            # contour
            'comboBox_settings_mechanics_contourdata',
            'comboBox_settings_mechanics_contourtype',
            'comboBox_settings_mechanics_contourcmap',
            'groupBox_settings_mechanics_contour',
        ])

        # spinBox_mech_expertmode_layernum
        self.ui.spinBox_mech_expertmode_layernum.setValue(self.get_mechchndata('spinBox_mech_expertmode_layernum'))
        
        # load film layers data 
        self.load_film_layers_widgets()

        # tableWidget_settings_mechanics_contoursettings
        self.set_init_table_value(
            'tableWidget_settings_mechanics_contoursettings',
            'mech_contour_lim_tab_vheaders',
            'mech_contour_lim_tab_hheaders',
            'contour_plot_lim_tab',
        )

        # plot contours
        self.make_contours()

        ## end of load_settings


    def update_refsource(self):
        '''
        update widgets related to reference source
        '''
        logger.info('ref_channel_opts') 
        logger.info(self.settings['comboBox_settings_data_samprefsource']) 
        self.load_comboBox(self.ui.comboBox_settings_data_samprefsource)
        self.load_comboBox(self.ui.comboBox_settings_data_refrefsource)
        self.ui.lineEdit_settings_data_sampidx.setText(str(self.settings['lineEdit_settings_data_sampidx']))
        self.ui.lineEdit_settings_data_refidx.setText(str(self.settings['lineEdit_settings_data_refidx']))
        self.ui.lineEdit_settings_data_samprefidx.setText(str(self.settings['lineEdit_settings_data_samprefidx']))
        self.ui.lineEdit_settings_data_refrefidx.setText(str(self.settings['lineEdit_settings_data_refrefidx']))

        # temp ref
        self.load_comboBox(self.ui.comboBox_settings_data_ref_crystmode)
        self.load_comboBox(self.ui.comboBox_settings_data_ref_tempmode)
        self.load_comboBox(self.ui.comboBox_settings_data_ref_fitttype)


    def load_refsource(self):
        '''
        update widgets related to reference source from data_saver
        '''
        logger.info('ref_channel_opts') 
        self.settings['comboBox_settings_data_samprefsource'] = self.data_saver.exp_ref['samp_ref'][0]
        self.settings['comboBox_settings_data_refrefsource'] = self.data_saver.exp_ref['ref_ref'][0]

        if len(self.data_saver.exp_ref['samp_ref']) > 2: # ver >= 0.17.0
            self.settings['lineEdit_settings_data_sampidx'] = self.data_saver.exp_ref['samp_ref'][2]
            self.settings['lineEdit_settings_data_refidx'] = self.data_saver.exp_ref['ref_ref'][2]
        else:
            pass # do not change default value

        self.settings['lineEdit_settings_data_samprefidx'] = self.data_saver.exp_ref['samp_ref'][1]
        self.settings['lineEdit_settings_data_refrefidx'] = self.data_saver.exp_ref['ref_ref'][1]

        if 'mode' in self.data_saver.exp_ref:
            self.settings['comboBox_settings_data_ref_crystmode'] = self.data_saver.exp_ref['mode'].get('cryst')
            self.settings['comboBox_settings_data_ref_tempmode'] = self.data_saver.exp_ref['mode'].get('temp')
            self.settings['comboBox_settings_data_ref_fitttype'] = self.data_saver.exp_ref['mode'].get('fit')


    def set_progressbar(self, val=0, text=''):
        '''
        update progressBar_status_interval_time
        '''
        self.ui.progressBar_status_interval_time.setValue(val)
        self.ui.progressBar_status_interval_time.setFormat(text)


    def data_collection(self):
        '''
        data collecting routine
        '''

        self.idle = False
        # self.timer.setSingleShot(True)
        scan_interval = self.settings['spinBox_scaninterval'] * 1000 # in ms

        # update the interval of timer
        self.timer.setInterval(scan_interval)

        # update the bartimer set up
        bar_interval = scan_interval / config_default['progressbar_update_steps']
        if bar_interval < config_default['progressbar_min_interval']: # interval is to small
            bar_interval = config_default['progressbar_min_interval']
        elif bar_interval > config_default['progressbar_min_interval']: # interval is to big
            bar_interval = config_default['progressbar_max_interval']

        logger.info(scan_interval) 
        logger.info(bar_interval) 

        self.bartimer.setInterval(bar_interval)
        self.bartimer.start()

        ## start to read data
        # set channels to collect data
        chn_name_list = []
        logger.info(chn_name_list) 

        # only one channel can be 'none'
        if self.settings['checkBox_activechn_samp']: # sample channel is not selected
            chn_name_list.append('samp')
        if self.settings['checkBox_activechn_ref']: # reference channel is not selected
            chn_name_list.append('ref')

        harm_list = [harm for harm in self.all_harm_list(as_str=True) if self.settings.get('checkBox_harm' + harm, None)] # get all checked harmonic into a list

        logger.info(self.settings['comboBox_samp_channel']) 
        logger.info(self.settings['comboBox_ref_channel']) 
        logger.info(chn_name_list) 
        logger.info(harm_list) 

        f, G, B = {}, {}, {}
        fs = {} # peak centers
        gs = {} # dissipations hwhm
        ps = {} # 
        curr_time = {}
        curr_temp = {}
        marks = [0 for _ in harm_list] # 'samp' and 'ref' chn test the same harmonics
        for chn_name in chn_name_list:
            # scan harmonics (1, 3, 5...)
            f[chn_name], G[chn_name], B[chn_name] = {}, {}, {}
            fs[chn_name] = []
            gs[chn_name] = []
            ps[chn_name] = []
            curr_temp[chn_name] = None

            self.reading = True
            # read time
            curr_time[chn_name] = datetime.datetime.now().strftime(config_default['time_str_format'])
            logger.info(curr_time)

            # read temp if checked
            if self.settings['checkBox_settings_temp_sensor'] == True: # record temperature data
                curr_temp[chn_name] = self.temp_sensor.get_tempC()
                # update status bar
                self.statusbar_temp_update(curr_temp=curr_temp[chn_name])

            with self.vna:
                # data collecting and plot
                for harm in harm_list:
                    # get data
                    logger.info(harm_list) 
                    f[chn_name][harm], G[chn_name][harm], B[chn_name][harm] = self.get_vna_data_no_with(harm=harm, chn_name=chn_name)

                    logger.info('check:') 
                    logger.info(f[chn_name][harm] is None) 
                    logger.info(f[chn_name][harm][0] == f[chn_name][harm][-1]) 
                    if (f[chn_name][harm] is None) or (f[chn_name][harm][0] == f[chn_name][harm][-1]): # vna error
                        print('Analyzer connection error!')
                        # stop test
                        self.idle = True
                        self.ui.pushButton_runstop.setChecked(False)
                        # alert
                        process = self.process_messagebox(
                            text='Failed to connect with analyzer!',
                            message=['Please check the connection and power.'],
                            opts=False,
                            forcepop=True,
                        )
                        return

                    # put f, G, B to peak_tracker for later fitting and/or tracking
                    self.peak_tracker.update_input(chn_name, harm, harmdata=self.settings['harmdata'], freq_span=self.settings['freq_span'], fGB=[f[chn_name][harm], G[chn_name][harm], B[chn_name][harm]])

                    # plot data in sp<harm>
                    if self.settings['radioButton_spectra_showGp']: # checked
                        getattr(self.ui, 'mpl_sp' + str(harm)).update_data({'ln': 'lG', 'x': f[chn_name][harm], 'y': G[chn_name][harm]})
                    elif self.settings['radioButton_spectra_showBp']: # checked
                        getattr(self.ui, 'mpl_sp' + str(harm)).update_data({'ln': 'lG', 'x': f[chn_name][harm], 'y': G[chn_name][harm]}, {'ln': 'lB', 'x': f[chn_name][harm], 'y': B[chn_name][harm]})
                    elif self.settings['radioButton_spectra_showpolar']: # checked
                        getattr(self.ui, 'mpl_sp' + str(harm)).update_data({'ln': 'lP', 'x': G[chn_name][harm], 'y': B[chn_name][harm]})

            # set xticks
            # self.mpl_set_faxis(getattr(self.ui, 'mpl_sp' + str(harm)).ax[0])

            self.reading = False

            # fitting and tracking
            for harm in harm_list:
                if self.get_harmdata('checkBox_harmfit', harm=harm, chn_name=chn_name): # checked to fit

                    fit_result = self.peak_tracker.peak_fit(chn_name, harm, components=False)
                    logger.info(fit_result) 
                    logger.info(fit_result['v_fit']) 
                    # logger.info(fit_result['comp_g']) 

                    # plot fitted data
                    if self.settings['radioButton_spectra_showGp']: # checked
                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lGfit', 'x': f[chn_name][harm], 'y': fit_result['fit_g']})
                    elif self.settings['radioButton_spectra_showBp']: # checked
                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lGfit', 'x': f[chn_name][harm], 'y': fit_result['fit_g']}, {'ln': 'lBfit', 'x': f[chn_name][harm], 'y': fit_result['fit_b']})
                    elif self.settings['radioButton_spectra_showpolar']: # checked
                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lPfit', 'x': fit_result['fit_g'], 'y': fit_result['fit_b']})

                    # update lsp
                    factor_span = self.peak_tracker.get_output(key='factor_span', chn_name=chn_name, harm=harm)
                    if 'g_c' in fit_result['v_fit']: # fitting successed
                        gc_list = [fit_result['v_fit']['g_c']['value']] * 2 # make its len() == 2
                        bc_list = [fit_result['v_fit']['b_c']['value']] * 2 # make its len() == 2
                    else: # fitting failed
                        gc_list = [np.nan, np.nan]
                        bc_list = [np.nan, np.nan]

                    logger.info(factor_span) 
                    logger.info(gc_list) 
                    if self.settings['radioButton_spectra_showGp'] or self.settings['radioButton_spectra_showBp']: # show G or GB

                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lsp', 'x':factor_span, 'y': gc_list})
                    elif self.settings['radioButton_spectra_showpolar']: # polar plot
                        idx = np.where((f[chn_name][harm] >= factor_span[0]) & (f[chn_name][harm] <= factor_span[1]))

                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lsp', 'x':fit_result['fit_g'][idx], 'y': fit_result['fit_b'][idx]})


                    # update srec
                    cen_rec_freq = fit_result['v_fit']['cen_rec']['value']
                    cen_rec_G = self.peak_tracker.get_output(key='gmod', chn_name=chn_name, harm=harm).eval(
                        self.peak_tracker.get_output(key='params', chn_name=chn_name, harm=harm),
                        f=cen_rec_freq
                    )

                    # save data to fs and gs
                    fs[chn_name].append(fit_result['v_fit']['cen_rec']['value']) # fs
                    gs[chn_name].append(fit_result['v_fit']['wid_rec']['value'] ) # gs = half_width
                    ps[chn_name].append(fit_result['v_fit']['amp_rec']['value'] ) # 
                    logger.info(cen_rec_freq) 
                    logger.info(cen_rec_G) 

                    if self.settings['radioButton_spectra_showGp'] or self.settings['radioButton_spectra_showBp']: # show G or GB
                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'srec', 'x': cen_rec_freq, 'y': cen_rec_G})
                    elif self.settings['radioButton_spectra_showpolar']: # polar plot
                        cen_rec_B = self.peak_tracker.get_output(key='bmod', chn_name=chn_name, harm=harm).eval(
                            self.peak_tracker.get_output(key='params', chn_name=chn_name, harm=harm),
                            f=cen_rec_freq
                        )

                        getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'srec', 'x': cen_rec_G, 'y': cen_rec_B})

                    if self.settings['checkBox_spectra_showchi']: # show chi square
                        getattr(self.ui, 'mpl_sp' + harm).update_sp_text_chi(fit_result['v_fit']['chisqr'])
                else: # collect data w/o fitting
                    # save data to fs and gs
                    fs[chn_name].append(np.nan) # fs
                    gs[chn_name].append(np.nan) # gs = half_width
                    ps[chn_name].append(np.nan) # gs = half_width
                    # clear lines
                    getattr(self.ui, 'mpl_sp' + harm).clr_lines(l_list=['lGfit', 'lBfit', 'lPfit', 'lsp', 'srec'])

                ## get tracking data
                # get span from tracking
                span, cen_trk_freq = self.peak_tracker.peak_track(chn_name=chn_name, harm=harm)
                # check span range is in range
                span = self.span_check(harm, *span)
                # save span
                self.set_freq_span(span, harm=harm, chn_name=chn_name)
                # update UI
                self.update_frequencies()

                # update strk
                cen_trk_G = G[chn_name][harm][
                    np.argmin(np.abs(f[chn_name][harm] - cen_trk_freq))
                    ]

                logger.info(cen_trk_freq) 
                logger.info(cen_trk_G) 


                if self.settings['radioButton_spectra_showGp'] or self.settings['radioButton_spectra_showBp']: # show G or GB
                    getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'strk', 'x': cen_trk_freq, 'y': cen_trk_G})
                elif self.settings['radioButton_spectra_showpolar']: # polar plot
                    cen_trk_B = B[chn_name][harm][
                    np.argmin(np.abs(f[chn_name][harm] - cen_trk_freq))
                    ]

                    getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'strk', 'x': cen_trk_G, 'y': cen_trk_B})

                # set xticks
                # self.mpl_set_faxis(getattr(self.ui, 'mpl_sp' + str(harm)).ax[0])

        # Save scan data to file, fitting data in RAM to file
        if self.spectra_refresh_modulus() == 0: # check if to save by intervals
            # if self.counter != 0:
            #     self.counter = 0 # restart counter
            # else:
            #     self.counter += 1 # increase counter
            self.writing = True
            # save raw
            self.data_saver.dynamic_save(chn_name_list, harm_list, t=curr_time, temp=curr_temp, f=f, G=G, B=B, fs=fs, gs=gs, ps=ps, marks=marks)

            # save data (THIS MIGHT MAKE THE PROCESS SLOW)
            self.data_saver.save_data()

            # plot data
            self.update_mpl_plt12()
        else: # data will not be saved (temperarily saved in peak_tracker)
            #TODO we can still plot the data 
            # self.counter += 1 # increase counter
            pass

        if not self.timer.isActive(): # if timer is stopped (test stopped while collecting data)
            # save data
            self.process_saving_when_stop()
            logger.info('data saved while collecting') 

        ## This is for continuous counter
        # increase counter 
        self.counter += 1

        self.idle = True

        self.writing = False

        # display total points collected
        self.set_status_pts()


    def data_refit(self, chn_name, sel_idx_dict, regenerate=False):
        '''
        data refit routine
        sel_idx_dict = {
            'harm': [idx]
        }
        from_raw: if get all (t, temp) information from raw
        '''
        if self.idle == False:
            print('Data collection is running!')
            return

        ## start to read data from data saver
        # set channels to collect data

        logger.info('sel_idx_dict\n%s', sel_idx_dict) 
        # reform dict
        sel_harm_dict = UIModules.idx_dict_to_harm_dict(sel_idx_dict)
        # queue_list = self.data_saver.get_queue_id(chn_name)[sel_harm_dict.keys()] # df
        indeces = sel_harm_dict.keys()
        logger.info('sel_harm_dict\n%s', sel_harm_dict) 

        # reset l['strk'] ("+")
        for harm in self.all_harm_list(as_str=True):
            getattr(self.ui, 'mpl_sp' + harm).clr_lines(l_list=['strk'])

        for idx in indeces:
            # initiate data of queue_id

            queue_id = self.data_saver.get_queue_id(chn_name)[idx]
            # scan harmonics (1, 3, 5...)
            fs = []
            gs = []
            ps = []

            self.reading = True

            # data reading and plot
            harm_list = sel_harm_dict[idx]
            for harm in harm_list: # TODO add poll here
                # get data
                f, G, B = self.data_saver.get_raw(chn_name, queue_id, harm)
                if f is None:
                    logger.info('got None') 
                else:
                    logger.info((len(f), len(G), len(B))) 

                # put f, G, B to peak_tracker for later fitting and/or tracking
                self.peak_tracker.update_input(chn_name, harm, harmdata=self.settings['harmdata'], freq_span=[], fGB=[f, G, B]) # freq_span set to [], since we don't need to track the peak

                # fitting
                fit_result = self.peak_tracker.peak_fit(chn_name, harm, components=False)
                logger.info(fit_result) 
                logger.info(fit_result['v_fit']) 
                # logger.info(fit_result['comp_g']) 

                # save data to fs and gs
                fs.append(fit_result['v_fit']['cen_rec']['value']) # fs
                gs.append(fit_result['v_fit']['wid_rec']['value'] ) # gs = half_width
                ps.append(fit_result['v_fit']['amp_rec']['value'] ) #

                # update lsp
                factor_span = self.peak_tracker.get_output(key='factor_span', chn_name=chn_name, harm=harm)
                gc_list = [fit_result['v_fit']['g_c']['value']] * 2 # make its len() == 2
                bc_list = [fit_result['v_fit']['b_c']['value']] * 2 # make its len() == 2
                logger.info(factor_span) 
                logger.info(gc_list) 

                # update srec
                cen_rec_freq = fit_result['v_fit']['cen_rec']['value']
                cen_rec_G = self.peak_tracker.get_output(key='gmod', chn_name=chn_name, harm=harm).eval(
                    self.peak_tracker.get_output(key='params', chn_name=chn_name, harm=harm),
                    f=cen_rec_freq
                )
                logger.info(cen_rec_freq) 
                logger.info(cen_rec_G) 

                # plot data in sp<harm> and fitting
                if self.settings['radioButton_spectra_showGp']: # checked
                    getattr(self.ui, 'mpl_sp' + harm).update_data(
                        {'ln': 'lG', 'x': f, 'y': G},
                        {'ln': 'lGfit','x': f, 'y': fit_result['fit_g']},
                        {'ln': 'lsp', 'x': factor_span, 'y': gc_list},
                        {'ln': 'srec', 'x': cen_rec_freq, 'y': cen_rec_G}
                    )
                elif self.settings['radioButton_spectra_showBp']: # checked
                    getattr(self.ui, 'mpl_sp' + harm).update_data(
                        {'ln':
                         'lG', 'x': f, 'y': G},
                        {'ln':
                         'lB', 'x': f, 'y': B},
                        {'ln':
                         'lGfit','x': f, 'y': fit_result['fit_g']},
                        {'ln':
                         'lBfit','x': f, 'y': fit_result['fit_b']},
                        {'ln':
                         'lsp', 'x': factor_span, 'y': gc_list},
                        {'ln':
                         'srec', 'x': cen_rec_freq, 'y': cen_rec_G},
                    )
                elif self.settings['radioButton_spectra_showpolar']: # checked
                    idx = np.where(f >= factor_span[0] & f <= factor_span[1])

                    cen_rec_B = self.peak_tracker.get_output(key='bmod', chn_name=chn_name, harm=harm).eval(
                        self.peak_tracker.get_output(key='params', chn_name=chn_name, harm=harm),
                        f=cen_rec_freq
                    )

                    getattr(self.ui, 'mpl_sp' + harm).update_data({'ln': 'lP', 'x': G, 'y': B},
                        {'ln':
                        'lPfit', 'x': fit_result['fit_g'], 'y': fit_result['fit_b']},
                        {'ln':
                        'lsp', 'x': fit_result['fit_g'][idx], 'y': fit_result['fit_b'][idx]},
                        {'ln':
                        'srec', 'x': cen_rec_G, 'y': cen_rec_B},
                    )

                if self.settings['checkBox_spectra_showchi']: # show chi square
                    getattr(self.ui, 'mpl_sp' + harm).update_sp_text_chi(fit_result['v_fit']['chisqr'])

            self.reading = False
            
            if regenerate:
                # get t 
                t = self.data_saver.get_t_str_from_raw(chn_name, queue_id)
                # get temp
                temp = self.data_saver.get_temp_C_from_raw(chn_name, queue_id)
                marks = [0 for _ in harm_list] # 'samp' and 'ref' chn have the same harmonics

                self.data_saver._save_queue_data([chn_name], harm_list, queue_id=queue_id, t={chn_name: t}, temp={chn_name: temp}, fs={chn_name: fs}, gs={chn_name: gs}, ps={chn_name: ps}, marks=marks)
            else:
                # save scan data to file fitting data in data_saver
                self.data_saver.update_refit_data(chn_name, queue_id, harm_list, fs=fs, gs=gs, ps=ps)

            # plot data
            self.update_mpl_plt12()


    def update_table_value(self, tablename, val_item):
        '''
        updat value of self.ui.<tablename> current item to val_item
        val_item: key in self.settings
        '''
        table = getattr(self.ui, tablename)
        item = table.currentItem()
        logger.info(table)
        logger.info(item)
        if item:
            text, r, c = item.text(), item.row(), item.column()
            logger.info('currentItem %s %s %s %s', item, text, r, c)
        else:
            return
            
        val_dict = self.settings.get(val_item, {})

        for ri, (rkey, rval) in enumerate(val_dict.items()):
            if ri == r: # find the same row key
                for ci, (ckey, cval) in enumerate(rval.items()): # find the same row column key
                    if ci == c:
                        if text:
                            val_dict[rkey][ckey] = float(text)
                        else:
                            val_dict[rkey][ckey] = None

                        break # stop the loop
                break
        self.settings[val_item] = val_dict # update to self.settings


    def set_table_value(self, tablename, vh_item, hh_item, val_dict):
        '''
        set value to self.ui.<tablename> from val_dict
        vh_item: name of dict of which stored verticalheader names
        hh_item: name of dict of which stored horizontalheader names
        valdict:
            'r0': {'col0': #, 'col1': #},
            'r1': {'col0': #, 'col1': #},
        '''
        table = getattr(self.ui, tablename)
        table.clearContents()

        for r, (rkey, rval) in enumerate(val_dict.items()):
            if rkey in config_default[vh_item].keys(): # verticla name exist
                for c, (ckey, cval) in enumerate(rval.items()):
                    if ckey in config_default[hh_item].keys():
                        logger.info('table name: %s', table.objectName()) 
                        logger.info('table r c: %s %s', r, c) 
                        tableitem = table.item(r, c)
                        if cval is not None:
                            if tableitem: # tableitem != 0
                                logger.info('item is set') 
                                tableitem.setText(str(cval))
                            else: # tableitem is not set
                                logger.info('item set') 
                                table.setItem(r, c, QTableWidgetItem(str(cval)))


    def set_init_table_value(self, tablename, vh_item, hh_item, val_item):
        '''
        set value to self.ui.<tablename> from val_dict
        vh_item: name of dict of which stored verticalheader names
        hh_item: name of dict of which stored horizontalheader names
        val_item: name of dict in self.settings. 
        '''
        val_dict = self.settings.get(val_item, {})
        if val_dict:
            self.set_table_value(tablename, vh_item, hh_item, val_dict)


    def on_changed_mech_contour_lim_tab(self):
        # tableWidget_settings_mechanics_contoursettings
        # update table in self.settings
        self.update_table_value(
            'tableWidget_settings_mechanics_contoursettings',
            'contour_plot_lim_tab',
        )

        # 
        table = self.ui.tableWidget_settings_mechanics_contoursettings
        item = table.currentItem()
        logger.info(table)
        logger.info(item)
        if item:
            self.make_contours()


    def get_all_checked_harms(self):
        '''
        return all checked harmonics in a list of str
        '''
        return [harm for harm in self.all_harm_list(as_str=True) if
        self.settings.get('checkBox_harm' + harm, None)] # get all checked harmonics


    def update_progressbar(self):
        '''
        update progressBar_status_interval_time
        '''

        # read reainingTime from self.timer
        timer_remain = self.timer.remainingTime() / 1000 # in s
        timer_interval = self.timer.interval() / 1000 # in s
        # logger.info(timer_remain) 
        # logger.info(timer_interval) 
        # logger.info(min(round((1 - timer_remain / timer_interval) * 100), 100)) 

        if self.spectra_refresh_modulus() == 0: # going to save data
            txt = '{:.1f} s'.format(timer_remain)
        else:
            txt = '{:.1f} s + {}'.format(timer_remain, self.settings['spinBox_refreshresolution'] - self.spectra_refresh_modulus())

        self.set_progressbar(
            val=min(round((1 - timer_remain / timer_interval) * 100), 100),
            text=txt
        )


    def spectra_refresh_modulus(self):
        '''
        calculate how many times refresh left before recording
        '''
        return int(self.counter) % int(self.settings['spinBox_refreshresolution'])


    def all_harm_list(self, as_str=False):
        '''
        return a list with all harms as int
        '''
        if as_str:
            return [str(i) for i in range(1, self.settings['max_harmonic']+2, 2)]
            
        else:
            return list(range(1, self.settings['max_harmonic']+2, 2))


#endregion



def run():
    # import sys
    # import traceback
    # import logging
    # import logging.config

    # set logger
    setup_logging()

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    # replace system exception hook
    if QT_VERSION >= 0x50501:
        sys._excepthook = sys.excepthook 
        def exception_hook(exctype, value, traceback):
            # logger.exception('Exception occurred')
            logger.error('Exceptiion error', exc_info=(exctype, value, traceback))
            qFatal('UI error occured.')
            sys._excepthook(exctype, value, traceback) 
            sys.exit(1) 
        sys.excepthook = exception_hook 

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    run()
