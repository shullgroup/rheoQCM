'''
This is the main code of the QCM acquization program

'''

import os
import csv
import importlib
import math
import json
import datetime, time
import numpy as np
import scipy.signal
# import types
from PyQt5.QtCore import pyqtSlot, Qt, QEvent, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QGridLayout, QLineEdit, QCheckBox, QComboBox, QSpinBox, QRadioButton, QMenu, QMessageBox
)
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QValidator, QIntValidator, QDoubleValidator, QRegExpValidator

# packages
from MainWindow import Ui_MainWindow
from UISettings import settings_init, settings_default
from modules import UIModules, MathModules, GBFitting, PeakTracker

from MatplotlibWidget import MatplotlibWidget


if UIModules.system_check() == 'win32': # windows
    import struct
    if struct.calcsize('P') * 8 == 32: # 32-bit version Python
        try:
            # from modules.AccessMyVNA_dummy import AccessMyVNA
            from modules.AccessMyVNA_np import AccessMyVNA
            print(AccessMyVNA)
            # test if MyVNA program is available
            with AccessMyVNA() as vna:
                if vna.Init() == 0: # connection with myVNA is available
                    from modules import tempDevices
        except Exception as e: # no myVNA connected. Analysis only
            print('Failed to import AccessMyVNA module!')
            print(e)
    else: # 64-bit version Python which doesn't work with AccessMyVNA
        # A 32-bit server may help 64-bit Python work with 32-bit dll
        print('Current version of MyVNA does not work with 64-bit Python!\nData analysis only!')
else: # linux or MacOS
    # for test only
    # from modules.AccessMyVNA_dummy import AccessMyVNA
        print('Current version of MyVNA does not work with MacOS and Linux!\nData analysis only!')


class VNATracker:
    def __init__(self):
        self.f =None       # current end frequency span in Hz (ndarray([float, float])
        self.steps = None   # current number of steps (int)
        self.chn = None     # current reflection ADC channel (1 or 2)
        self.avg = None     # average of scans (int)
        self.speed = None   # vna speed set up (int 1 to 10)
        self.instrmode = 0  # instrument mode (0: reflection)
        
        self.setflg = {} # if vna needs to reset (set with reset selections)
        self.setflg.update(self.__dict__) # get all attributes in a dict
        self.setflg.pop('setflg', None) # remove setflg itself
        # print(self.setflg)
    
    def set_check(self, **kwargs):
        for key, val in kwargs.items():
            print(key, val)
            print(type(val))
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


class DataStruct:
    def __init__(self):
        self.samp = None # attribute to save data of sample
        self.ref = None # attribute to save data of reference
        self.samp_settings = {
            'fstar0': None,     # f0
            't0': None,         # reference time
            'tshift': 0,        # reference time shift
            'ref': None,        # reference type
            'raw_file': None    # raw_file information (for searching raw file if needed in the future)
        } # attribute to save f, t, reference define
        self.calc = None # attribute to save the results of mechanics calculation

class RawStruct:
    def __init__(self):
        self.samp = None # attribute to save data of sample
        self.ref = None # attribute to save data of reference
        self.settings = None # attribute to UI settings
        self.ver = {}


class QCMApp(QMainWindow):
    '''
    The settings of the app is stored in a dict by widget names
    '''
    def __init__(self):
        super(QCMApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fileName = ''
        self.fileFlag = False
        self.settings = settings_default # import default settings. It will be initalized latter
        self.peak_tracker = PeakTracker.PeakTracker()
        self.vna_tracker = VNATracker()
  
        # define instrument state variables
        self.vna = None # vna class
        self.tempsensor = None # class for temp sensor
        self.idle = True # if test is running
        self.reading = False # if myVNA/tempsensor is scanning and reading data
        self.writing = False # if UI is saving data

        self.UITab = 0 # 0: Control; 1: Settings;, 2: Data; 3: Mechanics
        self.settings_harm = 1 # active harmonic in Settings Tab
        self.active_harm = 1 # active harmonic in Data Tab
        self.active_chn = {'name': 'samp', 'chn': 1} # active channel 'samp' or 'ref'
        #### initialize the attributes for data saving
        self.data = DataStruct()
        self.raw = RawStruct()
        
        # check system
        self.system = UIModules.system_check()
        # initialize AccessMyVNA
        #TODO add more code to disable settings_control tab and widges in settings_settings tab
        if self.system == 'win32': # windows
            try:
                # test if MyVNA program is available
                with AccessMyVNA() as vna:
                    if vna.Init() == 0: # is available
                        self.vna = AccessMyVNA() # save class AccessMyVNA to vna
                    else: # not available
                        pass
            except:
                print('Initiating MyVNA failed!\nMake sure analyser is connected and MyVNA is correctly installed!')

        else: # other system, data analysis only
            # self.vna = AccessMyVNA() # for test only
            pass
        print(self.vna)

        if self.vna is None:
            # initiate a timer for test
            self.timer = QTimer()
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.data_collection)

        self.main()
        self.load_settings()


    def main(self):
#region ###### initiate UI #################################

#region main UI 
        # link tabWidget_settings and stackedWidget_spectra and stackedWidget_data
        self.ui.tabWidget_settings.currentChanged.connect(self.link_tab_page)

#endregion


#region cross different sections
        # harmonic widgets
        # loop for setting harmonics 
        for i in range(1, settings_init['max_harmonic']+2, 2):
            # set to visable which is default. nothing to do

            # set all frame_sp<n> hided
            getattr(self.ui, 'frame_sp' +str(i)).setVisible(False)

            # add checkbox to tabWidget_ham for harmonic selection
            setattr(self.ui, 'checkBox_tree_harm' + str(i), QCheckBox())
            self.ui.tabWidget_settings_settings_harm.tabBar().setTabButton(
                self.ui.tabWidget_settings_settings_harm.indexOf(
                    getattr(self.ui, 'tab_settings_settings_harm' + str(i))
                ), 
                QTabBar.LeftSide, 
                getattr(self.ui, 'checkBox_tree_harm' + str(i)
                )
            )

            # set signal
            getattr(self.ui, 'checkBox_tree_harm' + str(i)).toggled['bool'].connect(
                getattr(self.ui, 'checkBox_harm' + str(i)).setChecked
            )
            getattr(self.ui, 'checkBox_harm' + str(i)).toggled['bool'].connect(
                getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked
            )
            # getattr(self.ui, 'checkBox_tree_harm' + str(i)).toggled['bool'].connect(
            #     getattr(self.ui, 'frame_sp' +str(i)).setVisible
            # )
            getattr(self.ui, 'checkBox_harm' + str(i)).toggled['bool'].connect(
                getattr(self.ui, 'frame_sp' +str(i)).setVisible
            )

        # hid reference related widgets 
        self.setvisible_refwidgets(False)
        # set comboBox_plt1_choice, comboBox_plt2_choice
        # dict for the comboboxes
        for key, val in settings_init['data_plt_choose'].items():
            # userData is setup for geting the plot type
            # userDat can be access with itemData(index)
            self.ui.comboBox_plt1_choice.addItem(val, key)
            self.ui.comboBox_plt2_choice.addItem(val, key)

        # set RUN/STOP button
        self.ui.pushButton_runstop.clicked.connect(self.on_clicked_pushButton_runstop)

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
        for i in range(1, settings_init['max_harmonic']+2, 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_endf' + str(i)).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_startf' + str(i) + '_r').setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_endf' + str(i) + '_r').setStyleSheet(
                "QLineEdit { background: transparent; }"
            )

        # dateTimeEdit_reftime on dateTimeChanged
        self.ui.dateTimeEdit_reftime.dateTimeChanged.connect(self.on_dateTimeChanged_dateTimeEdit_reftime)
        # set pushButton_resetreftime
        self.ui.pushButton_resetreftime.clicked.connect(self.reset_reftime)

        # set lineEdit_scaninterval value
        self.ui.lineEdit_recordinterval.editingFinished.connect(self.set_lineEdit_scaninterval)
        self.ui.lineEdit_refreshresolution.editingFinished.connect(self.set_lineEdit_scaninterval)

        # add value to the comboBox_settings_control_dispmode
        for key, val in settings_init['display_choose'].items():
            self.ui.comboBox_settings_control_dispmode.addItem(val, key)
        self.ui.comboBox_settings_control_dispmode.activated.connect(self.update_widget)
        self.ui.comboBox_settings_control_dispmode.activated.connect(self. update_freq_display_mode)

        # set pushButton_gotofolder
        self.ui.pushButton_gotofolder.clicked.connect(self.on_clicked_pushButton_gotofolder)

        # set pushButton_newdata
        self.ui.pushButton_newdata.clicked.connect(self.on_triggered_new_exp)

        # set pushButton_appenddata
        self.ui.pushButton_appenddata.clicked.connect(self.on_triggered_load_exp)

        # set lineEdit_scaninterval background
        self.ui.lineEdit_scaninterval.setStyleSheet(
            "QLineEdit { background: transparent; }"
        )

        self.ui.checkBox_dynamicfit.stateChanged.connect(self.update_widget)
        self.ui.spinBox_fitfactor.valueChanged.connect(self.update_widget)
        self.ui.checkBox_dynamicfitbyharm.clicked['bool'].connect(self.update_widget)
        self.ui.checkBox_fitfactorbyharm.clicked['bool'].connect(self.update_widget)

        # set signals to update spectra show display options
        self.ui.radioButton_spectra_showGp.toggled.connect(self.update_widget)
        self.ui.radioButton_spectra_showBp.toggled.connect(self.update_widget)
        self.ui.radioButton_spectra_showpolar.toggled.connect(self.update_widget)
        self.ui.checkBox_spectra_shoechi.toggled.connect(self.update_widget)

        # set signals to checkBox_control_rectemp
        self.ui.checkBox_control_rectemp.clicked['bool'].connect(self.on_clicked_set_temp_sensor)

        # set checkBox_dynamicfitbyharm
        self.ui.checkBox_dynamicfitbyharm.clicked['bool'].connect(self.on_clicked_checkBox_dynamicfitbyharm)

        # set checkBox_fitfactorbyharm
        self.ui.checkBox_fitfactorbyharm.clicked['bool'].connect(self.on_clicked_checkBox_fitfactorbyharm)

#endregion


#region settings_settings

        # hide raido buttons radioButton_settings_settings_harmchnsamp
        self.ui.radioButton_settings_settings_harmchnsamp.setVisible(False)
        # hide raido buttons radioButton_settings_settings_harmchnref
        self.ui.radioButton_settings_settings_harmchnref.setVisible(False)

        # set signal
        self.ui.radioButton_settings_settings_harmchnsamp.toggled.connect(self.update_active_chn)
        self.ui.radioButton_settings_settings_harmchnref.toggled.connect(self.update_active_chn)

        ### add combobox into treewidget
        self.ui.tabWidget_settings_settings_harm.currentChanged.connect(self.update_harmonic_tab)
        # move lineEdit_scan_harmstart
        self.move_to_col2(
            self.ui.lineEdit_scan_harmstart,
            self.ui.treeWidget_settings_settings_harmtree,
            'Start',
            100,
        )

        # move lineEdit_scan_harmend
        self.move_to_col2(
            self.ui.lineEdit_scan_harmend,
            self.ui.treeWidget_settings_settings_harmtree,
            'End',
            100,
        )

        # move lineEdit_scan_harmsteps
        self.move_to_col2(
            self.ui.lineEdit_scan_harmsteps,
            self.ui.treeWidget_settings_settings_harmtree,
            'Steps',
            100,
        )

        # move frame_peaks_num
        self.move_to_col2(
            self.ui.frame_peaks_num,
            self.ui.treeWidget_settings_settings_harmtree,
            'Num.',
            160,
        )

        # move frame_peaks_policy
        self.move_to_col2(
            self.ui.frame_peaks_policy,
            self.ui.treeWidget_settings_settings_harmtree,
            'Policy',
            160,
        )

        # move lineEdit_peaks_threshold
        self.move_to_col2(
            self.ui.lineEdit_peaks_threshold,
            self.ui.treeWidget_settings_settings_harmtree,
            'Threshold',
            100,
        )

        # move lineEdit_peaks_prominence
        self.move_to_col2(
            self.ui.lineEdit_peaks_prominence,
            self.ui.treeWidget_settings_settings_harmtree,
            'Prominence',
            100,
        )

        # move checkBox_harmfit
        self.move_to_col2(
            self.ui.checkBox_harmfit,
            self.ui.treeWidget_settings_settings_harmtree,
            'Fit',
            100,
        )

        # move spinBox_harmfitfactor
        self.move_to_col2(
            self.ui.spinBox_harmfitfactor,
            self.ui.treeWidget_settings_settings_harmtree,
            'Factor',
            100,
        )

        # comboBox_tracking_method
        self.create_combobox(
            'comboBox_tracking_method', 
            settings_init['span_mehtod_choose'], 
            100, 
            'Method', 
            self.ui.treeWidget_settings_settings_harmtree
        )

        # add span track_method
        self.create_combobox(
            'comboBox_tracking_condition', 
            settings_init['span_track_choose'], 
            100, 
            'Condition', 
            self.ui.treeWidget_settings_settings_harmtree
        )


        # insert sample_channel
        self.create_combobox(
            'comboBox_sample_channel', 
            settings_init['sample_channel_choose'], 
            100, 
            'Sample Channel', 
            self.ui.treeWidget_settings_settings_hardware
        )

        # inser ref_channel
        self.create_combobox(
            'comboBox_ref_channel', 
            settings_init['ref_channel_choose'], 
            100, 
            'Ref. Channel', 
            self.ui.treeWidget_settings_settings_hardware
        )
        # connect ref_channel
        # self.ui.comboBox_ref_channel.currentIndexChanged.connect() #TODO add function checking if sample and ref have the same channel

        # insert base_frequency
        self.create_combobox(
            'comboBox_base_frequency', 
            settings_init['base_frequency_choose'], 
            100, 
            'Base Frequency', 
            self.ui.treeWidget_settings_settings_hardware
        )

        # insert bandwidth
        self.create_combobox(
            'comboBox_bandwidth', 
            settings_init['bandwidth_choose'], 
            100, 
            'Bandwidth', 
            self.ui.treeWidget_settings_settings_hardware
        )

        # add comBox_tempmodule to treeWidget_settings_settings_hardware
        self.create_combobox(
            'comboBox_tempmodule',
            UIModules.list_modules(settings_init['tempmodules_path']),  
            100,
            'Module',
            self.ui.treeWidget_settings_settings_hardware, 
        )
        self.settings['comboBox_tempmodule'] = self.ui.comboBox_tempmodule.itemData(self.ui.comboBox_tempmodule.currentIndex())
        self.ui.comboBox_tempmodule.activated.connect(self.update_widget)

        # add comboBox_tempdevice to treeWidget_settings_settings_hardware
        if self.vna and self.system == 'win32':
            self.settings['tempdevs_choose'] = \
            tempdevs_choose = \
            tempDevices.dict_available_devs(settings_init['devices_dict'])
            self.create_combobox(
                'comboBox_tempdevice',
                tempdevs_choose,  
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware, 
            )
            self.settings['comboBox_tempdevice'] = self.ui.comboBox_tempdevice.itemData(self.ui.comboBox_tempdevice.currentIndex())
            self.ui.comboBox_tempdevice.activated.connect(self.update_tempdevice)
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
            settings_init['thrmcpl_choose'], 
            100, 
            'Thrmcpl Type', 
            self.ui.treeWidget_settings_settings_hardware
        )

        if not self.settings['comboBox_tempdevice']: # vna or tempdevice are not availabel
            # set temp related widgets unavailable
            self.ui.checkBox_control_rectemp.setEnabled(False)
            self.ui.checkBox_settings_temp_sensor.setEnabled(False)
            self.ui.comboBox_tempmodule.setEnabled(False)
            self.ui.comboBox_tempdevice.setEnabled(False)
            self.ui.comboBox_thrmcpltype.setEnabled(False)


        # insert time_unit
        self.create_combobox(
            'comboBox_timeunit', 
            settings_init['time_unit_choose'], 
            100, 
            'Time Unit', 
            self.ui.treeWidget_settings_settings_plots
        )

        # insert temp_unit
        self.create_combobox(
            'comboBox_tempunit', 
            settings_init['temp_unit_choose'], 
            100, 
            'Temp. Unit', 
            self.ui.treeWidget_settings_settings_plots
        )

        # insert time scale
        self.create_combobox(
            'comboBox_timescale', 
            settings_init['scale_choose'], 
            100, 
            'Time Scale', 
            self.ui.treeWidget_settings_settings_plots
        )

        # insert gamma scale
        self.create_combobox(
            'comboBox_yscale', 
            settings_init['scale_choose'], 
            100, 
            'Y Scale', 
            self.ui.treeWidget_settings_settings_plots
        )

        # move checkBox_linktime to treeWidget_settings_settings_plots
        self.move_to_col2(
            self.ui.checkBox_linktime, 
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
        self.move_to_col2(
            self.ui.pushButton_settings_harm_cntr, 
            self.ui.treeWidget_settings_settings_harmtree, 
            'Scan', 
            50
        )
        
        # move center checkBox_settings_temp_sensor to treeWidget_settings_settings_hardware
        self.move_to_col2(
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
            "QTabBar::tab { height: 20px; width: 38px; padding: 0px; }" 
            "QTabBar::tab:selected, QTabBar::tab:hover { background: white; }"
            "QTabBar::tab:selected { height: 22px; width: 40px; border-bottom-color: none; }"
            "QTabBar::tab:selected { margin-left: -2px; margin-right: -2px; }"
            "QTabBar::tab:first:selected { margin-left: 0; width: 40px; }"
            "QTabBar::tab:last:selected { margin-right: 0; width: 40px; }"
            "QTabBar::tab:!selected { margin-top: 2px; }"
            )

        self.ui.lineEdit_recordinterval.setValidator(QDoubleValidator(0, math.inf, 12))
        self.ui.lineEdit_refreshresolution.setValidator(QIntValidator(0, 2147483647))
        self.ui.lineEdit_scan_harmstart.setValidator(QDoubleValidator(1, math.inf, 12))
        self.ui.lineEdit_scan_harmend.setValidator(QDoubleValidator(1, math.inf, 12))
        self.ui.lineEdit_scan_harmsteps.setValidator(QIntValidator(0, 2147483647))
        self.ui.lineEdit_peaks_threshold.setValidator(QDoubleValidator(0, math.inf, 12))
        self.ui.lineEdit_peaks_prominence.setValidator(QDoubleValidator(0, math.inf, 12))

        # set signals of widgets in tabWidget_settings_settings_harm
        self.ui.lineEdit_scan_harmstart.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.lineEdit_scan_harmend.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.comboBox_base_frequency.currentIndexChanged.connect(self.update_base_freq)
        self.ui.comboBox_bandwidth.currentIndexChanged.connect(self.update_bandwidth)

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
    
        # set signals to update hardware settings_settings
        self.ui.comboBox_sample_channel.activated.connect(self.update_widget)
        self.ui.comboBox_sample_channel.activated.connect(self.update_vnachannel)
        self.ui.comboBox_sample_channel.activated.connect(self.update_active_chn)
        self.ui.comboBox_ref_channel.activated.connect(self.update_widget)
        self.ui.comboBox_ref_channel.activated.connect(self.update_vnachannel)
        self.ui.comboBox_ref_channel.activated.connect(self.update_active_chn)

        # self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.update_tempsensor)
        self.ui.checkBox_settings_temp_sensor.clicked['bool'].connect(self.on_clicked_set_temp_sensor)
        self.ui.comboBox_thrmcpltype.activated.connect(self.update_tempdevice)
        self.ui.comboBox_thrmcpltype.activated.connect(self.update_thrmcpltype)

        # set signals to update plots settings_settings
        self.ui.comboBox_timeunit.activated.connect(self.update_timeunit)
        self.ui.comboBox_tempunit.activated.connect(self.update_tempunit)
        self.ui.comboBox_timescale.activated.connect(self.update_timescale)
        self.ui.comboBox_yscale.activated.connect(self.update_yscale)
        self.ui.checkBox_linktime.stateChanged.connect(self.update_linktime)
        
#endregion


#region settings_data

        # set treeWidget_settings_data_settings background
        self.ui.treeWidget_settings_data_settings.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # insert refernence type
        self.create_combobox(
            'comboBox_ref_type', 
            settings_init['ref_type_choose'], 
            100, 
            'Type', 
            self.ui.treeWidget_settings_data_settings
        )
        
       # set treeWidget_settings_data_settings expanded
        self.ui.treeWidget_settings_data_settings.expandToDepth(0)

#endregion 


#region settings_mechanis
        ######### 
        # hide tableWidget_settings_mechanics_errortab
        self.ui.tableWidget_settings_mechanics_errortab.hide()
        # hide tableWidget_settings_mechanics_contoursettings
        self.ui.tableWidget_settings_mechanics_contoursettings.hide()
        # hide groupBox_settings_mechanics_simulator
        self.ui.groupBox_settings_mechanics_simulator.hide()

#endregion


#region spectra_show

#endregion


#region spectra_fit

        self.ui.horizontalSlider_spectra_fit_spanctrl.valueChanged.connect(self.on_changed_slider_spanctrl)
        self.ui.horizontalSlider_spectra_fit_spanctrl.sliderReleased.connect(self.on_released_slider_spanctrl)
        self.ui.horizontalSlider_spectra_fit_spanctrl.actionTriggered .connect(self.on_acctiontriggered_slider_spanctrl)

        # pushButton_spectra_fit_refresh
        self.ui.pushButton_spectra_fit_refresh.clicked.connect(self.on_clicked_pushButton_spectra_fit_refresh)
        self.ui.pushButton_spectra_fit_showall.clicked.connect(self.on_clicked_pushButton_spectra_fit_showall)
        self.ui.pushButton_spectra_fit_fit.clicked.connect(self.on_clicked_pushButton_spectra_fit_fit)

#endregion


#region spectra_mechanics


#endregion


#region data_data
        # set signals to update plot 1 & 2 options
        for i in range(1, self.settings['max_harmonics']+2, 2):
            getattr(self.ui, 'checkBox_plt1_h' + str(i)).stateChanged.connect(self.update_widget)
            getattr(self.ui, 'checkBox_plt2_h' + str(i)).stateChanged.connect(self.update_widget)

        # set signals to update plot 1 options
        self.ui.comboBox_plt1_choice.activated.connect(self.update_widget)
        self.ui.radioButton_plt1_ref.toggled.connect(self.update_widget)
        self.ui.radioButton_plt1_samp.toggled.connect(self.update_widget)

        # set signals to update plot 2 options
        self.ui.comboBox_plt2_choice.activated.connect(self.update_widget)
        self.ui.radioButton_plt2_ref.toggled.connect(self.update_widget)
        self.ui.radioButton_plt2_samp.toggled.connect(self.update_widget)

#endregion


#region data_mechanics

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
        # move label_status_f0BW to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_f0BW)

#endregion


#region action group

        # add menu to toolbutton

        # toolButton_settings_data_refit
        # create menu: menu_settings_data_refit
        self.ui.menu_settings_data_refit = QMenu(self.ui.toolButton_settings_data_refit)
        self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_all)
        self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_marked)
        self.ui.menu_settings_data_refit.addAction(self.ui.actionFit_selected)
        # add menu to toolbutton
        self.ui.toolButton_settings_data_refit.setMenu(self.ui.menu_settings_data_refit)

        # toolButton_settings_mechanics_solve
        # create menu: menu_settings_mechanics_solve
        self.ui.menu_settings_mechanics_solve = QMenu(self.ui.toolButton_settings_mechanics_solve)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_all)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_marked)
        self.ui.menu_settings_mechanics_solve.addAction(self.ui.actionSolve_selected)
        # add menu to toolbutton
        self.ui.toolButton_settings_mechanics_solve.setMenu(self.ui.menu_settings_mechanics_solve)

        # toolButton_spectra_mechanics_plotrows
        # create menu: menu_spectra_mechanics_plotrows
        self.ui.menu_spectra_mechanics_plotrows = QMenu(self.ui.toolButton_spectra_mechanics_plotrows)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRows_time)
        self.ui.menu_spectra_mechanics_plotrows.addAction(self.ui.actionRow_s1_Row_s2)
        # add menu to toolbutton
        self.ui.toolButton_spectra_mechanics_plotrows.setMenu(self.ui.menu_spectra_mechanics_plotrows)

        # set action group channel
        self.ui.group_channel = QActionGroup(self, exclusive=True)
        self.ui.group_channel.addAction(self.ui.actionADC_1)
        self.ui.group_channel.addAction(self.ui.actionADC_2)

        # set action group refType
        self.ui.group_refType = QActionGroup(self, exclusive=True)
        self.ui.group_refType.addAction(self.ui.actionData_File)
        self.ui.group_refType.addAction(self.ui.actionSingle_Point)
        self.ui.group_refType.addAction(self.ui.actionExternal)

        # set action group f0
        self.ui.group_f0 = QActionGroup(self, exclusive=True)
        self.ui.group_f0.addAction(self.ui.action5_MHz)
        self.ui.group_f0.addAction(self.ui.action6_MHz)
        self.ui.group_f0.addAction(self.ui.action9_MHz)
        self.ui.group_f0.addAction(self.ui.action10_MHz)

        # set action group BW
        self.ui.group_BW = QActionGroup(self, exclusive=True)
        self.ui.group_BW.addAction(self.ui.actionBW_2_MHz)
        self.ui.group_BW.addAction(self.ui.actionBW_1_MHz)
        self.ui.group_BW.addAction(self.ui.actionBW_0_5_MHz)
        self.ui.group_BW.addAction(self.ui.actionBW_0_25_MHz)
        self.ui.group_BW.addAction(self.ui.actionBW_0_1_MHz)

        # set QAction
        self.ui.actionLoad_Settings.triggered.connect(self.on_triggered_load_settings)
        self.ui.actionLoad_Exp.triggered.connect(self.on_triggered_load_exp)
        self.ui.actionNew_Exp.triggered.connect(self.on_triggered_new_exp)
        self.ui.actionSave.triggered.connect(self.on_triggered_actionSave)
        self.ui.actionSave_As.triggered.connect(self.on_triggered_actionSave_As)
        self.ui.actionExport.triggered.connect(self.on_triggered_actionExport)
        self.ui.actionReset.triggered.connect(self.on_triggered_actionReset)

#endregion


#region ###### add Matplotlib figures in to frames ##########

        # # create an empty figure and move its toolbar to TopToolBarArea of main window
        # self.ui.mpl_dummy_fig = MatplotlibWidget()
        # self.addToolBar(Qt.TopToolBarArea, self.ui.mpl_dummy_fig.toolbar)
        # self.ui.mpl_dummy_fig.hide() # hide the figure

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
        
        # add figure mpl_sp[n] into frame_sp[n]
        for i in range(1, settings_init['max_harmonic']+2, 2):
            # add first ax
            setattr(
                self.ui, 'mpl_sp' + str(i), 
                MatplotlibWidget(
                    parent=getattr(self.ui, 'frame_sp' + str(i)), 
                    axtype='sp',
                    showtoolbar=False,
                )
            )
            getattr(self.ui, 'mpl_sp' + str(i)).fig.text(0.01, 0.98, str(i), va='top',ha='left') # option: weight='bold'
            # set mpl_sp<n> border
            getattr(self.ui, 'mpl_sp' + str(i)).setStyleSheet(
                "border: 0;"
            )
            getattr(self.ui, 'mpl_sp' + str(i)).setContentsMargins(0, 0, 0, 0)
            getattr(self.ui, 'frame_sp' + str(i)).setLayout(
                self.set_frame_layout(
                    getattr(self.ui, 'mpl_sp' + str(i))
                )
            )


        # add figure mpl_spectra_fit_polar into frame_spectra_fit_polar
        self.ui.mpl_spectra_fit_polar = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit_polar, 
            axtype='sp_polar'
            )
        self.ui.frame_spectra_fit_polar.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit_polar))

        # add figure mpl_spectra_fit into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            axtype='sp_fit',
            showtoolbar=('Back', 'Forward', 'Pan', 'Zoom')
            ) 
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))
        # connect signal
        self.ui.mpl_spectra_fit.ax[0].cidx = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('xlim_changed', self.on_fit_lims_change)
        self.ui.mpl_spectra_fit.ax[0].cidy = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('ylim_changed', self.on_fit_lims_change)
        
        # disconnect signal while dragging
        self.ui.mpl_spectra_fit.canvas.mpl_connect('button_press_event', self.spectra_fit_axesevent_disconnect)
        # reconnect signal after dragging (mouse release)
        self.ui.mpl_spectra_fit.canvas.mpl_connect('button_release_event', self.spectra_fit_axesevent_connect)
            
        # add figure mpl_countour1 into frame_spectra_mechanics_contour1
        self.ui.mpl_countour1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour1, 
            axtype='contour'
            )
        self.ui.frame_spectra_mechanics_contour1.setLayout(self.set_frame_layout(self.ui.mpl_countour1))

        # add figure mpl_countour2 into frame_spectra_mechanics_contour2
        self.ui.mpl_countour2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour2, 
            axtype='contour',
            )
        self.ui.frame_spectra_mechanics_contour2.setLayout(self.set_frame_layout(self.ui.mpl_countour2))

        # add figure mpl_plt1 into frame_spactra_fit
        self.ui.mpl_plt1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            axtype='data',
            ylabel=r'$\Delta f/n$ (Hz)',
            )
        self.ui.frame_plt1.setLayout(self.set_frame_layout(self.ui.mpl_plt1))

        # add figure mpl_plt2 into frame_spactra_fit
        self.ui.mpl_plt2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            axtype='data',
            ylabel=r'$\Delta \Gamma$ (Hz)',
            )
        self.ui.frame_plt2.setLayout(self.set_frame_layout(self.ui.mpl_plt2))

#endregion


#endregion

#region ###### set UI value ###############################

        # for i in range(1, settings_init['max_harmonic']+2, 2):
        #     if i in self.settings['harmonics_check']: # in the default range 
        #         # settings/control/Harmonics
        #         getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(True)
        #         getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(True)

        #     else: # out of the default range
        #         getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(False)
        #         getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(False)
        #         # hide spectra/sp
        #         getattr(self.ui, 'frame_sp' + str(i)).setVisible(False)


        # self.ui.comboBox_plt1_choice.setCurrentIndex(2)
        # self.ui.comboBox_plt2_choice.setCurrentIndex(3)

        # set time interval
        # self.ui.lineEdit_scaninterval.setText(str(self.settings['lineEdit_scaninterval']))
        # self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        # self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))

#endregion


#region #########  functions ##############

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
            self.move_to_col2(obj_box, parent, row_text, box_width)
            

    def move_to_col2(self, obj, parent, row_text, width=[]): 
        if width: # set width of obj
            obj.setMaximumWidth(width)
        # find item with row_text
        item = self.find_text_item(parent, row_text)
        # insert the combobox in to the 2nd column of row_text
        parent.setItemWidget(item, 1, obj)        

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
            # check filename avaialbe

            # disable features

            # check active harmonice if no, stop



            self.ui.pushButton_runstop.setText('STOP')
        else:
            # stop running timer and/or test

            # save data

        # write UI information to file

            # enable features

            # 
            self.ui.pushButton_runstop.setText('START RECORD')
            self.idle = True
            return

        ####### below is the recording routine ########
        self.idle = False
        # file initiate or get append information
        # filename format check?

        # if no filename, set a temp file for data saving

        # cmd diary?

        # test scheduler? start/end increasement

        # start the timer
        self.timer.start(1000)

        self.reading = True
        # read time

        # read temp if checked 

        # scan harmonics (1, 3, 5...)

        self.reading = False

        self.writing = True
        # save scans to file

        self.writing = False

        # plot scans

        # dynamic fit if checked (for checked harmonics)

            # plot fitting
            
            # plot data 

        # peak treaking ( if dynamic fitted, use fitted data)

        # write fitting to dataframe
        # 
        # display total points collected 

        # wait bar









    # @pyqtSlot()
    def reset_reftime(self):
        ''' set time in dateTimeEdit_reftime '''
        # use qt use python deal with datetime. But show the time with QdatetimeEdit
        current_time = datetime.datetime.now()
        self.ui.dateTimeEdit_reftime.setDateTime(current_time)
        # # update reftime in settings dict
        # self.settings['dateTimeEdit_reftime'] = current_time
        # print(self.settings['dateTimeEdit_reftime'])
    
    def on_dateTimeChanged_dateTimeEdit_reftime(self, datetime):
        '''
        get time in dateTimeEdit_reftime and save it to self.settings
        '''
        self.settings['dateTimeEdit_reftime'] = datetime.toPyDateTime()
        print(self.settings['dateTimeEdit_reftime'])
        
    # @pyqtSlot()
    def set_lineEdit_scaninterval(self):
        # get text
        record_interval = self.ui.lineEdit_recordinterval.text()
        refresh_resolution = self.ui.lineEdit_refreshresolution.text()
        #convert to flot
        try:
            record_interval = float(record_interval)
            if record_interval <= 0: # illegal value
                raise ZeroDivisionError
        except:
            record_interval = self.settings['lineEdit_recordinterval']
            self.ui.lineEdit_recordinterval.setText(str(record_interval))
            self.settings['lineEdit_recordinterval'] = record_interval
        try:
            refresh_resolution = float(refresh_resolution)
            if refresh_resolution <= 0: # illegal value
                raise ZeroDivisionError
        except:
            refresh_resolution = settings_init['lineEdit_refreshresolution']
            self.ui.lineEdit_refreshresolution.setText(refresh_resolution)
            self.settings['lineEdit_refreshresolution'] = refresh_resolution
            
        # set lineEdit_scaninterval
        # self.ui.lineEdit_scaninterval.setText(f'{record_interval * refresh_resolution}  s')
        self.settings['lineEdit_recordinterval'] = float(record_interval)
        self.settings['lineEdit_refreshresolution'] = float(refresh_resolution)
        try:
            self.settings['lineEdit_scaninterval'] = record_interval / refresh_resolution
            self.ui.lineEdit_scaninterval.setText('{0:.3g}'.format(record_interval / refresh_resolution)) # python < 3.5
        except ZeroDivisionError:
            self.settings['lineEdit_scaninterval'] = 1
            self.ui.lineEdit_scaninterval.setText('{0:.3g}'.format(math.inf)) # python < 3.5

    ## functions for open and save file
    def openFileNameDialog(self, title, path='', filetype=settings_init['default_datafiletype']):  
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, title, path, filetype, options=options)
        if fileName:
            print(type(fileName))
        else:
            fileName = ''
        return fileName
        
    # def openFileNamesDialog(self, title, path=''):    
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     files, _ = QFileDialog.getOpenFileNames(self,title, "","All Files (*);;Python Files (*.py)", options=options)
    #     if files:
    #         print(files)
 
    def saveFileDialog(self, title, path='', filetype=settings_init['default_datafiletype']):    
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,title, path, filetype, options=options)
        if fileName:
            print(fileName)
        else:
            fileName = ''
        return fileName 

    def on_triggered_new_exp(self):
        fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        if fileName:
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(fileName)
            # reset dateTimeEdit_reftime
            self.reset_reftime()
            # set dateTimeEdit_reftime editabled and enable pushButton_resetreftime
            self.ui.dateTimeEdit_reftime.setReadOnly(False)
            self.ui.pushButton_resetreftime.setEnabled(True)
            self.fileName = fileName

    def on_triggered_load_exp(self): 
        fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
        if fileName:
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(self.fileName)
            # set dateTimeEdit_reftime
            # set dateTimeEdit_reftime read only and disable pushButton_resetreftime
            self.ui.dateTimeEdit_reftime.setReadOnly(True)
            # ??  set reftime in fileName to dateTimeEdit_reftime

            self.ui.pushButton_resetreftime.setEnabled(False)
            self.fileName = fileName

    # open folder in explorer
    # methods for different OS could be added
    def on_clicked_pushButton_gotofolder(self):
        file_path = self.ui.lineEdit_datafilestr.text() #TODO replace with reading from settings dict
        path = os.path.abspath(os.path.join(file_path, os.pardir)) # get the folder of the file
        UIModules.open_file(path)

    # 
    def on_triggered_load_settings(self):
        self.fileName = self.openFileNameDialog('Choose a file to use its setting') # !! add path of last opened folder
        try:
            # load json file containing formerly saved settings
            with open(self.fileName, 'r') as f:
                self.settings = json.load(f)
            # load settings from file into gui
            self.load_settings()
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(self.fileName)
            # indicate that a file has been loadded
            self.fileFlag = True
        # pass if action cancelled
        except:
            pass

    def on_triggered_actionSave(self):
        # save current data to file if file has been opened
        if self.fileFlag == True:
            with open(self.fileName, 'w') as f:
                line = json.dumps(dict(self.settings), indent=4) + "\n"
                f.write(line)
        # save current data to new file otherwise
        else:
            self.on_triggered_actionSave_As()


    def on_triggered_actionSave_As(self):
        # save current data to a new file 
        self.fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        try:
            with open(self.fileName, 'w') as f:
                line = json.dumps(dict(self.settings), indent=4) + "\n"
                f.write(line)
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(self.fileName)
            # indicate that a file has been loaded
            self.fileFlag = True
        # pass if action cancelled
        except:
            pass

    def on_triggered_actionExport(self):
        # export data to a selected form
        fileName = self.saveFileDialog(title='Choose a file and data type', filetype=settings_init['export_datafiletype']) # !! add path of last opened folder
        # codes for data exporting

    def on_triggered_actionReset(self):
        # reset MainWindow
        pass

    def on_acctiontriggered_slider_spanctrl(self, value):
        '''
        disable the actions other than mouse dragging
        '''
        # print(value)
        if value < 7: # mous dragging == 7
            # reset slider to 1
            self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)

    def on_changed_slider_spanctrl(self):
        # get slider value
        n = 10 ** (self.ui.horizontalSlider_spectra_fit_spanctrl.value() / 10)
        # format n
        if n >= 1:
            # n = f'{round(n)} *'
            n = '{} *'.format(min(settings_init['span_ctrl_steps'], key=lambda x:abs(x-n))) # python < 3.5
        else:
            # n = f'1/{round(1/n)} *'
            n = '1/{} *'.format(min(settings_init['span_ctrl_steps'], key=lambda x:abs(x-1/n))) # python < 3.5
        # set label_spectra_fit_zoomtimes value
        self.ui.label_spectra_fit_zoomtimes.setText(str(n))

    def on_released_slider_spanctrl(self):

        # get slider value
        n = 10 ** (self.ui.horizontalSlider_spectra_fit_spanctrl.value() / 10)
        # format n
        if n >= 1:
            n = min(settings_init['span_ctrl_steps'], key=lambda x:abs(x-n))
        else:
            n = 1/min(settings_init['span_ctrl_steps'], key=lambda x:abs(x-1/n))

        # get f1, f2
        # f1, f2 = self.ui.mpl_spectra_fit.ax[0].get_xlim()
        f1, f2 = self.get_freq_span()
        # convert start/end (f1/f2) to center/span (fc/fs)
        fc, fs = MathModules.converter_startstop_to_centerspan(f1, f2)
        # multiply fs
        fs = fs * n
        # fc/fs back to f1/f2
        f1, f2 = MathModules.converter_centerspan_to_startstop(fc, fs)

        # set lineEdit_scan_harmstart & lineEdit_scan_harmend
        self.ui.lineEdit_scan_harmstart.setText(str(f1*1e-6)) # in MHz
        self.ui.lineEdit_scan_harmend.setText(str(f2*1e-6)) # in MHz

        # reset xlim to active on_fit_lims_change
        self.ui.mpl_spectra_fit.ax[0].set_xlim(f1, f2)

        # # update limit of active harmonic
        # self.on_editingfinished_harm_freq()
        
        # # get new data
        # f, G, B = self.sepectra_fit_get_data()
        
        # # plot
        # self.tab_spectra_fit_update_mpls(f, G, B)

        # reset slider to 1
        self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)


    def span_check(self, harm=None, f1=None, f2=None):
        '''
        check if lower limit ('f1' in Hz) and upper limit ('f2' in Hz) in base freq +/- BW of harmonic 'harm'
        if out of the limit, return the part in the range
        and show alert in statusbar
        '''
        if harm is None:
            harm = self.settings_harm
        # get freq_range
        bf1, bf2 = self.settings['freq_range'][harm] # in Hz
        # check f1, and f2
        if f1 and (f1 < bf1 or f1 >= bf2): # f1 out of limt
            f1 = bf1
            #TODO update statusbar 'lower bound out of limit and reseted. (You can increase the bandwidth in settings)'
        if f2 and (f2 > bf2 or f2 <= bf1): # f2 out of limt
            f2 = bf2
            #TODO update statusbar 'upper bond out of limit and reseted. (You can increase the bandwidth in settings)'
        if f1 and f2 and (f1 >= f2):
            f2 = bf2

        return np.array([f1, f2])

    def get_spectraTab_mode(self):
        '''
        get the current UI condition from attributes and 
        set the mode for spectra_fit
        '''
        mode = None,   # None/center/refit
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
        return mode

    def sepectra_fit_get_data(self):
        ''' 
        get data fro mpl_spectra_fit by spectraTab_mode and 
        return f, G, B
        '''
        f = None
        G = None
        B = None
        if self.get_spectraTab_mode() == 'center': # for peak centering
            # get harmonic from self.settings_harm
            harm = self.settings_harm
            # get f1, f2
            freq_span = self.get_freq_span()
            steps = int(self.get_harmdata('lineEdit_scan_harmsteps', harm=harm))
            chn = self.active_chn['chn']

            # get the vna reset flag
            setflg = self.vna_tracker.set_check(f=freq_span, steps=steps, chn=chn)
            print(setflg)

            with self.vna as vna:
                ret = vna.set_vna(setflg)
                if ret == 0:
                    ret, f, G, B = vna.single_scan()
                    return f, G, B
                else:
                    print('There is an error while setting VNA!')
        elif self.get_spectraTab_mode() == 'refit': # for refitting
            #TODO getting data from selected index
            pass
        else:
            print('Change Tab to Settings or Data to active the function.')
        
        return f, G, B

    def tab_spectra_fit_update_mpls(self, f, G, B):
        ''' update mpl_spectra_fit and mpl_spectra_fit_polar '''
        ## disconnect axes event
        self.mpl_disconnect_cid(self.ui.mpl_spectra_fit) 
               
        self.ui.mpl_spectra_fit.update_data(('lG', f, G))
        self.ui.mpl_spectra_fit.update_data(('lB', f, B))

        # constrain xlim
        self.ui.mpl_spectra_fit.ax[0].set_xlim(f[0], f[-1])
        self.ui.mpl_spectra_fit.ax[1].set_xlim(f[0], f[-1])
        self.ui.mpl_spectra_fit.ax[0].set_ylim(min(G)-0.05*(max(G)-min(G)), max(G)+0.05*(max(G)-min(G)))
        self.ui.mpl_spectra_fit.ax[1].set_ylim(min(B)-0.05*(max(B)-min(B)), max(B)+0.05*(max(B)-min(B)))

        ## connect axes event
        self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)

        self.ui.mpl_spectra_fit.canvas.draw()

        self.ui.mpl_spectra_fit_polar.update_data(('l', G, B))
        
        # set xlabel
        self.mpl_set_faxis(self.ui.mpl_spectra_fit.ax[0])

        # update lineedit_fit_span
        self.update_lineedit_fit_span(f)


    def on_clicked_pushButton_spectra_fit_refresh(self):
        print('vna', self.vna)
        # get data
        f, G, B = self.sepectra_fit_get_data()

        # update raw
        self.tab_spectra_fit_update_mpls(f, G, B)


    def on_clicked_pushButton_spectra_fit_showall(self):
        ''' show whole range of current harmonic'''
        # get harmonic
        harm = self.settings_harm
        # set freq_span[harm] to the maximum range (freq_range[harm])
        self.set_freq_span(self.settings['freq_range'][harm])
        # get data
        f, G, B = self.sepectra_fit_get_data()
        # updata data
        self.tab_spectra_fit_update_mpls(f, G, B)      

    def on_fit_lims_change(self, axes):
        print('on lim changed')
        axG = self.ui.mpl_spectra_fit.ax[0]

        # print('g', axG.get_contains())
        # print('r', axG.contains('button_release_event'))
        # print('p', axG.contains('button_press_event'))

        # data lims [min, max]
        # df1, df2 = MathModules.datarange(self.ui.mpl_spectra_fit.l['lB'][0].get_xdata())
        # get axes lims
        f1, f2 = axG.get_xlim()
        # check lim with BW
        f1, f2 = self.span_check(harm=self.settings_harm, f1=f1, f2=f2)
        print('get_navigate_mode()', axG.get_navigate_mode())
        print('flims', f1, f2)
        # print(df1, df2)
        
        print(axG.get_navigate_mode())
        # if axG.get_navigate_mode() == 'PAN': # pan
        #     # set a new x range: combine span of dflims and flims
        #     f1 = min([f1, df1])
        #     f2 = max([f2, df2])
        # elif axG.get_navigate_mode() == 'ZOOM': # zoom
        #     pass
        # else: # axG.get_navigate_mode() == 'None'
        #     pass
        print('f12', f1, f2)

        # set lineEdit_scan_harmstart & lineEdit_scan_harmend
        self.ui.lineEdit_scan_harmstart.setText(str(f1*1e-6)) # in MHz
        self.ui.lineEdit_scan_harmend.setText(str(f2*1e-6)) # in MHz

        # update limit of active harmonic
        self.on_editingfinished_harm_freq()

        # get new data
        f, G, B = self.sepectra_fit_get_data()
        
        # plot
        self.tab_spectra_fit_update_mpls(f, G, B)

    def update_lineedit_fit_span(self, f):
        ''' 
        update lineEdit_spectra_fit_span text 
        input
        f: list like data in Hz
        '''
        span = max(f) - min(f)

        # update 
        self.ui.lineEdit_spectra_fit_span.setText(MathModules.num2str((span / 1000), precision=5)) # in kHz

    def spectra_fit_axesevent_disconnect(self, event):
        print('disconnect')
        self.mpl_disconnect_cid(self.ui.mpl_spectra_fit)

    def spectra_fit_axesevent_connect(self, event):
        print('connect')
        self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)
        # since pan changes xlim before button up, change ylim a little to trigger ylim_changed
        ax = self.ui.mpl_spectra_fit.ax[0]
        print('cn', ax.get_navigate_mode())
        if ax.get_navigate_mode() == 'PAN':
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], ylim[1] * 1.01)

    def mpl_disconnect_cid(self, mpl):
        mpl.ax[0].callbacks.disconnect(mpl.ax[0].cidx)
        mpl.ax[0].callbacks.disconnect(mpl.ax[0].cidy)

    def mpl_connect_cid(self, mpl, fun):
        mpl.ax[0].cidx = mpl.ax[0].callbacks.connect('xlim_changed', fun)
        mpl.ax[0].cidy = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('ylim_changed', fun)
    
    def mpl_set_faxis(self, ax):
        '''
        set freq axis tack as: [-1/2*span, 1/2*span] and
        freq axis label as: f (+cnter Hz)
        '''
        # get xlim
        xlim = ax.get_xlim()
        print(xlim)
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

    def on_clicked_pushButton_spectra_fit_fit(self):
        '''
        fit Gp, Bp data shown in mpl_spectra_fit ('lG' and 'lB')
        '''
        # get data in tuple (x, y)
        data_lG, data_lB = self.ui.mpl_spectra_fit.get_data(ls=['lG', 'lB'])

        # factor = self.get_harmdata('spinBox_harmfitfactor')

        # get guessed value of cen and wid

        ## fitting peak
        print('main set harm', self.settings_harm)
        self.peak_tracker.update_input(self.active_chn['name'], self.settings_harm, data_lG[0], data_lG[1], data_lB[1], self.settings['harmdata'])

        fit_result = self.peak_tracker.peak_fit(self.active_chn['name'], self.settings_harm, components=True)
        print(fit_result['v_fit'])
        # print(fit_result['comp_g'])
        # plot fitted data
        self.ui.mpl_spectra_fit.update_data(('lGfit',data_lG[0], fit_result['fit_g']), ('lBfit',data_lB[0], fit_result['fit_b']))
        self.ui.mpl_spectra_fit_polar.update_data(('lfit',fit_result['fit_g'], fit_result['fit_b']))

        # clear l.['temp'][:]
        self.ui.mpl_spectra_fit.del_templines()
        self.ui.mpl_spectra_fit_polar.del_templines()
        # add devided peaks
        self.ui.mpl_spectra_fit.add_temp_lines(self.ui.mpl_spectra_fit.ax[0], xlist=[data_lG[0]] * len(fit_result['comp_g']), ylist=fit_result['comp_g'])
        self.ui.mpl_spectra_fit_polar.add_temp_lines(self.ui.mpl_spectra_fit_polar.ax[0],xlist=fit_result['comp_g'], ylist=fit_result['comp_b'])

        # update lsp
        factor_span = self.peak_tracker.get_output(key='factor_span', chn_name=self.active_chn['name'], harm=self.settings_harm)
        gc_list = [fit_result['v_fit']['g_c']['value']] * 2 # make its len() == 2

        print(factor_span)
        print(gc_list)

        self.ui.mpl_spectra_fit.update_data(('lsp', factor_span, gc_list))

        # update strk
        cen_trk_freq = fit_result['v_fit']['cen_trk']['value']
        cen_trk_G = self.peak_tracker.get_output(key='gmod', chn_name=self.active_chn['name'], harm=self.settings_harm).eval(
            self.peak_tracker.get_output(key='params', chn_name=self.active_chn['name'], harm=self.settings_harm),
            x=cen_trk_freq
        ) 

        print(cen_trk_freq)
        print(cen_trk_G)

        self.ui.mpl_spectra_fit.update_data(('strk', cen_trk_freq, cen_trk_G))

        # update srec
        cen_rec_freq = fit_result['v_fit']['cen_rec']['value']
        cen_rec_G = self.peak_tracker.get_output(key='gmod', chn_name=self.active_chn['name'], harm=self.settings_harm).eval(
            self.peak_tracker.get_output(key='params', chn_name=self.active_chn['name'], harm=self.settings_harm),
            x=cen_rec_freq
        ) 

        print(cen_rec_freq)
        print(cen_rec_G)

        self.ui.mpl_spectra_fit.update_data(('srec', cen_rec_freq, cen_rec_G))








    def on_clicked_set_temp_sensor(self, checked):
        # below only runs when vna is available
        if self.vna: # add not for testing code    
            if checked: # checkbox is checked
                # if not self.tempsensor: # tempModule is not initialized 
                # get all tempsensor settings 
                tempmodule_name = self.settings['comboBox_tempmodule'] # get temp module

                thrmcpltype = self.settings['comboBox_thrmcpltype'] # get thermocouple type
                tempdevice = tempDevices.device_info(self.settings['comboBox_tempdevice']) #get temp device info

                # check senor availability
                package_str = settings_init['tempmodules_path'][2:].replace('/', '.') + tempmodule_name
                print(package_str)
                # import package
                tempsensor = getattr(importlib.import_module(package_str), 'TempSensor')

                try:
                    self.tempsensor = tempsensor(
                        tempdevice,
                        settings_init['devices_dict'][tempdevice.product_type],
                        thrmcpltype,
                    )
                except Exception as e: # if failed return
                    print(e)
                    #TODO update in statusbar
                    return 

                # after tempModule loaded
                # # tempModule should take one arg 'thrmcpltype' and return temperature in C by calling tempModule.get_tempC
                try:
                    curr_temp = self.tempsensor.get_tempC()

                    # save values to self.settings
                    self.settings['checkBox_control_rectemp'] = True
                    self.settings['checkBox_settings_temp_sensor'] = True
                    # set statusbar pushButton_status_temp_sensor text
                    self.statusbar_temp_update()
                    # disable items to keep the setting
                    self.ui.comboBox_tempmodule.setEnabled(False)
                    self.ui.comboBox_tempdevice.setEnabled(False)
                    self.ui.comboBox_thrmcpltype.setEnabled(False)

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
                self.ui.comboBox_tempmodule.setEnabled(True)
                self.ui.comboBox_tempdevice.setEnabled(True)
                self.ui.comboBox_thrmcpltype.setEnabled(True)

                # reset self.tempsensor
                self.tempsensor = None
                
                
            # update checkBox_settings_temp_sensor to self.settings
            # self.update_tempsensor()

    def statusbar_temp_update(self):

        # update statusbar temp sensor image
        if self.settings['checkBox_settings_temp_sensor']: # checked
            self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor.svg"))
            try:
            # get temp and change temp unit by self.settings['temp_unit_choose']
                curr_temp = self.temp_by_unit(self.tempsensor.get_tempC())
                print(curr_temp)
                unit = settings_init['temp_unit_choose'].get(self.settings['comboBox_tempunit'])
                self.ui.pushButton_status_temp_sensor.setText('{:.1f} {}'.format(curr_temp, unit))
                self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor.svg"))
                self.ui.pushButton_status_temp_sensor.setToolTip('Temp. sensor is on.')
            except:
                #TODO update in statusbar
                pass
        else:
            self.ui.pushButton_status_temp_sensor.setIcon(QIcon(":/icon/rc/temp_sensor_off.svg"))
            self.ui.pushButton_status_temp_sensor.setText('')
            self.ui.pushButton_status_temp_sensor.setToolTip('Temp. sensor is off.')

    def temp_by_unit(self, data):
        '''
        data: double or ndarray
        unit: str. C for celsius, K for Kelvin and F for fahrenheit
        '''
        unit = self.settings['comboBox_tempunit']
        if unit == 'C':
            return data # temp data is saved as C
        elif unit == 'K': # convert to K
            return data + 273.15 
        elif unit == 'F': # convert to F
            return data * 9 / 5 + 32
        
    def on_clicked_checkBox_dynamicfitbyharm(self, value):
        self.ui.checkBox_dynamicfit.setEnabled(not value)
    
    def on_clicked_checkBox_fitfactorbyharm(self, value):
        self.ui.spinBox_fitfactor.setEnabled(not value)
        self.ui.label_fitfactor.setEnabled(not value)
        
    def set_stackedwidget_index(self, stwgt, idx=[], diret=[]):
        '''
        chenge the index of stwgt to given idx (if not []) 
        or to the given direction (if diret not [])
          diret=1: index += 1;
          diret=-1: index +=-1
        '''
        # print(self)
        if idx: # if index is not []
            stwgt.setCurrentIndex(idx) # set index to idx
        elif diret: # if diret is not []
            count = stwgt.count()  # get total pages
            current_index = stwgt.currentIndex()  # get current index
            stwgt.setCurrentIndex((current_index + diret) % count) # increase or decrease index by diret
    
    # update widget values in settings dict, only works with elements out of settings_settings
    
    def update_widget(self, signal):
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        print('update', signal)
        if isinstance(self.sender(), QLineEdit):
                try:
                    self.settings[self.sender().objectName()] = float(signal)
                except:
                    self.settings[self.sender().objectName()] = 0
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.settings[self.sender().objectName()] = not self.settings[self.sender().objectName()]
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.settings[self.sender().objectName()] = not self.settings[self.sender().objectName()]
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.settings[self.sender().objectName()] = value
        # if the sender of the signal isA QSpinBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QSpinBox):
            self.settings[self.sender().objectName()] = signal

    def update_harmwidget(self, signal):
        '''
        update widgets in treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        print('update', signal)
        harm = self.settings_harm

        if isinstance(self.sender(), QLineEdit):
                try:
                    self.set_harmdata(self.sender().objectName(), float(signal), harm=harm)
                except:
                    self.set_harmdata(self.sender().objectName(), 0, harm=harm)
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.set_harmdata(self.sender().objectName(), signal, harm=harm)
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.set_harmdata(self.sender().objectName(), signal, harm=harm)
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.set_harmdata(self.sender().objectName(), value, harm=harm)
        # if the sender of the signal isA QSpinBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QSpinBox):
            self.set_harmdata(self.sender().objectName(), signal, harm=harm)

    def update_active_chn(self):
        if self.sender().objectName() == 'radioButton_settings_settings_harmchnsamp': # switched to samp
            self.active_chn = {
                'name': 'samp', 
                'chn': self.settings['comboBox_sample_channel']
            }
        elif self.sender().objectName() == 'radioButton_settings_settings_harmchnref': # switched to ref
            self.active_chn = {
                'name': 'ref', 
                'chn': self.settings['comboBox_ref_channel']
            }
        elif self.sender().objectName() == 'comboBox_ref_channel' or 'comboBox_sample_channel': # define of samp/ref channel(s) changed
            # reset corrresponding ADC
            print(self.settings['comboBox_sample_channel'])
            print(self.settings['comboBox_ref_channel'])
            if self.active_chn['name'] == 'samp':
                self.active_chn['chn'] = self.settings['comboBox_sample_channel']
            elif self.active_chn['name'] == 'ref':
                self.active_chn['chn'] = self.settings['comboBox_ref_channel']
            print(self.active_chn)
        # update treeWidget_settings_settings_harmtree
        self.update_harmonic_tab()

    def update_harmonic_tab(self):
        #print("update_harmonic_tab was called")
        harm = 2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1
        self.settings_harm = harm
        
        self.update_frequencies()

        # update lineEdit_scan_harmsteps
        self.ui.lineEdit_scan_harmsteps.setText(
            str(self.get_harmdata('lineEdit_scan_harmsteps', harm=harm))
        )
        self.load_comboBox(self.ui.comboBox_tracking_method, 'span_mehtod_choose', harm=harm)
        self.load_comboBox(self.ui.comboBox_tracking_condition, 'span_track_choose', harm=harm)
        
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

        # update lineEdit_peaks_threshold
        self.ui.lineEdit_peaks_threshold.setText(
            str(self.get_harmdata('lineEdit_peaks_threshold', harm=harm))
        )

        # update lineEdit_peaks_prominence
        self.ui.lineEdit_peaks_prominence.setText(
            str(self.get_harmdata('lineEdit_peaks_prominence', harm=harm))
        )

    def get_harmdata(self, objname, harm=None):
        '''
        get data with given objname in 
        treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        if harm is None: # use harmonic displayed in UI
            harm = self.settings_harm
        else: # use given harmonic. It is useful for mpl_sp<n> getting params
            pass
        
        try:
            return self.settings['harmdata'][self.active_chn['name']][harm][objname]
        except:
            print(objname, 'is not found!')
            return None

    def set_harmdata(self, objname, val, harm=None):
        '''
        set data with given objname in 
        treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        if harm is None: # use harmonic displayed in UI
            harm = self.settings_harm
        else: # use given harmonic. It is useful for mpl_sp<n> getting params
            pass
        
        try:
            self.settings['harmdata'][self.active_chn['name']][harm][objname] = val
        except:
            print(objname, 'is not found!')

    def update_base_freq(self, base_freq_index):
        self.settings['comboBox_base_frequency'] = self.ui.comboBox_base_frequency.itemData(base_freq_index) # in MHz
        print(self.settings['comboBox_base_frequency'])
        # update freq_range
        self.update_freq_range()
        # check freq_span
        self.check_freq_spans()
        # update freqrency display
        self.update_frequencies()
        # update statusbar
        self.statusbar_f0bw_update()

    def update_bandwidth(self, bandwidth_index):
        self.settings['comboBox_bandwidth'] = self.ui.comboBox_bandwidth.itemData(bandwidth_index) # in MHz
        print(self.settings['comboBox_bandwidth'])
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
        BW = self.settings['comboBox_bandwidth']
        self.ui.label_status_f0BW.setText('{}\u00B1{} MHz'.format(fbase, BW))
        self.ui.label_status_f0BW.setToolTip('base frequency = {} MHz; band width = {} MHz'.format(fbase, BW))

    def update_freq_range(self):
        '''
        update settings['freq_range'] (freq range allowed for scan)
        '''
        fbase = self.settings['comboBox_base_frequency'] * 1e6 # in Hz
        BW = self.settings['comboBox_bandwidth'] * 1e6 # in Hz
        freq_range = {}
        for i in range(1, settings_init['max_harmonic']+2, 2):
            freq_range[i] = np.array([i*fbase-BW, i*fbase+BW])
        self.settings['freq_range'] = freq_range
        print(self.settings['freq_range'])

    def get_freq_span(self, harm=None, chn=None):
        '''
        return freq_span of given harm and chn
        if harm and chn not given, use self.settings
        '''
        if harm is None:
            harm = self.settings_harm
        if chn is None:
            chn = self.active_chn['name']

        return self.settings['freq_span'][chn][harm]

    def set_freq_span(self, span, harm=None, chn=None):
        '''
        set freq_span of given harm and chn
        if harm and chn not given, use self.settings
        span: ndarray of [f1, f2]
        '''
        if harm is None:
            harm = self.settings_harm
        if chn is None:
            chn = self.active_chn['name']

        self.settings['freq_span'][chn][harm] = span

    def check_freq_spans(self):
        '''
        check if settings['freq_span'] (freq span for each harmonic) values in the allowed range self.settings['freq_range']
        '''
        if 'freq_span' in self.settings and self.settings['freq_span']:  # if self.settings['freq_span'] exist
            print('##################\n', self.settings['freq_span'])
            freq_span = {'samp': {}, 'ref': {}}
            for i in range(1, settings_init['max_harmonic']+2, 2):
                freq_span['samp'][i] = self.span_check(i, *self.settings['freq_span']['samp'][i])
                freq_span['ref'][i] = self.span_check(i, *self.settings['freq_span']['ref'][i])

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
        for i in range(1, settings_init['max_harmonic']+2, 2):
            f1, f2 = self.settings['freq_span']['samp'][i] * 1e-6 # in MHz
            f1r, f2r = self.settings['freq_span']['ref'][i] * 1e-6 # in MHz
            if disp_mode == 'centerspan':
                # convert f1, f2 from start/stop to center/span
                f1, f2 = MathModules.converter_startstop_to_centerspan(f1, f2)
                f1r, f2r = MathModules.converter_startstop_to_centerspan(f1r, f2r)
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(MathModules.num2str(f1, precision=12)) # display as MHz
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(MathModules.num2str(f2, precision=12)) # display as MHz
            getattr(self.ui, 'lineEdit_startf' + str(i) + '_r').setText(MathModules.num2str(f1r, precision=12)) # display as MHz
            getattr(self.ui, 'lineEdit_endf' + str(i) + '_r').setText(MathModules.num2str(f2r, precision=12)) # display as MHz
                
        # update start/end in treeWidget_settings_settings_harmtree
        harm = self.settings_harm
        print(harm)
        f1, f2 = self.get_freq_span()
        # Set Start
        self.ui.lineEdit_scan_harmstart.setText(
            MathModules.num2str(f1*1e-6, precision=12)
        )
        # set End
        self.ui.lineEdit_scan_harmend.setText(
            MathModules.num2str(f2*1e-6, precision=12)
        )

    def update_freq_display_mode(self, signal):
        ''' update frequency dispaly in settings_control '''
        print(signal)
        disp_mode = self.settings['comboBox_settings_control_dispmode']
        # disp_mode = self.ui.comboBox_settings_control_dispmode.itemData(signal)

        # set label_settings_control_label1 & label_settings_control_label2
        if disp_mode == 'startstop':
            self.ui.label_settings_control_label1.setText('Start (MHz)')
            self.ui.label_settings_control_label2.setText('End (MHz)')
        elif disp_mode == 'centerspan':
            self.ui.label_settings_control_label1.setText('Center (MHz)')
            self.ui.label_settings_control_label2.setText('Span (MHz)')
        
        self.update_frequencies()
            

    def on_editingfinished_harm_freq(self):
        '''
        update frequency when lineEdit_scan_harmstart or  lineEdit_scan_harmend edited
        '''
        # print(self.sender().objectName())
        harmstart = float(self.ui.lineEdit_scan_harmstart.text()) * 1e6 # in Hz
        harmend = float(self.ui.lineEdit_scan_harmend.text()) * 1e6 # in Hz
        harm=self.settings_harm
        print(harm, harmstart, harmend)
        f1, f2 = self.span_check(harm=harm, f1=harmstart, f2=harmend)
        print(f1, f2)
        self.set_freq_span(np.array([f1, f2]))
        # self.settings['freq_span'][harm] = [harmstart, harmend] # in Hz
        # self.check_freq_spans()
        self.update_frequencies()

    def set_default_freqs(self):
        #NOTUSING
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))

    def update_spanmethod(self, fitmethod_index):
        #NOTUSING
        value = self.ui.comboBox_tracking_method.itemData(fitmethod_index)
        self.set_harmdata('comboBox_tracking_method', value, harm=self.settings_harm)

    def update_spantrack(self, trackmethod_index):
        #NOTUSING
        value = self.ui.comboBox_tracking_condition.itemData(trackmethod_index)
        self.set_harmdata('comboBox_tracking_condition', value, harm=self.settings_harm)

    def update_harmfitfactor(self, harmfitfactor_index):
        #NOTUSING
        self.set_harmdata('comboBox_harmfitfactor', value, harm=self.settings_harm)

    def setvisible_refwidgets(self, value=False):
        '''
        set the visibility of reference related widget
        '''
        # self.ui.widget_settings_control_sf.setVisible(value)
        self.ui.label_settings_control_samp.setVisible(value)
        self.ui.label_settings_control_ref.setVisible(value)
        self.ui.label_settings_control_label1_r.setVisible(value)
        self.ui.label_settings_control_label2_r.setVisible(value)
        for i in range(1, settings_init['max_harmonic']+2, 2):
            getattr(self.ui, 'lineEdit_startf' + str(i) + '_r').setVisible(value)
            getattr(self.ui, 'lineEdit_endf' + str(i) + '_r').setVisible(value)
        self.ui.radioButton_settings_settings_harmchnsamp.setVisible(value)
        self.ui.radioButton_settings_settings_harmchnref.setVisible(value)



    def update_vnachannel(self, index):
        '''
        update vna channels (sample and reference)
        if ref == sample: sample = 'none'
        '''
        sample_channel = self.settings['comboBox_sample_channel']
        ref_channel = self.settings['comboBox_ref_channel']

        if ref_channel == sample_channel:
            # make sure sample and ref channels are not the same
            ref_channel = 'none' # set ref_channel to none
            #TODO update in statusbar
        self.settings['comboBox_sample_channel'] = sample_channel
        self.settings['comboBox_ref_channel'] = ref_channel
        # load_comboBox has to be used after the value saved in self.settings
        self.load_comboBox(self.ui.comboBox_ref_channel, 'ref_channel_choose')
        # set visibility of reference related widgets
        self.setvisible_refwidgets(ref_channel != 'none')

    def update_tempsensor(self):
        print("update_tempsensor was called")
        self.settings['checkBox_settings_temp_sensor'] = not self.settings['checkBox_settings_temp_sensor']


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
        print(self.settings['comboBox_tempdevice'], self.settings['comboBox_thrmcpltype'])
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
        value = self.ui.comboBox_timescale.itemData(timescale_index)
        self.settings['comboBox_timescale'] = value
        #TODO update plt1 and plt2

    def update_yscale(self, yscale_index):
        value = self.ui.comboBox_yscale.itemData(yscale_index)
        self.settings['comboBox_yscale'] = value
        #TODO update plt1 and plt2

    def update_linktime(self):
       self.settings['checkBox_linktime'] = not self.settings['checkBox_linktime']
        # TODO update plt1 and plt2

    def load_comboBox(self, comboBox, choose_dict_name, harm=None):
        '''
        load combobox value from self.settings 
        if harm == None
            set the value of combox from self.settings[comboBox]
        if harm = int
            the combobox is in harmwidget
        '''
        comboBoxName = comboBox.objectName()
        for key in settings_init[choose_dict_name].keys():
            if harm is None: # not embeded in subdict
                if key == self.settings[comboBoxName]:
                    comboBox.setCurrentIndex(comboBox.findData(key))
                    break
            else:
                if key == self.get_harmdata(comboBoxName, harm):
                    comboBox.setCurrentIndex(comboBox.findData(key))
                    break
                

    def update_guichecks(self, checkBox, name_in_settings):
        #NOTUSING
        print("update_guichecks was called")
        checkBox.setChecked(self.get_harmdata(name_in_settings, harm=self.settings_harm))
        
    # debug func
    def log_update(self):
        #NOTUSING
        with open('settings.json', 'w') as f:
            line = json.dumps(dict(self.settings), indent=4) + "\n"
            f.write(line)

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
        self.setWindowTitle(settings_init['window_title'])
        # set window size
        self.resize(*settings_init['window_size'])
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
        # set actived harmonic tab
        # self.settings_harm = 1 #TODO
        # set active_chn
        self.ui.radioButton_settings_settings_harmchnsamp.setChecked(True)

        ## following data is read from self.settings
        # # hide harmonic related widgets which > max_disp_harmonic & < max_harmonic
        # for i in range(self.settings['max_disp_harmonic']+2, settings_init['max_harmonic']+2, 2):
        #     print(i)
        #     getattr(self.ui, 'checkBox_harm' +str(i)).setVisible(False)
        #     getattr(self.ui, 'lineEdit_startf' +str(i)).setVisible(False)
        #     getattr(self.ui, 'lineEdit_endf' +str(i)).setVisible(False)
        #     getattr(self.ui, 'lineEdit_startf' +str(i) + '_r').setVisible(False)
        #     getattr(self.ui, 'lineEdit_endf' +str(i) + '_r').setVisible(False)
        #     getattr(self.ui, 'tab_settings_settings_harm' +str(i)).setVisible(False)
        #     getattr(self.ui, 'checkBox_plt1_h' +str(i)).setVisible(False)
        #     getattr(self.ui, 'checkBox_plt2_h' +str(i)).setVisible(False)
        #     getattr(self.ui, 'tab_settings_data_harm_' +str(i)).setVisible(False)
        #     # more to be added here


        # load display_mode
        self.load_comboBox(self.ui.comboBox_settings_control_dispmode, 'display_choose')

        # load harm state
        for i in range(1, settings_init['max_harmonic']+2, 2):
            # settings/control/Harmonics
            getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(self.settings['checkBox_harm' + str(i)])
            getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(self.settings['checkBox_harm' + str(i)])


        # load default record interval
        self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        # load default spectra refresh resolution
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))
        # update lineEdit_scaninterval
        self.set_lineEdit_scaninterval()

        # load default fitting and display options
        self.ui.checkBox_dynamicfit.setChecked(self.settings['checkBox_dynamicfit'])
        # load default fit factor range
        self.ui.spinBox_fitfactor.setValue(self.settings['spinBox_fitfactor'])
        # load default dynamicfitbyharm
        self.ui.checkBox_dynamicfitbyharm.setChecked(self.settings['checkBox_dynamicfitbyharm'])
        # load default fitfactorbyharm
        self.ui.checkBox_fitfactorbyharm.setChecked(self.settings['checkBox_fitfactorbyharm'])

        # load this first to create self.settings['freq_range'] & self.settings['freq_span']
        self.load_comboBox(self.ui.comboBox_base_frequency, 'base_frequency_choose')
        self.load_comboBox(self.ui.comboBox_bandwidth, 'bandwidth_choose')
        # update statusbar
        self.statusbar_f0bw_update()

        # create self.settings['freq_range']. 
        # this has to be initated before any 
        self.update_freq_range()
        # update self.settings['freq_span']
        self.check_freq_spans()
        # update frequencies display
        self.update_frequencies()

        # load default VNA settings
        self.load_comboBox(self.ui.comboBox_sample_channel, 'sample_channel_choose')
        self.load_comboBox(self.ui.comboBox_ref_channel, 'ref_channel_choose')
        
        # set treeWidget_settings_settings_harmtree display
        self.update_harmonic_tab()

        # load default temperature settings
        self.load_comboBox(self.ui.comboBox_settings_mechanics_selectmodel, 'thrmcpl_choose')

        self.ui.checkBox_settings_temp_sensor.setChecked(self.settings['checkBox_settings_temp_sensor'])

        try:
            self.load_comboBox(self.ui.comboBox_tempdevice, 'tempdevs_choose')
        except:
            pass
        self.load_comboBox(self.ui.comboBox_thrmcpltype, 'thrmcpl_choose')
        # update display on label_temp_devthrmcpl
        self.set_label_temp_devthrmcpl() # this should be after temp_sensor & thrmcpl 

        # load default plots settings
        self.load_comboBox(self.ui.comboBox_timeunit, 'time_unit_choose')
        self.load_comboBox(self.ui.comboBox_tempunit, 'temp_unit_choose')
        self.load_comboBox(self.ui.comboBox_timescale, 'scale_choose')
        self.load_comboBox(self.ui.comboBox_yscale, 'scale_choose')

        self.ui.checkBox_linktime.setChecked(self.settings['checkBox_linktime'])

        # set default displaying of spectra show options
        self.ui.radioButton_spectra_showBp.setChecked(self.settings['radioButton_spectra_showBp'])
        self.ui.radioButton_spectra_showpolar.setChecked(self.settings['radioButton_spectra_showpolar'])
        self.ui.checkBox_spectra_shoechi.setChecked(self.settings['checkBox_spectra_shoechi'])

        # set default displaying of plot 1 options
        self.load_comboBox(self.ui.comboBox_plt1_choice, 'data_plt_choose')

        # set default displaying of plot 2 options
        self.load_comboBox(self.ui.comboBox_plt2_choice, 'data_plt_choose')


    def check_freq_range(self, harmonic, min_range, max_range):
        #NOTUSING
        startname = 'lineEdit_startf' + str(harmonic)
        endname = 'lineEdit_endf' + str(harmonic)
        # check start frequency range
        if float(self.settings[startname]) <= min_range or float(self.settings[startname]) >= max_range:
            print('ERROR')
            self.settings[startname] = float(min_range)
        if float(self.settings[startname]) >= float(self.settings[endname]):
            if float(self.settings[startname]) == float(self.settings[endname]):
                print('The start frequency cannot be the same as the end frequency!')
                self.settings[startname] = min_range
                # self.settings[endname] = max_range
            else:
                print('The start frequency is greater than the end frequency!')
                self.settings[startname] = min_range
        # check end frequency range
        if float(self.settings[endname]) <= min_range or float(self.settings[endname]) >= max_range:
            print('ERROR')
            self.settings[endname] = max_range
        if float(self.settings[endname]) <= float(self.settings[startname]):
            print('ERROR: The end frequency is less than the start frequency!')
            if float(self.settings[startname]) == max_range:
                print('The start frequency cannot be the same as the end frequency!')
                self.settings[startname] = min_range
                # self.settings[endname] = max_range - 0.9
            else:
                self.settings[endname] = max_range

    def smart_peak_tracker(self, harmonic=None, freq=None, conductance=None, susceptance=None, G_parameters=None):
        self.peak_tracker.f0 = G_parameters[0]
        self.peak_tracker.g0 = G_parameters[1]

        track_condition = self.get_harmdata('comboBox_tracking_condition', harmonic) 
        track_method = self.get_harmdata('comboBox_tracking_method', harmonic)
        chn = self.active_chn['name']
        # determine the structure field that should be used to extract out the initial-guessing method
        if track_method == 'bmax':
            resonance = susceptance
        else:
            resonance = conductance
        index = GBFitting.findpeaks(resonance, output='indices', sortstr='descend')
        cen = freq[index[0]] # peak center
        # determine the estimated associated conductance (or susceptance) value at the resonance peak
        Gmax = resonance[index[0]] 
        # determine the estimated half-max conductance (or susceptance) of the resonance peak
        half_amp = (Gmax-np.amin(resonance))/2 + np.amin(resonance) 
        half_wid = np.absolute(freq[np.where(np.abs(half_amp-resonance)==np.min(np.abs(half_amp-resonance)))[0][0]] -  cen)
        current_xlim = self.get_freq_span(harm=harmonic, chn=chn)
        # get the current center and current span of the data in Hz
        current_center, current_span = MathModules.converter_startstop_to_centerspan(current_xlim[0], current_xlim[1])
        # find the starting and ending frequency of only the peak in Hz
        if track_condition == 'fixspan':
            if np.absolute(np.mean(np.array([freq[0],freq[-1]]))-cen) > 0.1 * current_span:
                # new start and end frequencies in Hz
                new_xlim=np.array([cen-0.5*current_span,cen+0.5*current_span])
        elif track_condition == 'fixcenter':
            # peak_xlim = np.array([cen-half_wid*3, cen+half_wid*3])
            if np.sum(np.absolute(np.subtract(current_xlim, np.array([current_center-3*half_wid, current_center + 3*half_wid])))) > 3e3:
                #TODO above should equal to abs(sp - 6 * half_wid) > 3e3
                # set new start and end freq based on the location of the peak in Hz
                new_xlim = np.array(current_center-3*half_wid, current_center+3*half_wid)

        elif track_condition == 'auto':
            # adjust window if neither span or center is fixed (default)
            if(np.mean(current_xlim)-cen) > 1*current_span/12:
                new_xlim = current_xlim - current_span / 15  # new start and end frequencies in Hz
            elif (np.mean(current_xlim)-cen) < -1*current_span/12:
                new_xlim = current_xlim + current_span / 15  # new start and end frequencies in Hz
            else:
                thresh1 = .05 * current_span + current_xlim[0] # Threshold frequency in Hz
                thresh2 = .03 * current_span # Threshold frequency span in Hz
                LB_peak = cen - half_wid * 3 # lower bound of the resonance peak
                if LB_peak - thresh1 > half_wid * 8: # if peak is too thin, zoom into the peak
                    new_xlim[0] = (current_xlim[0] + thresh2) # Hz
                    new_xlim[1] = (current_xlim[1] - thresh2) # Hz
                elif thresh1 - LB_peak > -half_wid*5: # if the peak is too fat, zoom out of the peak
                    new_xlim[0] = current_xlim[0] - thresh2 # Hz
                    new_xlim[1] = current_xlim[1] + thresh2 # Hz
        elif track_condition == 'fixcntspn':
            # bothe span and cent are fixed
            # no changes
            return
        elif track_condition == 'usrdef': #run custom tracking algorithm
            ### CUSTOM, USER-DEFINED
            ### CUSTOM, USER-DEFINED
            ### CUSTOM, USER-DEFINED
            return

        # set new start/end freq in Hz
        self.set_freq_span(new_xlim, harm= harmonic, chn=chn)
        self.check_freq_spans()
        self.update_frequencies()
    
    def read_scan(self, harmonic):
        #NOTUSING
        # read in live data scans
        if self.peak_tracker.refit_flag == 0:
            flag = 0
            rawdata = np.array([])
            start1 = self.settings['lineEdit_startf' + str(harmonic)]
            end1 = self.settings['lineEdit_endf' + str(harmonic)]
            if harmonic < 11:
                rawfile = 'myVNAdata0' + str(harmonic) + '.csv'
            else:
                rawfile = 'myVNAdata11.csv'
            while flag == 0:
                with open(rawfile, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        np.append(rawdata, row[0])
                num_pts = self.get_harmdata('lineEdit_scan_harmsteps', harm=self.settings_harm)
                if len(rawdata) == num_pts*2:
                    self.Peak_tracker.G = 1e3 * rawdata[:num_pts+1]
                    self.peak_tracker.B = 1e3 * rawdata[num_pts:]
                    self.peak_tracker.f = np.arange(start1,end1-(end1-start1)/num_pts+1,(end1-start1)/num_pts)
                    flag = 1
                    print('Status: Scan successful.')
        #TODO refit loaded raw spectra data
        else:
            pass
    def data_collection(self):
        '''
        data collecting routine
        '''
        print(datetime.datetime.now())



#endregion



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())

