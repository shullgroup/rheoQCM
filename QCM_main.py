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

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QGridLayout, QLineEdit, QCheckBox, QComboBox, QRadioButton, QMenu
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.uic import loadUi

# packages
from MainWindow import Ui_MainWindow
from UISettings import settings_init, settings_default
from modules import UIModules, MathModules

if UIModules.system_check() == 'win32': # windows
    try:
        from modules.AccessMyVNA_dump import AccessMyVNA
        print(AccessMyVNA)
        # test if MyVNA program is available
        with AccessMyVNA() as accvna:
            if accvna.Init() == 0: # connection with myVNA is available
                from modules import tempDevices
    except Exception as e: # no myVNA connected. Analysis only
        print('Failed to import AccessMyVNA module!')
        print(e)

from MatplotlibWidget import MatplotlibWidget

class PeakTracker:
    def __init__(self):
        self.f0 = None
        self.gamma0 = None
        self.freq = None
        self.conductance = None
        self.susceptance = None
        self.refit_flag = 0
        self.refit_counter = 1

        self.harmonic_tab = 1

class QCMApp(QMainWindow):
    '''
    The settings of the app is stored in a dict
    '''
    def __init__(self):
        super(QCMApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fileName = ''
        self.fileFlag = False
        self.settings = settings_default
        self.peak_tracker = PeakTracker()
  
        # define instrument state variables
        self.accvna = None 
        self.idle = True # if test is running
        self.reading = False # if myVNA is scanning and reading data
        self.tempsensor = None # class for temp sensor
        
        # check system
        self.system = UIModules.system_check()
        # initialize AccessMyVNA
        #TODO add more code to disable settings_control tab and widges in settings_settings tab
        if self.system == 'win32': # windows
            try:
                # test if MyVNA program is available
                with AccessMyVNA() as accvna:
                    if accvna.Init() == 0: # is available
                        self.accvna = AccessMyVNA() # save class AccessMyVNA to accvna
                    else: # not available
                        pass
            except:
                pass

        else: # other system, data analysis only
            pass
        print(self.accvna)
        self.main()
        self.load_settings()


    def main(self):
 # loadUi('QCM_GUI_test4.ui', self) # read .ui file directly. You still need to compile the .qrc file
#region ###### initiate UI #################################

#region main UI 
        # set window title
        #self.setWindowTitle(settings_init['window_title'])
        # set window size
        #self.resize(*settings_init['window_size'])
        # set delault displaying of tab_settings
        #self.ui.tabWidget_settings.setCurrentIndex(0)
        # set delault displaying of stackedWidget_spectra
        #self.ui.stackedWidget_spectra.setCurrentIndex(0)
        # set delault displaying of stackedWidget_data
        #self.ui.stackedWidget_data.setCurrentIndex(0)

        # set delault displaying of harmonics
        #self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)

        # link tabWidget_settings and stackedWidget_spectra and stackedWidget_data
        self.ui.tabWidget_settings.currentChanged.connect(self.link_tab_page)

#endregion


#region cross different sections
        # harmonic widgets
        # loop for setting harmonics 
        for i in range(1, settings_init['max_harmonic']+2, 2):
            # set to visable which is default. nothing to do

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
            getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(
                getattr(self.ui, 'checkBox_harm' + str(i)).setChecked
            )
            getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(
                getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked
            )
            getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(
                getattr(self.ui, 'frame_sp' +str(i)).setVisible
            )
            getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(
                getattr(self.ui, 'frame_sp' +str(i)).setVisible
            )


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

        # set pushButton_resetreftime
        self.ui.pushButton_resetreftime.clicked.connect(self.reset_reftime)

        # set lineEdit_scaninterval value
        self.ui.lineEdit_recordinterval.textEdited.connect(self.set_lineEdit_scaninterval)
        self.ui.lineEdit_refreshresolution.textEdited.connect(self.set_lineEdit_scaninterval)

        # add value to the comboBox_settings_control_scanmode
        for key, val in settings_init['scan_mode'].items():
            self.ui.comboBox_settings_control_scanmode.addItem(val, key)

        # add value to the comboBox_fitfactor
        for key, val in settings_init['fit_factor_choose'].items():
            self.ui.comboBox_fitfactor.addItem(val, key)

        # set pushButton_gotofolder
        self.ui.pushButton_gotofolder.clicked.connect(self.on_clicked_pushButton_gotofolder)

        # set pushButton_newdata
        self.ui.pushButton_newdata.clicked.connect(self.on_triggered_new_data)

        # set pushButton_appenddata
        self.ui.pushButton_appenddata.clicked.connect(self.on_triggered_load_data)

        # set signals to update lineEdit_start settings
        self.ui.lineEdit_startf1.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_startf3.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_startf5.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_startf7.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_startf9.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_startf11.textChanged[str].connect(self.update_widget)

        # set signals to update lineEdit_end settings
        self.ui.lineEdit_endf1.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_endf3.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_endf5.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_endf7.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_endf9.textChanged[str].connect(self.update_widget)
        self.ui.lineEdit_endf11.textChanged[str].connect(self.update_widget)

        # set lineEdit_scaninterval background
        self.ui.lineEdit_scaninterval.setStyleSheet(
            "QLineEdit { background: transparent; }"
        )

        # set signals to update fitting and display settings
        self.ui.checkBox_dynamicfit.stateChanged.connect(self.update_widget)
        #self.ui.checkBox_showsusceptance.stateChanged.connect(self.update_widget)
        #self.ui.checkBox_showchi.stateChanged.connect(self.update_widget)
        #self.ui.checkBox_showpolar.stateChanged.connect(self.update_widget)
        self.ui.comboBox_fitfactor.activated.connect(self.update_widget)

        # set signals to update spectra show display options
        self.ui.radioButton_spectra_showBp.toggled.connect(self.update_widget)
        #self.ui.radioButton_spectra_showpolar.toggled.connect(self.update_widget)
        self.ui.checkBox_spectra_shoechi.toggled.connect(self.update_widget)

        # set signals to checkBox_control_rectemp
        self.ui.checkBox_control_rectemp.clicked['bool'].connect(self.on_clicked_set_temp_sensor)

#endregion


#region settings_settings
        ### add combobox into treewidget
        # comboBox_fit_method
        self.create_combobox(
            'comboBox_fit_method', 
            settings_init['span_mehtod_choose'], 
            100, 
            'Method', 
            self.ui.treeWidget_settings_settings_harmtree
        )

        # add track_method
        self.create_combobox(
            'comboBox_track_method', 
            settings_init['track_mehtod_choose'], 
            100, 
            'Tracking', 
            self.ui.treeWidget_settings_settings_harmtree
        )

        # add fit factor
        self.create_combobox(
            'comboBox_harmfitfactor', 
            settings_init['fit_factor_choose'], 
            100, 
            'Factor', 
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

        # add comBox_tempdevice to treeWidget_settings_settings_hardware
        if self.accvna:
            self.create_combobox(
                'comBox_tempdevice',
                tempDevices.dict_available_devs(settings_init['devices_dict']),  
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware, 
            )
            self.settings['comBox_tempdevice'] = self.ui.comBox_tempdevice.itemData(self.ui.comBox_tempdevice.currentIndex())
            self.ui.comBox_tempdevice.activated.connect(self.update_widget)
        else: # accvna is not available
            self.create_combobox(
                'comBox_tempdevice',
                [],  # an empty list
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware, 
            )
            self.settings['comBox_tempdevice'] = None # set to None 

        # insert thrmcpl type
        self.create_combobox(
            'comboBox_thrmcpltype', 
            settings_init['thrmcpl_choose'], 
            100, 
            'Thrmcpl Type', 
            self.ui.treeWidget_settings_settings_hardware
        )

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
            settings_init['time_scale_choose'], 
            100, 
            'Time Scale', 
            self.ui.treeWidget_settings_settings_plots
        )

        # insert gamma scale
        self.create_combobox(
            'comboBox_gammascale', 
            settings_init['gamma_scale_choose'], 
            100, 
            'Γ Scale', 
            self.ui.treeWidget_settings_settings_plots
        )

        # move checkBox_settings_settings_linktime to treeWidget_settings_settings_plots
        self.move_to_col2(
            self.ui.checkBox_settings_settings_linktime, 
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

        # set signals to update scan and crystal settings_settings
        self.ui.tabWidget_settings_settings_harm.currentChanged.connect(self.update_harmonic_tab)
        self.ui.comboBox_base_frequency.activated.connect(self.update_base_freq)
        self.ui.comboBox_bandwidth.activated.connect(self.update_bandwidth)

        # set signals to update span settings_settings
        self.ui.comboBox_fit_method.activated.connect(self.update_fitmethod)
        self.ui.comboBox_track_method.activated.connect(self.update_trackmethod)

        self.ui.comboBox_harmfitfactor.activated.connect(self.update_harmfitfactor)

        # set signals to update fit settings_settings
        self.ui.comboBox_sample_channel.activated.connect(self.update_samplechannel)
        self.ui.comboBox_ref_channel.activated.connect(self.update_refchannel)

        # set signals to update temperature settings_settings
        self.ui.comboBox_settings_mechanics_selectmodel.activated[str].connect(self.update_module)
        # self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.update_tempsensor)
        self.ui.checkBox_settings_temp_sensor.clicked['bool'].connect(self.on_clicked_set_temp_sensor)
        self.ui.comboBox_thrmcpltype.activated.connect(self.update_thrmcpltype)

        # set signals to update plots settings_settings
        self.ui.comboBox_timeunit.activated.connect(self.update_timeunit)
        self.ui.comboBox_tempunit.activated.connect(self.update_tempunit)
        self.ui.comboBox_timescale.activated.connect(self.update_timescale)
        self.ui.comboBox_gammascale.activated.connect(self.update_gammascale)
        self.ui.checkBox_settings_settings_linktime.stateChanged.connect(self.update_widget)
        
        # set default values
        self.ui.comboBox_base_frequency.setCurrentIndex(0)
        self.ui.comboBox_bandwidth.setCurrentIndex(4)
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)
 
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

        # pushButton_spectra_fit_refresh
        self.ui.pushButton_spectra_fit_refresh.clicked.connect(self.on_clicked_pushButton_spectra_fit_refresh)

#endregion


#region spectra_mechanics


#endregion


#region data_data
        # set signals to update plot 1 options
        self.ui.comboBox_plt1_choice.activated.connect(self.update_widget)
        self.ui.checkBox_plt1_h1.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt1_h3.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt1_h5.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt1_h7.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt1_h9.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt1_h11.stateChanged.connect(self.update_widget)
        self.ui.radioButton_plt1_ref.toggled.connect(self.update_widget)
        self.ui.radioButton_plt1_samp.toggled.connect(self.update_widget)

        # set signals to update plot 2 options
        self.ui.comboBox_plt2_choice.activated.connect(self.update_widget)
        self.ui.checkBox_plt2_h1.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt2_h3.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt2_h5.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt2_h7.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt2_h9.stateChanged.connect(self.update_widget)
        self.ui.checkBox_plt2_h11.stateChanged.connect(self.update_widget)
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
        self.ui.actionLoad_Data.triggered.connect(self.on_triggered_load_data)
        self.ui.actionNew_Data.triggered.connect(self.on_triggered_new_data)
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
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit_polar.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit_polar))

        # add figure mpl_spectra_fit into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            axtype='sp_fit',
            showtoolbar=('Back', 'Forward', 'Pan', 'Zoom')
            )
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))

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

        for i in range(1, settings_init['max_harmonic']+2, 2):
            if i in self.settings['harmonics_check']: # in the default range 
                # settings/control/Harmonics
                getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(True)
                getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(True)

            else: # out of the default range
                getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(False)
                getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(False)
                # hide spectra/sp
                getattr(self.ui, 'frame_sp' + str(i)).setVisible(False)


        self.ui.comboBox_plt1_choice.setCurrentIndex(2)
        self.ui.comboBox_plt2_choice.setCurrentIndex(3)

        # set time interval
        self.ui.lineEdit_scaninterval.setText(str(self.settings['lineEdit_scaninterval']))
        self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))

#endregion


#region #########  functions ##############

    def link_tab_page(self, tab_idx):
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
        item = parent.findItems(row_text, Qt.MatchExactly | Qt.MatchRecursive, 0)
        if len(item) == 1:
            item = item[0]
        else:
            return
        # insert the combobox in to the 2nd column of row_text
        parent.setItemWidget(item, 1, obj)        

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
            self.ui.pushButton_runstop.setText('STOP')
        else:
            self.ui.pushButton_runstop.setText('START RECORD')


    # @pyqtSlot()
    def reset_reftime(self):
        ''' set time in lineEdit_reftime '''
        current_time = datetime.datetime.now()
        self.ui.lineEdit_reftime.setText(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        # update reftime in settings dict
        self.settings['lineEdit_reftime'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # @pyqtSlot()
    def set_lineEdit_scaninterval(self):
        # get text
        record_interval = self.ui.lineEdit_recordinterval.text()
        refresh_resolution = self.ui.lineEdit_refreshresolution.text()
        #convert to flot
        try:
            record_interval = float(record_interval)
        except:
            record_interval = 0
        try:
            refresh_resolution = float(refresh_resolution)
        except:
            refresh_resolution = 0
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

    def on_triggered_new_data(self):
        fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        if fileName:
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(fileName)
            # reset lineEdit_reftime
            self.reset_reftime()
            # set lineEdit_reftime editabled and enable pushButton_resetreftime
            self.ui.lineEdit_reftime.setReadOnly(False)
            self.ui.pushButton_resetreftime.setEnabled(True)
            self.fileName = fileName

    def on_triggered_load_data(self): 
        fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
        if fileName:
            # change the displayed file directory in lineEdit_datafilestr
            self.ui.lineEdit_datafilestr.setText(self.fileName)
            # set lineEdit_reftime
            # set lineEdit_reftime read only and disable pushButton_resetreftime
            self.ui.lineEdit_reftime.setReadOnly(True)
            # ??  set reftime in fileName to lineEdit_reftime

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

        # set span

        # start a single scan

        # set span text

        # reset slider to 1
        self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)

    def on_clicked_pushButton_spectra_fit_refresh(self):
        print('accvna', self.accvna)
        # get parameters from current setup: harm_tab
        with self.accvna as accvna:
            accvna.set_steps_freq()
            ret, f, G, B = accvna.single_scan()

        # f = G = B = range(10)
        self.ui.mpl_spectra_fit.update_data(ls=['lG'], xdata=[f], ydata=[G])
        self.ui.mpl_spectra_fit.update_data(ls=['lB'], xdata=[f], ydata=[B])

        self.ui.mpl_spectra_fit_polar.update_data(ls=['l'], xdata=[G], ydata=[B])


    def on_clicked_set_temp_sensor(self, checked):
        # below only runs when accvna is available
        if self.accvna: # add not for testing code    
            if checked: # checkbox is checked
                # if not self.tempsensor: # tempModule is not initialized 
                # get all tempsensor settings 
                tempmodule_name = self.settings['comboBox_tempmodule'] # get temp module

                thrmcpltype = self.settings['comboBox_thrmcpltype'] # get thermocouple type
                tempdevice = tempDevices.device_info(self.settings['comBox_tempdevice']) #get temp device info

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
                    self.ui.comBox_tempdevice.setEnabled(False)
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
                self.ui.comBox_tempdevice.setEnabled(True)
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
        # if the sender of the signal isA QLineEdit object, update QLineEdit vals in dict
        if isinstance(self.sender(), QLineEdit):
                try:
                    if 'lineEdit_startf' in self.sender().objectName() or 'lineEdit_endf' in self.sender().objectName():
                        self.settings[self.sender().objectName()] = float(signal)*1e6
                    else:
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
        #print(self.sender().objectName(), self.settings[self.sender().objectName()])
    #def update_dynamicfit(self):
    #    self.settings['checkBox_dynamicfit'] = not self.settings['checkBox_dynamicfit']

    #def update_showsusceptance(self):
    #    self.settings['checkBox_showsusceptance'] = not self.settings['checkBox_showsusceptance']

    #def update_showchi(self):
    #    self.settings['checkBox_showchi'] = not self.settings['checkBox_showchi']

    #def update_showpolarplot(self):
    #    self.settings['checkBox_showpolar'] = not self.settings['checkBox_showpolar']

    def update_harmonic_tab(self):
        #print("update_harmonic_tab was called")
        self.peak_tracker.harmonic_tab = 2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1
        self.update_frequencies()

        #self.update_guichecks(self.ui.checkBox_settings_temp_sensor, 'checkBox_settings_temp_sensor')
        #self.update_guichecks(self.ui.checkBox_settings_settings_linktime, 'checkBox_settings_settings_linktime')

        self.update_guicombos(self.ui.comboBox_fit_method, 'comboBox_fit_method', 'span_mehtod_choose')
        self.update_guicombos(self.ui.comboBox_track_method, 'comboBox_track_method', 'track_mehtod_choose')
        self.update_guicombos(self.ui.comboBox_harmfitfactor, 'comboBox_harmfitfactor', 'fit_factor_choose')
        #self.update_guicombos(self.ui.comboBox_sample_channel, 'comboBox_sample_channel', 'sample_channel_choose')
        #self.update_guicombos(self.ui.comboBox_ref_channel, 'comboBox_ref_channel', 'ref_channel_choose')
        #self.update_guicombos(self.ui.comboBox_thrmcpltype, 'comboBox_thrmcpltype', 'thrmcpl_choose')
        #self.update_guicombos(self.ui.comboBox_timeunit, 'comboBox_timeunit', 'time_unit_choose')
        #self.update_guicombos(self.ui.comboBox_tempunit, 'comboBox_tempunit', 'temp_unit_choose')
        #self.update_guicombos(self.ui.comboBox_timescale, 'comboBox_timescale', 'time_scale_choose')
        #self.update_guicombos(self.ui.comboBox_gammascale, 'comboBox_gammascale', 'gamma_scale_choose')
        
        #self.update_guicombos(self.ui.comboBox_bandwidth, 'comboBox_bandwidth', 'bandwidth_choose')
        #self.update_guicombos(self.ui.comboBox_base_frequency, 'comboBox_base_frequency', 'base_frequency_choose')

    def update_base_freq(self, base_freq_index):
        value = self.ui.comboBox_base_frequency.itemData(base_freq_index)
        self.settings['comboBox_base_frequency'] = value
        self.update_frequencies()

    def update_bandwidth(self, bandwidth_index):
        value = self.ui.comboBox_bandwidth.itemData(bandwidth_index)
        self.settings['comboBox_bandwidth'] = value
        self.update_frequencies()

    def update_frequencies(self):
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['start_freq'] = self.peak_tracker.harmonic_tab * self.settings['comboBox_base_frequency'] - self.settings['comboBox_bandwidth']
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['end_freq'] = self.peak_tracker.harmonic_tab * self.settings['comboBox_base_frequency'] + self.settings['comboBox_bandwidth']
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(0).setText(1, str(self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['start_freq']))
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(1).setText(1, str(self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['end_freq']))

    def set_default_freqs(self):
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))

    def update_fitmethod(self, fitmethod_index):
        value = self.ui.comboBox_fit_method.itemData(fitmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['comboBox_fit_method'] = value

    def update_trackmethod(self, trackmethod_index):
        value = self.ui.comboBox_track_method.itemData(trackmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['comboBox_track_method'] = value

    def update_harmfitfactor(self, harmfitfactor_index):
        value = self.ui.comboBox_harmfitfactor.itemData(harmfitfactor_index)
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['comboBox_harmfitfactor'] = value

    def update_samplechannel(self, samplechannel_index):
        value = self.ui.comboBox_sample_channel.itemData(samplechannel_index)
        self.settings['comboBox_sample_channel'] = value

    def update_refchannel(self, refchannel_index):
        value = self.ui.comboBox_ref_channel.itemData(refchannel_index)
        self.settings['comboBox_ref_channel'] = value

    def update_module(self, module_text):
        self.settings['comboBox_settings_mechanics_selectmodel'] = module_text

    def update_tempsensor(self):
        print("update_tempsensor was called")
        self.settings['checkBox_settings_temp_sensor'] = not self.settings['checkBox_settings_temp_sensor']

    def update_thrmcpltype(self, thrmcpltype_index):
        value = self.ui.comboBox_thrmcpltype.itemData(thrmcpltype_index)
        self.settings['comboBox_thrmcpltype'] = value

    def update_timeunit(self, timeunit_index):
        value = self.ui.comboBox_timeunit.itemData(timeunit_index)
        self.settings['comboBox_timeunit'] = value

    def update_tempunit(self, tempunit_index):
        value = self.ui.comboBox_tempunit.itemData(tempunit_index)
        self.settings['comboBox_tempunit'] = value

    def update_timescale(self, timescale_index):
        value = self.ui.comboBox_timescale.itemData(timescale_index)
        self.settings['comboBox_timescale'] = value

    def update_gammascale(self, gammascale_index):
        value = self.ui.comboBox_gammascale.itemData(gammascale_index)
        self.settings['comboBox_gammascale'] = value

    #def update_linktime(self):
    #    self.settings['checkBox_settings_settings_linktime'] = not self.settings['checkBox_settings_settings_linktime']

    def update_guicombos(self, comboBox, name_in_settings, init_dict):
        for key, val in settings_init[init_dict].items():
            if key == self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)][name_in_settings]:
                comboBox.setCurrentIndex(comboBox.findData(key))
                break

    def update_guichecks(self, checkBox, name_in_settings):
        print("update_guichecks was called")
        checkBox.setChecked(self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)][name_in_settings])
        
    # debug func
    def log_update(self):
        with open('settings.json', 'w') as f:
            line = json.dumps(dict(self.settings), indent=4) + "\n"
            f.write(line)

    def load_settings(self):
        # set window title
        self.setWindowTitle(settings_init['window_title'])
        # set window size
        self.resize(*settings_init['window_size'])
        # set delault displaying of tab_settings
        self.ui.tabWidget_settings.setCurrentIndex(0)
        # set delault displaying of stackedWidget_spetratop
        self.ui.stackedWidget_spetratop.setCurrentIndex(0)
        # set delault displaying of stackedWidget_spectra
        self.ui.stackedWidget_spectra.setCurrentIndex(0)
        # set delault displaying of stackedWidget_data
        self.ui.stackedWidget_data.setCurrentIndex(0)

        # load default start and end frequencies for lineEdit harmonics
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(MathModules.num2str(self.settings['lineEdit_startf' + str(i)]*1e-6, precision=12)) # display as MHz
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(MathModules.num2str(self.settings['lineEdit_endf' + str(i)]*1e-6, precision=12)) # display as MHz
        # load default record interval
        self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        # load default spectra refresh resolution
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))
        # load default fitting and display options
        self.ui.checkBox_dynamicfit.setChecked(self.settings['checkBox_dynamicfit'])
        #self.ui.checkBox_showsusceptance.setChecked(self.settings['checkBox_showsusceptance'])
        #self.ui.checkBox_showchi.setChecked(self.settings['checkBox_showchi'])
        #self.ui.checkBox_showpolar.setChecked(self.settings['checkBox_showpolar'])

        def load_comboBox(comboBox, comboBoxName, choose_dict):
            for key, val in settings_init[choose_dict].items():
                if key == self.settings[comboBoxName]:
                    comboBox.setCurrentIndex(comboBox.findData(key))
                    break

        # load default fit factor range
        load_comboBox(self.ui.comboBox_fitfactor, 'comboBox_fitfactor', 'fit_factor_choose')

        # load default VNA settings
        load_comboBox(self.ui.comboBox_sample_channel, 'comboBox_sample_channel', 'sample_channel_choose')
        load_comboBox(self.ui.comboBox_ref_channel, 'comboBox_ref_channel', 'ref_channel_choose')
        # load default crystal settings
        load_comboBox(self.ui.comboBox_base_frequency, 'comboBox_base_frequency', 'base_frequency_choose')
        load_comboBox(self.ui.comboBox_bandwidth, 'comboBox_bandwidth', 'bandwidth_choose')
        # load default temperature settings
        self.ui.comboBox_settings_mechanics_selectmodel.setCurrentIndex\
        (self.ui.comboBox_settings_mechanics_selectmodel.findText(self.settings['comboBox_settings_mechanics_selectmodel']))
        self.ui.checkBox_settings_temp_sensor.setChecked(self.settings['checkBox_settings_temp_sensor'])
        load_comboBox(self.ui.comboBox_thrmcpltype, 'comboBox_thrmcpltype', 'thrmcpl_choose')
        # load default plots settings
        load_comboBox(self.ui.comboBox_timeunit, 'comboBox_timeunit', 'time_unit_choose')
        load_comboBox(self.ui.comboBox_tempunit, 'comboBox_tempunit', 'temp_unit_choose')
        load_comboBox(self.ui.comboBox_timescale, 'comboBox_timescale', 'time_scale_choose')
        load_comboBox(self.ui.comboBox_gammascale, 'comboBox_gammascale', 'gamma_scale_choose')
        self.ui.checkBox_settings_settings_linktime.setChecked(self.settings['checkBox_settings_settings_linktime'])

        # set opened harmonic tab
        self.peak_tracker.harmonic_tab = 1
        # set tab displayed to harm 1
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)
        self.update_harmonic_tab()

        # set default displaying of spectra show options
        self.ui.radioButton_spectra_showBp.setChecked(self.settings['radioButton_spectra_showBp'])
        self.ui.radioButton_spectra_showpolar.setChecked(self.settings['radioButton_spectra_showpolar'])
        self.ui.checkBox_spectra_shoechi.setChecked(self.settings['checkBox_spectra_shoechi'])

        # set default displaying of plot 1 options
        for key, val in settings_init['data_plt_choose'].items():
            if key == self.settings['comboBox_plt1_choice']:
                self.ui.comboBox_plt1_choice.setCurrentIndex(self.ui.comboBox_plt1_choice.findData(key))
                break
        self.ui.checkBox_plt1_h1.setChecked(self.settings['checkBox_plt1_h1'])
        self.ui.checkBox_plt1_h3.setChecked(self.settings['checkBox_plt1_h3'])
        self.ui.checkBox_plt1_h5.setChecked(self.settings['checkBox_plt1_h5'])
        self.ui.checkBox_plt1_h7.setChecked(self.settings['checkBox_plt1_h7'])
        self.ui.checkBox_plt1_h9.setChecked(self.settings['checkBox_plt1_h9'])
        self.ui.checkBox_plt1_h11.setChecked(self.settings['checkBox_plt1_h11'])
        self.ui.radioButton_plt1_samp.setChecked(self.settings['radioButton_plt1_samp'])
        self.ui.radioButton_plt1_ref.setChecked(self.settings['radioButton_plt1_ref'])

        # set default displaying of plot 2 options
        for key, val in settings_init['data_plt_choose'].items():
            if key == self.settings['comboBox_plt2_choice']:
                self.ui.comboBox_plt2_choice.setCurrentIndex(self.ui.comboBox_plt2_choice.findData(key))
                break
        self.ui.checkBox_plt2_h1.setChecked(self.settings['checkBox_plt2_h1'])
        self.ui.checkBox_plt2_h3.setChecked(self.settings['checkBox_plt2_h3'])
        self.ui.checkBox_plt2_h5.setChecked(self.settings['checkBox_plt2_h5'])
        self.ui.checkBox_plt2_h7.setChecked(self.settings['checkBox_plt2_h7'])
        self.ui.checkBox_plt2_h9.setChecked(self.settings['checkBox_plt2_h9'])
        self.ui.checkBox_plt2_h11.setChecked(self.settings['checkBox_plt2_h11'])
        self.ui.radioButton_plt2_samp.setChecked(self.settings['radioButton_plt2_samp'])
        self.ui.radioButton_plt2_ref.setChecked(self.settings['radioButton_plt2_ref'])

    def check_freq_range(self, harmonic, min_range, max_range):
        startname = 'lineEdit_startf' + str(harmonic)
        endname = 'lineEdit_endf' + str(harmonic)
        # check start frequency range
        if float(self.settings[startname]) <= min_range or float(self.settings[startname]) >= max_range:
            print('ERROR')
            self.settings[startname] = float(min_range) + 0.9
        if float(self.settings[startname]) >= float(self.settings[endname]):
            if float(self.settings[startname]) == float(self.settings[endname]):
                print('The start frequency cannot be the same as the end frequency!')
                self.settings[startname] = min_range + 0.9
                self.settings[endname] = max_range - 0.9
            else:
                print('The start frequency is greater than the end frequency!')
                self.settings[startname] = min_range + 0.9
        # check end frequency range
        if float(self.settings[endname]) <= min_range or float(self.settings[endname]) >= max_range:
            print('ERROR')
            self.settings[endname] = max_range - 0.9
        if float(self.settings[endname]) <= float(self.settings[startname]):
            print('ERROR: The end frequency is less than the start frequency!')
            if float(self.settings[startname]) == max_range:
                print('The start frequency cannot be the same as the end frequency!')
                self.settings[startname] = min_range + 0.9
                self.settings[endname] = max_range - 0.9
            else:
                self.settings[endname] = max_range - 0.9

    def smart_peak_tracker(self, harmonic, freq, conductance, susceptance, G_parameters):
        self.peak_tracker.f0 = G_parameters[0]
        self.peak_tracker.gamma0 = G_parameters[1]

        # determine the structure field that should be used to extract out the initial-guessing method
        if self.settings['tab_settings_settings_harm' + str(harmonic)]['comboBox_fit_method'] == 'bmax':
            resonance = susceptance
        else:
            resonance = conductance
        index = findpeaks(resonance, output='indices', sortstr='descend')
        peak_f = freq[index[0]]
        # determine the estimated associated conductance (or susceptance) value at the resonance peak
        Gmax = resonance[index[0]] 
        # determine the estimated half-max conductance (or susceptance) of the resonance peak
        halfg = (Gmax-np.amin(resonance))/2 + np.amin(resonance) 
        halfg_freq = np.absolute(freq[np.where(np.abs(halfg-resonance)==np.min(np.abs(halfg-resonance)))[0][0]])
        # extract the peak tracking conditions
        track_method = self.settings['tab_settings_settings_harm' + str(harmonic)]['comboBox_track_method'] 
        if track_method == 'fixspan':
            current_span = (float(self.settings['lineEdit_endf' + str(harmonic)]) - \
            # get the current span of the data in Hz
            float(self.settings['lineEdit_startf' + str(harmonic)]))
            if np.absolute(np.mean(np.array([freq[0],freq[len(freq)-1]]))-peak_f) > 0.1 * current_span:
                # new start and end frequencies in Hz
                new_xlim=np.array([peak_f-0.5*current_span,peak_f+0.5*current_span])
                self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0]
                self.settings['lineEdit_endf' + str(harmonic)] = new_xlim[1]
        elif track_method == 'fixcenter':
            # get current start and end frequencies of the data in Hz
            current_xlim = np.array([float(self.settings['lineEdit_startf' + str(harmonic)]),float(self.settings['lineEdit_endf' + str(harmonic)])])
            # get the current center of the data in Hz
            current_center = (float(self.settings['lineEdit_startf' + str(harmonic)]) + float(self.settings['lineEdit_endf' + str(harmonic)]))/2 
            # find the starting and ending frequency of only the peak in Hz
            peak_xlim = np.array([peak_f-halfg_freq*3, peak_f+halfg_freq*3]) 
            if np.sum(np.absolute(np.subtract(current_xlim, np.array([current_center-3*halfg_freq, current_center + 3*halfg_freq])))) > 3e3:
                # set new start and end freq based on the location of the peak in Hz
                new_xlim = np.array(current_center-3*halfg_freq, current_center+3*halfg_freq)
                # set new start freq in Hz
                self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0]
                # set new end freq in Hz
                self.settings['lineEdit_endf' + str(harmonic)] = new_xlim[1]
        elif track_method == 'fixrange':
            # adjust window if neither span or center is fixed (default)
            current_xlim = np.array([float(self.settings['lineEdit_startf' + str(harmonic)]),float(self.settings['lineEdit_endf' + str(harmonic)])])
            # get the current span of the data in Hz
            current_span = float(self.settings['lineEdit_endf' + str(harmonic)]) - float(self.settings['lineEdit_startf' + str(harmonic)])
            if(np.mean(current_xlim)-peak_f) > 1*current_span/12:
                new_xlim = current_xlim-current_span/15  # new start and end frequencies in Hz
                self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0] # set new start freq in Hz
                self.settings['lineEdit_endf' + str(harmonic)] = new_xlim[1] # set new end freq in Hz
            elif (np.mean(current_xlim)-peak_f) < -1*current_span/12:
                new_xlim = current_xlim+current_span/15  # new start and end frequencies in Hz
                self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0] # set new start freq in Hz
                self.settings['lineEdit_endf' + str(harmonic)] = new_xlim[1] # set new end freq in Hz
            else:
                thresh1=.05*current_span + current_xlim[0] # Threshold frequency in Hz
                thresh2=.03*current_span # Threshold frequency span in Hz
                LB_peak=peak_f-halfg_freq*3 # lower bound of the resonance peak
                if LB_peak-thresh1 > halfg_freq*8: # if peak is too thin, zoom into the peak
                    new_xlim[0]=(current_xlim[0] + thresh2) # Hz
                    new_xlim[1]=(current_xlim[1] - thresh2) # Hz
                    self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0] # set new start freq in Hz
                    self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[1] # set new end freq in Hz
                elif thresh1-LB_peak > -halfg_freq*5: # if the peak is too fat, zoom out of the peak
                    new_xlim[0]=current_xlim[0]-thresh2 # Hz
                    new_xlim[1]=current_xlim[1]+thresh2 # Hz
                    self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[0] # set new start freq in Hz
                    self.settings['lineEdit_startf' + str(harmonic)] = new_xlim[1] # set new end freq in Hz
        elif track_method == 'usrdef': #run custom tracking algorithm
            ### CUSTOM, USER-DEFINED
            ### CUSTOM, USER-DEFINED
            ### CUSTOM, USER-DEFINED
            pass
        self.check_freq_range(harmonic, self.settings['freq_range'][harmonic][0], self.settings['freq_range'][harmonic][1])
    
    def read_scan(self, harmonic):
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
                num_pts = self.settings['tab_settings_settings_harm' + str(harmonic)][num_datapoints]
                if len(rawdata) == num_pts*2:
                    self.peak_tracker.conductance = 1e3 * rawdata[:num_pts+1]
                    self.peak_tracker.susceptance = 1e3 * rawdata[num_pts:]
                    self.peak_tracker.freq = np.arange(start1,end1-(end1-start1)/num_pts+1,(end1-start1)/num_pts)
                    flag = 1
                    print('Status: Scan successful.')
        # TODO refit loaded raw spectra data
        else:
            pass

#endregion



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())

