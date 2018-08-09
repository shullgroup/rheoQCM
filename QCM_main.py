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
import types
from PyQt5.QtCore import pyqtSlot, Qt, QEvent
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QGridLayout, QLineEdit, QCheckBox, QComboBox, QRadioButton, QMenu
)
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent
# from PyQt5.uic import loadUi

# packages
from MainWindow import Ui_MainWindow
from UISettings import settings_init, settings_default
from modules import UIModules, MathModules

from MatplotlibWidget import MatplotlibWidget


if UIModules.system_check() == 'win32': # windows
    try:
        from modules.AccessMyVNA_dummy import AccessMyVNA
        print(AccessMyVNA)
        # test if MyVNA program is available
        with AccessMyVNA() as accvna:
            if accvna.Init() == 0: # connection with myVNA is available
                from modules import tempDevices
    except Exception as e: # no myVNA connected. Analysis only
        print('Failed to import AccessMyVNA module!')
        print(e)
else: # linux or MacOS
    # for test only
    from modules.AccessMyVNA_dummy import AccessMyVNA

class PeakTracker:
    def __init__(self):
        self.f0 = None
        self.gamma0 = None
        self.freq = None
        self.G = None # conductance
        self.B = None # susceptance
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
        self.tempsensor = None # class for temp sensor
        self.idle = True # if test is running
        self.reading = False # if myVNA is scanning and reading data
        
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
            self.accvna = AccessMyVNA() # for test only
        print(self.accvna)
        self.main()
        self.load_settings()


    def main(self):
 # loadUi('QCM_GUI_test4.ui', self) # read .ui file directly. You still need to compile the .qrc file
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
        # set lineEdit_startf<n> & lineEdit_endf<n> background
        for i in range(1, settings_init['max_harmonic']+2, 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )
            getattr(self.ui, 'lineEdit_endf' + str(i)).setStyleSheet(
                "QLineEdit { background: transparent; }"
            )

        # set pushButton_resetreftime
        self.ui.pushButton_resetreftime.clicked.connect(self.reset_reftime)

        # set lineEdit_scaninterval value
        self.ui.lineEdit_recordinterval.editingFinished.connect(self.set_lineEdit_scaninterval)
        self.ui.lineEdit_refreshresolution.editingFinished.connect(self.set_lineEdit_scaninterval)

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

        # move lineEdit_scan_harmpoints
        self.move_to_col2(
            self.ui.lineEdit_scan_harmpoints,
            self.ui.treeWidget_settings_settings_harmtree,
            'Points',
            100,
        )

        # move lineEdit_peaks_maxnum
        self.move_to_col2(
            self.ui.lineEdit_peaks_maxnum,
            self.ui.treeWidget_settings_settings_harmtree,
            'Max #',
            100,
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

        # comboBox_span_method
        self.create_combobox(
            'comboBox_span_method', 
            settings_init['span_mehtod_choose'], 
            100, 
            'Method', 
            self.ui.treeWidget_settings_settings_harmtree
        )

        # add span track_method
        self.create_combobox(
            'comboBox_span_track', 
            settings_init['span_track_choose'], 
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
        if self.accvna and self.system == 'win32':
            self.settings['tempdevs_choose'] = \
            tempdevs_choose = \
            tempDevices.dict_available_devs(settings_init['devices_dict'])
            self.create_combobox(
                'comBox_tempdevice',
                tempdevs_choose,  
                100,
                'Device',
                self.ui.treeWidget_settings_settings_hardware, 
            )
            self.settings['comBox_tempdevice'] = self.ui.comBox_tempdevice.itemData(self.ui.comBox_tempdevice.currentIndex())
            self.ui.comBox_tempdevice.activated.connect(self.update_tempdevice)
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
            'comboBox_yscale', 
            settings_init['y_scale_choose'], 
            100, 
            'Γ Scale', 
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

        # set signals of widgets in tabWidget_settings_settings_harm
        self.ui.lineEdit_scan_harmstart.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.lineEdit_scan_harmend.editingFinished.connect(self.on_editingfinished_harm_freq)
        self.ui.comboBox_base_frequency.activated.connect(self.update_base_freq)
        self.ui.comboBox_bandwidth.activated.connect(self.update_bandwidth)

        # set signals to update span settings_settings
        self.ui.comboBox_span_method.activated.connect(self.update_harmwidget)
        self.ui.comboBox_span_track.activated.connect(self.update_harmwidget)
        self.ui.checkBox_harmfit.clicked['bool'].connect(self.update_harmwidget)
        self.ui.comboBox_harmfitfactor.activated.connect(self.update_harmwidget)
        self.ui.lineEdit_peaks_maxnum.editingFinished.connect(self.update_harmwidget)
        self.ui.lineEdit_peaks_threshold.editingFinished.connect(self.update_harmwidget)
        self.ui.lineEdit_peaks_prominence.editingFinished.connect(self.update_harmwidget)

        # set signals to update hardware settings_settings
        self.ui.comboBox_sample_channel.activated.connect(self.update_samplechannel)
        self.ui.comboBox_ref_channel.activated.connect(self.update_refchannel)

        # set signals to update temperature settings_settings
        self.ui.comboBox_settings_mechanics_selectmodel.activated[str].connect(self.update_module)
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
            ) #TODO add 'Pan' and figure out how to identify mouse is draging 
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))
        # connect signal
        self.ui.mpl_spectra_fit.ax[0].cidx = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('xlim_changed', self.on_fit_lims_change)
        self.ui.mpl_spectra_fit.ax[0].cidy = self.ui.mpl_spectra_fit.ax[0].callbacks.connect('ylim_changed', self.on_fit_lims_change)
        self.ui.mpl_spectra_fit.canvas.mpl_connect('button_press_event', self.spectra_fit_axesevent_disconnect)
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


    def span_check(self, harm, f1=None, f2=None):
        '''
        check if lower limit ('f1' in Hz) and upper limit ('f2' in Hz) in base freq +/- BW of harmonic 'harm'
        if out of the limit, return the part in the range
        and show alert in statusbar
        '''
        # get freq_range
        bf1, bf2 = np.array(self.settings['freq_range'][harm]) # in Hz
        # check f1, and f2
        if f1 and (f1 < bf1 or f1 >= bf2): # f1 out of limt
            f1 = bf1
            #TODO update statusbar 'lower bond out of limit and reseted. (You can increase the bandwidth in settings)'
        if f2 and (f2 > bf2 or f2 <= bf1): # f2 out of limt
            f2 = bf2
            #TODO update statusbar 'upper bond out of limit and reseted. (You can increase the bandwidth in settings)'
        if f1 and f2 and (f1 >= f2):
            f2 = bf2

        return [f1, f2]


    def on_clicked_pushButton_spectra_fit_refresh(self):
        print('accvna', self.accvna)
        #TODOO get parameters from current setup: harm_tab
        if self.idle == False: # test is running
            # get params from tabWidget_settings_settings_harm & treeWidget_settings_settings_harmtree
            pass

            # check lim with BW
            # flim1, flim2 = self.span_check(harm=self.settings['tabWidget_settings_settings_harm'], f1=flim1, f2=flim2)
        with self.accvna as accvna:
            accvna.set_steps_freq()
            ret, f, G, B = accvna.single_scan()

        ## disconnect axes event
        self.mpl_disconnect_cid(self.ui.mpl_spectra_fit)        
        self.ui.mpl_spectra_fit.update_data(ls=['lG'], xdata=[f], ydata=[G])
        self.ui.mpl_spectra_fit.update_data(ls=['lB'], xdata=[f], ydata=[B])
        ## connect axes event
        self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)

        self.ui.mpl_spectra_fit_polar.update_data(ls=['l'], xdata=[G], ydata=[B])
        
        # update lineedit_fit_span
        self.update_lineedit_fit_span(f)

    def on_fit_lims_change(self, axes):
        print('on lim changed')
        axG = self.ui.mpl_spectra_fit.ax[0]
        axB = self.ui.mpl_spectra_fit.ax[1]
        axP = self.ui.mpl_spectra_fit_polar.ax[0]

        # print('g', axG.get_contains())
        # print('r', axG.contains('button_release_event'))
        # print('p', axG.contains('button_press_event'))

        # data lims [min, max]
        dflim1, dflim2 = MathModules.datarange(self.ui.mpl_spectra_fit.l['lB'][0].get_xdata())
        # get axes lims
        flim1, flim2 = axG.get_xlim()
        # check lim with BW
        flim1, flim2 = self.span_check(harm=self.settings['tabWidget_settings_settings_harm'], f1=flim1, f2=flim2)
        print('get_navigate_mode()', axG.get_navigate_mode())
        print('flims', flim1, flim2)
        print(dflim1, dflim2)
        
        print(axG.get_navigate_mode())
        # if axG.get_navigate_mode() == 'PAN': # pan
        #     # set a new x range: combine span of dflims and flims
        #     flim1 = min([flim1, dflim1])
        #     flim2 = max([flim2, dflim2])
        # elif axG.get_navigate_mode() == 'ZOOM': # zoom
        #     pass
        # else: # axG.get_navigate_mode() == 'None'
        #     pass
        print('flim', flim1, flim2)

        # accvna setup frequency
        self.accvna.SetFequencies(f1=flim1, f2=flim2, nFlags=1)
        ret, f, G, B = self.accvna.single_scan()
        
        ### reset x,y lim
        ## disconnect axes event
        self.mpl_disconnect_cid(self.ui.mpl_spectra_fit)

        # axG.autoscale(axis='y')
        # axB.autoscale(axis='y')

        # mpl_spectra_fit 
        # clear lines
        self.ui.mpl_spectra_fit.clr_alldata()
        # plot data
        self.ui.mpl_spectra_fit.update_data(ls=['lG', 'lB'], xdata=[f, f], ydata=[G, B])

        # mpl_spectra_fit_polar
        # clear lines
        self.ui.mpl_spectra_fit_polar.clr_alldata()
        # plot data
        self.ui.mpl_spectra_fit_polar.update_data(ls=['l'], xdata=[G], ydata=[B])
        axG.set_xlim(f[0], f[-1])
        axG.set_ylim(min(G), max(G))
        axB.set_ylim(min(B), max(B))

        ## connect axes event
        self.mpl_connect_cid(self.ui.mpl_spectra_fit, self.on_fit_lims_change)

        # set xlabel
        self.mpl_set_faxis(axG)

        # update lineEdit_spectra_fit_span
        self.update_lineedit_fit_span(f)

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

    def update_harmwidget(self, signal):
        '''
        update widgets in treeWidget_settings_settings_harmtree
        except lineEdit_harmstart & lineEdit_harmend
        '''
        #  of the signal isA QLineEdit object, update QLineEdit vals in dict
        print('update', signal)
        harm = self.settings['tabWidget_settings_settings_harm']
        tabwidget_name = 'tab_settings_settings_harm' + str(harm)
        if isinstance(self.sender(), QLineEdit):
                try:
                    self.settings[tabwidget_name][self.sender().objectName()] = float(signal)
                except:
                    self.settings[tabwidget_name][self.sender().objectName()] = 0
        # if the sender of the signal isA QCheckBox object, update QCheckBox vals in dict
        elif isinstance(self.sender(), QCheckBox):
            self.settings[tabwidget_name][self.sender().objectName()] = not self.settings[tabwidget_name][self.sender().objectName()]
        # if the sender of the signal isA QRadioButton object, update QRadioButton vals in dict
        elif isinstance(self.sender(), QRadioButton):
            self.settings[tabwidget_name][self.sender().objectName()] = not self.settings[tabwidget_name][self.sender().objectName()]
        # if the sender of the signal isA QComboBox object, udpate QComboBox vals in dict
        elif isinstance(self.sender(), QComboBox):
            try: # if w/ userData, use userData
                value = self.sender().itemData(signal)
            except: # if w/o userData, use the text
                value = self.sender().itemText(signal)
            self.settings[tabwidget_name][self.sender().objectName()] = value

    def update_harmonic_tab(self):
        #print("update_harmonic_tab was called")
        harm = 2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1
        self.peak_tracker.harmonic_tab = \
        self.settings['tabWidget_settings_settings_harm'] = \
            harm
        
        self.update_frequencies()

        # update lineEdit_scan_harmpoints
        self.ui.lineEdit_scan_harmpoints.setText(
            str(self.settings['tab_settings_settings_harm' + str(harm)]['lineEdit_scan_harmpoints'])
        )
        self.load_comboBox(self.ui.comboBox_span_method, 'span_mehtod_choose', parent='tab_settings_settings_harm' + str(harm))
        self.load_comboBox(self.ui.comboBox_span_track, 'span_track_choose', parent='tab_settings_settings_harm' + str(harm)) 
        self.load_comboBox(self.ui.comboBox_harmfitfactor, 'fit_factor_choose', parent='tab_settings_settings_harm' + str(harm))

        # update lineEdit_peaks_maxnum
        self.ui.lineEdit_peaks_maxnum.setText(
            str(self.settings['tab_settings_settings_harm' + str(harm)]['lineEdit_peaks_maxnum'])
        )
 
        # update lineEdit_peaks_threshold
        self.ui.lineEdit_peaks_threshold.setText(
            str(self.settings['tab_settings_settings_harm' + str(harm)]['lineEdit_peaks_threshold'])
        )

        # update lineEdit_peaks_prominence
        self.ui.lineEdit_peaks_prominence.setText(
            str(self.settings['tab_settings_settings_harm' + str(harm)]['lineEdit_peaks_prominence'])
        )

    def update_base_freq(self, base_freq_index):
        fbase = self.ui.comboBox_base_frequency.itemData(base_freq_index) # in MHz
        self.settings['comboBox_base_frequency'] = fbase
        # update statusbar
        self.statusbar_f0bw_update()
        # update freq_range
        self.update_freq_range()
        # check freq_span
        self.check_freq_span()
        # update freqrency display
        self.update_frequencies()

    def update_bandwidth(self, bandwidth_index):
        BW = self.ui.comboBox_bandwidth.itemData(bandwidth_index) # in MHz
        self.settings['comboBox_bandwidth'] = BW
        # update statusbar
        self.statusbar_f0bw_update()
        # update freq_range
        self.update_freq_range()
        # check freq_span
        self.check_freq_span()
        # update freqrency display
        self.update_frequencies()

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
        for i in range(1, settings_init['max_harmonic']+2, 2):
            self.settings['freq_range'][i] = [i*fbase-BW, i*fbase+BW]
        print(self.settings['freq_range'])

    def check_freq_span(self):
        '''
        check if settings['freq_span'] (freq span for each harmonic) values in the allowed range self.settings['freq_range']
        '''
        try: 
            self.settings['freq_span'] # check if self.settings['freq_span'] exist
            for i in range(1, settings_init['max_harmonic']+2, 2):
                self.settings['freq_span'][i] = self.span_check(i, self.settings['freq_span'][i][0], self.settings['freq_span'][i][1])
        except: # if self.settings['freq_span'] does not exist
            try: # check if self.settings['freq_range'] exist
                self.settings['freq_span'] = self.settings['freq_range']
            except: # if self.settings['freq_range'] does not exist
                self.update_freq_range() # initiate self.settings['freq_range']
                self.settings['freq_span'] = self.settings['freq_range']
        print('f_sp', self.settings['freq_span'])

    def update_frequencies(self):
        
        # update lineEdit_startf<n> & lineEdit_endf<n>
        for i in range(1, settings_init['max_harmonic']+2, 2):
            # check f1 (start), f2 (end)
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(MathModules.num2str(self.settings['freq_span'][i][0]*1e-6, precision=12)) # display as MHz
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(MathModules.num2str(self.settings['freq_span'][i][1]*1e-6, precision=12)) # display as MHz
            
        # update start/end in treeWidget_settings_settings_harmtree
            harm = self.settings['tabWidget_settings_settings_harm']
            # Set Start
            self.ui.lineEdit_scan_harmstart.setText(
                MathModules.num2str(self.settings['freq_span'][harm][0]*1e-6, precision=12)
            )
            # set End
            self.ui.lineEdit_scan_harmend.setText(
                MathModules.num2str(self.settings['freq_span'][harm][1]*1e-6, precision=12)
            )

    def on_editingfinished_harm_freq(self):
        '''
        update frequency when lineEdit_scan_harmstart or  lineEdit_scan_harmend edited
        '''
        print(self.sender().objectName())
        harmstart = float(self.ui.lineEdit_scan_harmstart.text()) * 1e6 # in Hz
        harmend = float(self.ui.lineEdit_scan_harmend.text()) * 1e6 # in Hz
        harm=self.settings['tabWidget_settings_settings_harm']
        print(harm, harmstart, harmend)
        f1, f2 = self.span_check(harm=harm, f1=harmstart, f2=harmend)
        print(f1, f2)
        self.settings['freq_span'][harm] = [f1, f2]
        # self.settings['freq_span'][harm] = [harmstart, harmend] # in Hz
        # self.check_freq_span()
        self.update_frequencies()

    def set_default_freqs(self):
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))

    def update_spanmethod(self, fitmethod_index):
        value = self.ui.comboBox_span_method.itemData(fitmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['comboBox_span_method'] = value

    def update_spantrack(self, trackmethod_index):
        value = self.ui.comboBox_span_track.itemData(trackmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.peak_tracker.harmonic_tab)]['comboBox_span_track'] = value

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


    def update_tempdevice(self, tempdevice_index):
        value = self.ui.comBox_tempdevice.itemData(tempdevice_index)
        self.settings['comBox_tempdevice'] = value
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
        print(self.settings['comBox_tempdevice'], self.settings['comboBox_thrmcpltype'])
        self.ui.label_temp_devthrmcpl.setText(
            'Dev/Thermocouple: {}/{}'.format(
                self.settings['comBox_tempdevice'], 
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

    def load_comboBox(self, comboBox, choose_dict, parent=None):
        comboBoxName = comboBox.objectName()
        for key, val in settings_init[choose_dict].items():
            if not parent: # not embeded in subdict
                if key == self.settings[comboBoxName]:
                    comboBox.setCurrentIndex(comboBox.findData(key))
                    break
            else:
                if key == self.settings[parent][comboBoxName]:
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
        '''
        setup the UI with the value from self.settings
        '''

        # load default crystal settings 

        # create self.settings['freq_range']. 
        # this has to be initated before any 
        self.update_freq_range()
        # update self.settings['freq_span']
        self.check_freq_span()


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
        # set opened harmonic tab
        self.peak_tracker.harmonic_tab = self.settings['tabWidget_settings_settings_harm'] = 1 #TODO

        ## following data is read from self.settings

        # load default start and end frequencies fo
        # r lineEdit harmonics

        # load default record interval
        self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        # load default spectra refresh resolution
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))
        # load default fitting and display options
        self.ui.checkBox_dynamicfit.setChecked(self.settings['checkBox_dynamicfit'])

        # load this first to create self.settings['freq_range'] & self.settings['freq_span']
        self.load_comboBox(self.ui.comboBox_base_frequency, 'base_frequency_choose')
        self.load_comboBox(self.ui.comboBox_bandwidth, 'bandwidth_choose')
        # create self.settings['freq_range']. 
        # this has to be initated before any 
        self.update_freq_range()
        # update self.settings['freq_span']
        self.check_freq_span()
        # update frequencies display
        self.update_frequencies()

        # load default fit factor range
        self.load_comboBox(self.ui.comboBox_fitfactor, 'fit_factor_choose')

        # load default VNA settings
        self.load_comboBox(self.ui.comboBox_sample_channel, 'sample_channel_choose')
        self.load_comboBox(self.ui.comboBox_ref_channel, 'ref_channel_choose')
        
        # set treeWidget_settings_settings_harmtree display
        self.update_harmonic_tab()

        # load default temperature settings
        self.load_comboBox(self.ui.comboBox_settings_mechanics_selectmodel, 'thrmcpl_choose')

        self.ui.checkBox_settings_temp_sensor.setChecked(self.settings['checkBox_settings_temp_sensor'])

        try:
            self.load_comboBox(self.ui.comBox_tempdevice, 'tempdevs_choose')
        except:
            pass
        self.load_comboBox(self.ui.comboBox_thrmcpltype, 'thrmcpl_choose')
        # update display on label_temp_devthrmcpl
        self.set_label_temp_devthrmcpl() # this should be after temp_sensor & thrmcpl 

        # load default plots settings
        self.load_comboBox(self.ui.comboBox_timeunit, 'time_unit_choose')
        self.load_comboBox(self.ui.comboBox_tempunit, 'temp_unit_choose')
        self.load_comboBox(self.ui.comboBox_timescale, 'time_scale_choose')
        self.load_comboBox(self.ui.comboBox_yscale, 'y_scale_choose')

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

    def smart_peak_tracker(self, harmonic, freq, conductance, susceptance, G_parameters):
        self.peak_tracker.f0 = G_parameters[0]
        self.peak_tracker.gamma0 = G_parameters[1]

        # determine the structure field that should be used to extract out the initial-guessing method
        if self.settings['tab_settings_settings_harm' + str(harmonic)]['comboBox_span_method'] == 'bmax':
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
        track_method = self.settings['tab_settings_settings_harm' + str(harmonic)]['comboBox_span_track'] 
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
            current_center = (float(self.se5ttings['lineEdit_startf' + str(harmonic)]) + float(self.settings['lineEdit_endf' + str(harmonic)]))/2 
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
                    self.Peak_tracker.G = 1e3 * rawdata[:num_pts+1]
                    self.peak_tracker.B = 1e3 * rawdata[num_pts:]
                    self.peak_tracker.freq = np.arange(start1,end1-(end1-start1)/num_pts+1,(end1-start1)/num_pts)
                    flag = 1
                    print('Status: Scan successful.')
        #TODO refit loaded raw spectra data
        else:
            pass

#endregion



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())

