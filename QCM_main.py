'''
This is the main code of the QCM acquization program

'''

import os
import math
import json
import datetime, time
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.uic import loadUi

# packages
from MainWindow import Ui_MainWindow
from UISettings import settings_init, settings_default
from modules import UIModules

from MatplotlibWidget import MatplotlibWidget

class QCMApp(QMainWindow):
    '''
    The settings of the app is stored in a dict
    '''
    def __init__(self):
        super(QCMApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fileFlag = False
        self.fileName = ''
        self.settings = settings_default
        self.load_settings()
        self.main()

        # check system
        self.system = UIModules.system_check()
        # initialize AccessMyVNA
        #?? add more code to disable settings_control tab and widges in settings_settings tab
        if self.system == 'win32': # windows
            from modules.AccessMyVNA import AccessMyVNA
            self.accvna = AccessMyVNA()
            # test if MyVNA program is available
            with self.accvna as accvna:
                ret = accvna.Init()
                if ret == 0: # is available
                    pass
                else: # not available
                    self.accvna = None
        else: # other system, data analysis only
            self.accvna = None    

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
        self.ui.lineEdit_startf1.textChanged[str].connect(self.update_startf1)
        self.ui.lineEdit_startf3.textChanged[str].connect(self.update_startf3)
        self.ui.lineEdit_startf5.textChanged[str].connect(self.update_startf5)
        self.ui.lineEdit_startf7.textChanged[str].connect(self.update_startf7)
        self.ui.lineEdit_startf9.textChanged[str].connect(self.update_startf9)
        self.ui.lineEdit_startf11.textChanged[str].connect(self.update_startf11)

        # set signals to update lineEdit_end settings
        self.ui.lineEdit_endf1.textChanged[str].connect(self.update_endf1)
        self.ui.lineEdit_endf3.textChanged[str].connect(self.update_endf3)
        self.ui.lineEdit_endf5.textChanged[str].connect(self.update_endf5)
        self.ui.lineEdit_endf7.textChanged[str].connect(self.update_endf7)
        self.ui.lineEdit_endf9.textChanged[str].connect(self.update_endf9)
        self.ui.lineEdit_endf11.textChanged[str].connect(self.update_endf11)

        # set lineEdit_scaninterval background
        self.ui.lineEdit_scaninterval.setStyleSheet(
            "QLineEdit { background: transparent; }"
        )

        # set signals to update fitting and display options
        self.ui.checkBox_dynamicfit.stateChanged.connect(self.update_dynamicfit)
        self.ui.checkBox_showsusceptance.stateChanged.connect(self.update_showsusceptance)
        self.ui.checkBox_showchi.stateChanged.connect(self.update_showchi)
        self.ui.checkBox_showpolar.stateChanged.connect(self.update_showpolarplot)
        self.ui.comboBox_fitfactor.activated.connect(self.update_fitfactor)

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
        # self.ui.comboBox_ref_channel.currentIndexChanged.connect() #?? add function checking if sample and ref have the same channel

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

        # move temp. module to treeWidget_settings_settings_hardware
        self.move_to_row2(
            self.ui.lineEdit_settings_settings_tempmodule, 
            self.ui.treeWidget_settings_settings_hardware, 
            'Module'
        )
        
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
        self.move_to_row2(
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
        self.move_to_row2(
            self.ui.pushButton_settings_harm_cntr, 
            self.ui.treeWidget_settings_settings_harmtree, 
            'Scan', 
            50
        )
        
        # move center checkBox_settings_temp_sensor to treeWidget_settings_settings_hardware
        self.move_to_row2(
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

        # set signals to update fit settings_settings
        self.ui.comboBox_harmfitfactor.activated.connect(self.update_harmfitfactor)
        self.ui.comboBox_sample_channel.activated.connect(self.update_samplechannel)
        self.ui.comboBox_ref_channel.activated.connect(self.update_refchannel)

        # set signals to update temperature settings_settings
        self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.update_tempsensor)
        self.ui.comboBox_thrmcpltype.activated.connect(self.update_thrmcpltype)

        # set signals to update plots settings_settings
        self.ui.comboBox_timeunit.activated.connect(self.update_timeunit)
        self.ui.comboBox_tempunit.activated.connect(self.update_tempunit)
        self.ui.comboBox_timescale.activated.connect(self.update_timescale)
        self.ui.comboBox_gammascale.activated.connect(self.update_gammascale)
        self.ui.checkBox_settings_settings_linktime.stateChanged.connect(self.update_linktime)
        
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
        self.ui.pushButton_spectra_fit_refresh.clicked.connect(self.on_click_pushButton_spectra_fit_refresh)

#endregion


#region spectra_mechanics


#endregion


#region data_data

#endregion


#region data_mechanics

#endregion


#region status bar

        #### add widgets to status bar. from left to right
        # move label_status_coordinates to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_coordinates)
        # move progressBar_status_interval_time to statusbar
        self.ui.progressBar_status_interval_time.setAlignment(Qt.AlignCenter)
        self.ui.statusbar.addPermanentWidget(self.ui.progressBar_status_interval_time)
        # move label_status_pts to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_pts)
        # move label_status_signal_ch to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_signal_ch)
        # move label_status_reftype to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_reftype)
        # move label_status_temp_sensor to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_temp_sensor)
        # move label_status_f0BW to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_f0BW)

#endregion


#region action group
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
            getattr(self.ui, 'mpl_sp' + str(i)).fig.text(0.01, 0.98, str(i), va='top',ha='left', weight='bold')
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
            self.move_to_row2(obj_box, parent, row_text, box_width)
            

    def move_to_row2(self, obj, parent, row_text, width=[]): 
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
    # @pyqtSlot(bool)
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
        # change the displayed file directory in lineEdit_datafilestr
        self.ui.lineEdit_datafilestr.setText(fileName)
        # reset lineEdit_reftime
        self.reset_reftime()
        # set lineEdit_reftime editable and enable pushButton_resetreftime
        self.ui.lineEdit_reftime.setReadOnly(False)
        self.ui.pushButton_resetreftime.setEnabled(True)

    def on_triggered_load_data(self):
        self.fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_datafilestr
        self.ui.lineEdit_datafilestr.setText(fileName)
        # set lineEdit_reftime
        # set lineEdit_reftime read only and disable pushButton_resetreftime
        self.ui.lineEdit_reftime.setReadOnly(True)
        self.ui.pushButton_resetreftime.setEnabled(False)

    # open folder in explorer
    # methods for different OS could be added
    def on_clicked_pushButton_gotofolder(self):
        file_path = self.ui.lineEdit_datafilestr.text() #?? replace with reading from settings dict
        path = os.path.abspath(os.path.join(file_path, os.pardir)) # get the folder of the file
        UIModules.open_file(path)

    # 
    def on_triggered_load_settings(self):
        self.fileName = self.openFileNameDialog('Choose a file to use its setting') # !! add path of last opened folder
        try:
            # load json file containing formerly saved settings
            with open(self.fileName, "r") as f:
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
        # set treeWidget_settings_settings_harmtree value
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

    def on_click_pushButton_spectra_fit_refresh(self):
        # ret, nSteps = self.accvna.GetScanSteps()
        # ret, f1, f2 = self.accvna.SetFequencies()
        # self.accvna.SingleScan()
        # time.sleep(2)
        # ret, f, G = self.accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=15)
        # ret, _, B = self.accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=16)
        # self.accvna.Close()
        # print(f)
        # print(G)
        # print(self.ui.mpl_spectra_fit.lG[0].get_xdata())
        with self.accvna as accvna:
            accvna.set_steps_freq()
            ret, f, G, B = accvna.single_scan()

        self.ui.mpl_spectra_fit.lG[0].set_xdata(f)
        self.ui.mpl_spectra_fit.lG[0].set_ydata(G)
        self.ui.mpl_spectra_fit.lB[0].set_xdata(f)
        self.ui.mpl_spectra_fit.lB[0].set_ydata(B)

        self.ui.mpl_spectra_fit.ax[0].set_xlim([f[0], f[-1]])
        self.ui.mpl_spectra_fit.ax[0].set_ylim([min(G), max(G)])
        self.ui.mpl_spectra_fit.ax[1].set_xlim([f[0], f[-1]])
        self.ui.mpl_spectra_fit.ax[1].set_ylim([min(B), max(B)])
        self.ui.mpl_spectra_fit.canvas.draw()
        # print(self.ui.mpl_spectra_fit.canvas)
        # print(self.ui.mpl_spectra_fit.lG[0].get_xdata())
        
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
    
    def update_startf1(self, freq_text):
        try:
            self.settings['lineEdit_startf1'] = float(freq_text)
        except:
            self.settings['lineEdit_startf1'] = 0

    def update_startf3(self, freq_text):
        try:
            self.settings['lineEdit_startf3'] = float(freq_text)
        except:
            self.settings['lineEdit_startf3'] = 0

    def update_startf5(self, freq_text):
        try:
            self.settings['lineEdit_startf5'] = float(freq_text)
        except:
            self.settings['lineEdit_startf5'] = 0

    def update_startf7(self, freq_text):
        try:
            self.settings['lineEdit_startf7'] = float(freq_text)
        except:
            self.settings['lineEdit_startf7'] = 0

    def update_startf9(self, freq_text):
        try:
            self.settings['lineEdit_startf9'] = float(freq_text)
        except:
            self.settings['lineEdit_startf9'] = 0

    def update_startf11(self, freq_text):
        try:
            self.settings['lineEdit_startf11'] = float(freq_text)
        except:
            self.settings['lineEdit_startf11'] = 0

    def update_endf1(self, freq_text):
        try:
            self.settings['lineEdit_endf1'] = float(freq_text)
        except:
            self.settings['lineEdit_endf1'] = 0

    def update_endf3(self, freq_text):
        try:
            self.settings['lineEdit_endf3'] = float(freq_text)
        except:
            self.settings['lineEdit_endf3'] = 0

    def update_endf5(self, freq_text):
        try:
            self.settings['lineEdit_endf5'] = float(freq_text)
        except:
            self.settings['lineEdit_endf5'] = 0

    def update_endf7(self, freq_text):
        try:
            self.settings['lineEdit_endf7'] = float(freq_text)
        except:
            self.settings['lineEdit_endf7'] = 0

    def update_endf9(self, freq_text):
        try:
            self.settings['lineEdit_endf9'] = float(freq_text)
        except:
            self.settings['lineEdit_endf9'] = 0

    def update_endf11(self, freq_text):
        try:
            self.settings['lineEdit_endf11'] = float(freq_text)
        except:
            self.settings['lineEdit_endf11'] = 0

        
    def update_recordinterval(self, recordinterval_text):
        try:
            self.settings['lineEdit_recordinterval'] = float(recorddinterval_text)
        except:
            self.settings['lineEdit_recordinterval'] = 0
    
    def update_refreshresolution(self, refreshsolution_text):
        try:
            self.settings['lineEdit_refreshsolution'] = float(refreshsolution_text)
        except:
            self.settings['lineEdit_refreshsolution'] = 0

    def update_dynamicfit(self):
        self.settings['checkBox_dynamicfit'] = not self.settings['checkBox_dynamicfit']

    def update_showsusceptance(self):
        self.settings['checkBox_showsusceptance'] = not self.settings['checkBox_showsusceptance']

    def update_showchi(self):
        self.settings['checkBox_showchi'] = not self.settings['checkBox_showchi']

    def update_showpolarplot(self):
        self.settings['checkBox_showpolar'] = not self.settings['checkBox_showpolar']

    def update_fitfactor(self, fitfactor_index):
        value = self.ui.comboBox_fitfactor.itemData(fitfactor_index)
        self.settings['comboBox_fitfactor'] = value

    def update_harmonic_tab(self):
        self.harmonic_tab = 2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1
        self.update_frequencies()

        self.update_guichecks(self.ui.checkBox_settings_temp_sensor, 'checkBox_settings_temp_sensor')
        self.update_guichecks(self.ui.checkBox_settings_settings_linktime, 'checkBox_settings_settings_linktime')

        self.update_guicombos(self.ui.comboBox_fit_method, 'comboBox_fit_method', 'span_mehtod_choose')
        self.update_guicombos(self.ui.comboBox_track_method, 'comboBox_track_method', 'track_mehtod_choose')
        self.update_guicombos(self.ui.comboBox_sample_channel, 'comboBox_sample_channel', 'sample_channel_choose')
        self.update_guicombos(self.ui.comboBox_ref_channel, 'comboBox_ref_channel', 'ref_channel_choose')
        self.update_guicombos(self.ui.comboBox_thrmcpltype, 'comboBox_thrmcpltype', 'thrmcpl_choose')
        self.update_guicombos(self.ui.comboBox_timeunit, 'comboBox_timeunit', 'time_unit_choose')
        self.update_guicombos(self.ui.comboBox_tempunit, 'comboBox_tempunit', 'temp_unit_choose')
        self.update_guicombos(self.ui.comboBox_timescale, 'comboBox_timescale', 'time_scale_choose')
        self.update_guicombos(self.ui.comboBox_gammascale, 'comboBox_gammascale', 'gamma_scale_choose')
        
        self.update_guicombos(self.ui.comboBox_bandwidth, 'comboBox_bandwidth', 'bandwidth_choose')
        self.update_guicombos(self.ui.comboBox_base_frequency, 'comboBox_base_frequency', 'base_frequency_choose')

    def update_base_freq(self, base_freq_index):
        value = self.ui.comboBox_base_frequency.itemData(base_freq_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_base_frequency'] = value
        self.update_frequencies()

    def update_bandwidth(self, bandwidth_index):
        value = self.ui.comboBox_bandwidth.itemData(bandwidth_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_bandwidth'] = value
        self.update_frequencies()

    def update_frequencies(self):
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['start_freq'] = self.harmonic_tab * self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_base_frequency'] - self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_bandwidth']
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['end_freq'] = self.harmonic_tab * self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_base_frequency'] + self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_bandwidth']
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(0).setText(1, str(self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['start_freq']))
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(1).setText(1, str(self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['end_freq']))

    def set_default_freqs(self):
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))

    def update_fitmethod(self, fitmethod_index):
        value = self.ui.comboBox_fit_method.itemData(fitmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_fit_method'] = value

    def update_trackmethod(self, trackmethod_index):
        value = self.ui.comboBox_track_method.itemData(trackmethod_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_track_method'] = value

    def update_harmfitfactor(self, harmfitfactor_index):
        value = self.ui.comboBox_harmfitfactor.itemData(harmfitfactor_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_harmfitfactor'] = value

    def update_samplechannel(self, samplechannel_index):
        value = self.ui.comboBox_sample_channel.itemData(samplechannel_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_sample_channel'] = value

    def update_refchannel(self, refchannel_index):
        value = self.ui.comboBox_ref_channel.itemData(refchannel_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_ref_channel'] = value

    def update_tempsensor(self):
        print(str(not self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['checkBox_settings_temp_sensor']))
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['checkBox_settings_temp_sensor'] = not self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['checkBox_settings_temp_sensor']

    def update_thrmcpltype(self, thrmcpltype_index):
        value = self.ui.comboBox_thrmcpltype.itemData(thrmcpltype_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_thrmcpltype'] = value

    def update_timeunit(self, timeunit_index):
        value = self.ui.comboBox_timeunit.itemData(timeunit_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_timeunit'] = value

    def update_tempunit(self, tempunit_index):
        value = self.ui.comboBox_tempunit.itemData(tempunit_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_tempunit'] = value


    def update_timescale(self, timescale_index):
        value = self.ui.comboBox_timescale.itemData(timescale_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_timescale'] = value

    def update_gammascale(self, gammascale_index):
        value = self.ui.comboBox_gammascale.itemData(gammascale_index)
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['comboBox_gammascale'] = value

    def update_linktime(self):
        self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['checkBox_settings_settings_linktime'] = not self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)]['checkBox_settings_settings_linktime']

    def update_guicombos(self, comboBox, name_in_settings, init_dict):
        for key, val in settings_init[init_dict].items():
            if key == self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)][name_in_settings]:
                comboBox.setCurrentIndex(comboBox.findData(key))
                break

    def update_guichecks(self, checkBox, name_in_settings):
        checkBox.setChecked(self.settings['tab_settings_settings_harm' + str(self.harmonic_tab)][name_in_settings])
    
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
        # set delault displaying of stackedWidget_spectra
        self.ui.stackedWidget_spectra.setCurrentIndex(0)
        # set delault displaying of stackedWidget_data
        self.ui.stackedWidget_data.setCurrentIndex(0)

        # load default start and end frequencies for lineEdit harmonics
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))
        # load default record interval
        self.ui.lineEdit_recordinterval.setText(str(self.settings['lineEdit_recordinterval']))
        # load default spectra refresh resolution
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))
        # load default fitting and display options
        self.ui.checkBox_dynamicfit.setChecked(self.settings['checkBox_dynamicfit'])
        self.ui.checkBox_showsusceptance.setChecked(self.settings['checkBox_showsusceptance'])
        self.ui.checkBox_showchi.setChecked(self.settings['checkBox_showchi'])
        self.ui.checkBox_showpolar.setChecked(self.settings['checkBox_showpolar'])
        # load default fit factor range
        for key, val in settings_init['fit_factor_choose'].items():
            if key == self.settings['comboBox_fitfactor']:
                self.ui.comboBox_fitfactor.setCurrentIndex(self.ui.comboBox_fitfactor.findData(key))
                break
        # set opened harmonic tab
        self.harmonic_tab = 1
        # set delault displaying of harmonics, triggered when index changed
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)


#endregion



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())
    