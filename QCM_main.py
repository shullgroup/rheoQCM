'''
This is the main code of the QCM acquization program

'''

import os
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

        self.settings = settings_default
        self.set_default_freqs()
        self.harmonic_tab = 1
        self.update_all_settings()
        self.main()

        # check system
        self.system = UIModules.system_check()
        # initialize AccessMyVNA
        if self.system == 'win32':
            from modules.AccessMyVNA import AccessMyVNA
            self.accvna = AccessMyVNA()
            

    def main(self):
 # loadUi('QCM_GUI_test4.ui', self) # read .ui file directly. You still need to compile the .qrc file
#region ###### initiate UI #################################

#region main UI 
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

        # set delault displaying of harmonics
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)

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

        # set label_actualinterval value
        self.ui.lineEdit_acquisitioninterval.textEdited.connect(self.set_label_actualinterval)
        self.ui.lineEdit_refreshresolution.textEdited.connect(self.set_label_actualinterval)

        # add value for the comboBox_fitfactor
        for key, val in settings_init['fit_factor_choose'].items():
            self.ui.comboBox_fitfactor.addItem(val, key)

        # set pushButton_gotofolder
        self.ui.pushButton_gotofolder.clicked.connect(self.on_clicked_pushButton_gotofolder)

        # set pushButton_newdata
        self.ui.pushButton_newdata.clicked.connect(self.on_triggered_new_data)

        # set pushButton_appenddata
        self.ui.pushButton_appenddata.clicked.connect(self.on_triggered_load_data)

        # set signals to update settings
        self.ui.lineEdit_acquisitioninterval.textChanged[str].connect(self.update_acquisitioninterval)
        self.ui.lineEdit_refreshresolution.textChanged[str].connect(self.update_refreshresolution)
        self.ui.checkBox_dynamicfit.stateChanged.connect(self.update_dynamicfit)
        self.ui.checkBox_showsusceptance.stateChanged.connect(self.update_showsusceptance)
        self.ui.checkBox_showchi.stateChanged.connect(self.update_showchi)
        self.ui.checkBox_polarplot.stateChanged.connect(self.update_showpolarplot)
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
        
        # set signals to update
        self.ui.comboBox_fit_method.activated.connect(self.update_fitmethod)
        self.ui.comboBox_track_method.activated.connect(self.update_trackmethod)
        self.ui.comboBox_harmfitfactor.activated.connect(self.update_harmfitfactor)
        self.ui.comboBox_sample_channel.activated.connect(self.update_samplechannel)
        self.ui.comboBox_ref_channel.activated.connect(self.update_refchannel)

        self.ui.tabWidget_settings_settings_harm.currentChanged.connect(self.update_harmonic_tab)
        self.ui.comboBox_base_frequency.activated.connect(self.update_base_freq)
        self.ui.comboBox_bandwidth.activated.connect(self.update_bandwidth)

        self.ui.checkBox_settings_temp_sensor.stateChanged.connect(self.update_tempsensor)
        self.ui.comboBox_thrmcpltype.activated.connect(self.update_thrmcpltype)

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
                    xlabel=r'$f$ (Hz)', 
                    ylabel=r'$G_P$ (mS)', 
                    ylabel2=r'$B_P$ (mS)',
                    showtoolbar=False,
                )
            )
            getattr(self.ui, 'frame_sp' + str(i)).setLayout(
                self.set_frame_layout(
                    getattr(self.ui, 'mpl_sp' + str(i))
                )
            )


        # add figure mpl_spectra_fit_polar into frame_spectra_fit_polar
        self.ui.mpl_spectra_fit_polar = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit_polar, 
            xlabel=r'$G_P$ (mS)',
            ylabel=r'$B_P$ (mS)',
            )
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit_polar.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit_polar))

        # add figure mpl_spectra_fit into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            xlabel=r'$f$ (Hz)',
            ylabel=r'$G_P$ (mS)',
            ylabel2=r'$B_P$ (mS)',
            showtoolbar=('Back', 'Forward', 'Pan', 'Zoom')
            )
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))
        # add plot
        self.ui.mpl_spectra_fit.lG = self.ui.mpl_spectra_fit.ax[0].plot([], [], color='tab:blue') # G
        self.ui.mpl_spectra_fit.lB = self.ui.mpl_spectra_fit.ax[1].plot([], [], color='tab:red') # B

        # add figure mpl_countour1 into frame_spectra_mechanics_contour1
        self.ui.mpl_countour1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour1, 
            xlabel=r'$d/\lambda$',
            ylabel=r'$\Phi$ ($\degree$)',
            )
        # self.ui.mpl_countour1.update_figure()
        self.ui.mpl_countour1.ax.plot([0, 1, 2, 3], [3,2,1,0])
        self.ui.frame_spectra_mechanics_contour1.setLayout(self.set_frame_layout(self.ui.mpl_countour1))

        # add figure mpl_countour2 into frame_spectra_mechanics_contour2
        self.ui.mpl_countour2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_mechanics_contour2, 
            xlabel=r'$d/\lambda$',
            ylabel=r'$\Phi$ ($\degree$)',
            )
        # self.ui.mpl_countour2.update_figure()
        self.ui.mpl_countour2.ax.plot([0, 1, 2, 3], [3,2,1,0])
        self.ui.frame_spectra_mechanics_contour2.setLayout(self.set_frame_layout(self.ui.mpl_countour2))

        # add figure mpl_plt1 into frame_spactra_fit
        self.ui.mpl_plt1 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            xlabel='Time (s)',
            ylabel=r'$\Delta f/n$ (Hz)',
            )
        # self.ui.mpl_plt1.update_figure()
        self.ui.mpl_plt1.ax.plot([0, 1, 2, 3], [3,2,1,0])
        self.ui.frame_plt1.setLayout(self.set_frame_layout(self.ui.mpl_plt1))

        # add figure mpl_plt2 into frame_spactra_fit
        self.ui.mpl_plt2 = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            xlabel='Time (s)',
            ylabel=r'$\Delta \Gamma$ (Hz)',
            )
        # self.ui.mpl_plt2.update_figure()
        self.ui.mpl_plt2.ax.plot([0, 1, 2, 3], [3,2,1,0])
        self.ui.frame_plt2.setLayout(self.set_frame_layout(self.ui.mpl_plt2))

#endregion


#endregion

#region ###### set UI value ###############################

        for i in range(1, settings_init['max_harmonic']+2, 2):
            if i in settings_default['harmonics_check']: # in the default range 
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
        self.ui.label_actualinterval.setText(str(self.settings['label_actualinterval']) + '  s')
        self.ui.lineEdit_acquisitioninterval.setText(str(self.settings['lineEdit_acquisitioninterval']))
        self.ui.lineEdit_refreshresolution.setText(str(self.settings['lineEdit_refreshresolution']))

#endregion


#region #########  functions ##############

    def link_tab_page(self, tab_idx):
        if tab_idx in [0]: # link settings_control to spectra_show and data_data
            self.ui.stackedWidget_spectra.setCurrentIndex(0)
            self.ui.stackedWidget_data.setCurrentIndex(0)
        elif tab_idx in [1, 2]: # link settings_settings and settings_data to spectra_fit and data_data
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

    # @pyqtSlot()
    def set_label_actualinterval(self):
        # get text
        acquisition_interval = self.ui.lineEdit_acquisitioninterval.text()
        refresh_resolution = self.ui.lineEdit_refreshresolution.text()
        #convert to flot
        try:
            acquisition_interval = float(acquisition_interval)
        except:
            acquisition_interval = 0
        try:
            refresh_resolution = float(refresh_resolution)
        except:
            refresh_resolution = 0
        # set label_actualinterval
        self.ui.label_actualinterval.setText(f'{acquisition_interval * refresh_resolution}  s')

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
        fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
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
        # print(path)
        # subprocess.Popen(f'explorer "{path}"') # every time open a new window
        # os.startfile(f'{path}') # if the folder is opend, make it active
        UIModules.open_file(path)

    # 
    def on_triggered_load_settings(self):
        fileName = self.openFileNameDialog('Choose a file to use its setting') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_datafilestr
        self.ui.lineEdit_datafilestr.setText(fileName)

    def on_triggered_actionSave(self):
        # save current data to file
        print('save function  to be added...')

    def on_triggered_actionSave_As(self):
        # save current data to a new file 
        fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_datafilestr
        self.ui.lineEdit_datafilestr.setText(fileName)

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
            n = f'{round(n)} *'
        else:
            n = f'1/{round(1/n)} *'
        # set treeWidget_settings_settings_harmtree value
        self.ui.label_spectra_fit_zoomtimes.setText(str(n))

    def on_released_slider_spanctrl(self):

        # get slider value
        n = 10 ** (self.ui.horizontalSlider_spectra_fit_spanctrl.value() / 10)
        # format n
        if n >= 1:
            n = round(n)
        else:
            n = 1/round(1 / n)

        # set span

        # start a single scan

        # set span text

        # reset slider to 1
        self.ui.horizontalSlider_spectra_fit_spanctrl.setValue(0)

    def on_click_pushButton_spectra_fit_refresh(self):
        ret, nSteps = self.accvna.GetScanSteps()
        ret, f1, f2 = self.accvna.SetFequencies()
        self.accvna.SingleScan()
        time.sleep(2)
        ret, f, G = self.accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=15)
        ret, _, B = self.accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=16)
        self.accvna.Close()
        # print(f)
        # print(G)
        # print(self.ui.mpl_spectra_fit.lG[0].get_xdata())
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
        
    def update_acquisitioninterval(self, acquisitioninterval_text):
        if acquisitioninterval_text != '':
            self.settings['lineEdit_acquisitioninterval'] = float(acquisitioninterval_text)
    
    def update_refreshresolution(self, refreshsolution_text):
        if refreshsolution_text != '':
            self.settings['lineEdit_refreshsolution'] = float(refreshsolution_text)

    def update_dynamicfit(self):
        self.settings['checkBox_dynamicfit'] = not self.settings['checkBox_dynamicfit']

    def update_showsusceptance(self):
        self.settings['checkBox_showsusceptance'] = not self.settings['checkBox_showsusceptance']

    def update_showchi(self):
        self.settings['checkBox_showchi'] = not self.settings['checkBox_showchi']

    def update_showpolarplot(self):
        self.settings['checkBox_polarplot'] = not self.settings['checkBox_polarplot']

    def update_fitfactor(self, fitfactor_index):
        value = self.ui.comboBox_fitfactor.itemData(fitfactor_index)
        self.settings['comboBox_fitfactor'] = value

    def update_harmonic_tab(self):
        self.harmonic_tab = 2 * self.ui.tabWidget_settings_settings_harm.currentIndex() + 1
        self.update_all_settings()

    def update_base_freq(self, base_freq_index):
        value = self.ui.comboBox_base_frequency.itemData(base_freq_index)
        self.settings['comboBox_base_frequency'] = value
        self.update_all_settings()

    def update_bandwidth(self, bandwidth_index):
        value = self.ui.comboBox_bandwidth.itemData(bandwidth_index)
        self.settings['comboBox_bandwidth'] = value
        self.update_all_settings()

    def update_all_settings(self):
        self.start_freq = self.harmonic_tab * self.settings['comboBox_base_frequency'] - self.settings['comboBox_bandwidth']
        self.end_freq = self.harmonic_tab * self.settings['comboBox_base_frequency'] + self.settings['comboBox_bandwidth']
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(0).setText(1, str(self.start_freq))
        self.ui.treeWidget_settings_settings_harmtree.topLevelItem(0).child(1).setText(1, str(self.end_freq))

    def set_default_freqs(self):
        for i in range(1, int(settings_init['max_harmonic'] + 2), 2):
            getattr(self.ui, 'lineEdit_startf' + str(i)).setText(str(self.settings['lineEdit_startf' + str(i)]))
            getattr(self.ui, 'lineEdit_endf' + str(i)).setText(str(self.settings['lineEdit_endf' + str(i)]))

    def update_fitmethod(self, fitmethod_index):
        value = self.ui.comboBox_fit_method.itemData(fitmethod_index)
        self.settings['comboBox_fit_method'] = value

    def update_trackmethod(self, trackmethod_index):
        value = self.ui.comboBox_track_method.itemData(trackmethod_index)
        self.settings['comboBox_track_method'] = value

    def update_harmfitfactor(self, harmfitfactor_index):
        value = self.ui.comboBox_harmfitfactor.itemData(harmfitfactor_index)
        self.settings['comboBox_harmfitfactor'] = value

    def update_samplechannel(self, samplechannel_index):
        value = self.ui.comboBox_sample_channel.itemData(samplechannel_index)
        self.settings['comboBox_sample_channel'] = value

    def update_refchannel(self, refchannel_index):
        value = self.ui.comboBox_ref_channel.itemData(refchannel_index)
        self.settings['comboBox_ref_channel'] = value

    def update_tempsensor(self):
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

    def update_linktime(self):
        self.settings['checkBox_settings_settings_linktime'] = not self.settings['checkBox_settings_settings_linktime']

#endregion



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())
    