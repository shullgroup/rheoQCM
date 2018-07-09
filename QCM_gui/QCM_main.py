'''
This is the main code of the QCM acquization program

'''

import sys
import datetime
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.uic import loadUi

# packages
from UI.MainWindow import Ui_MainWindow
import GUISettings as constant
import GUIFunc
from MatplotlibWidget import MatplotlibWidget

class QCMApp(QMainWindow):
    '''
    The settings of the app is stored in a dict
    '''
    def __init__(self):
        super(QCMApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # loadUi('QCM_GUI_test4.ui', self) # read .ui file directly. You still need to compile the .qrc file

        ###### setup UI apperiance #################################
        # set window title
        self.setWindowTitle(constant.window_title)
        # set window size
        self.resize(*constant.window_size)
        # set displaying of harmonics
        self.ui.tabWidget_settings_settings_harm.setCurrentIndex(0)
        i = 1
        while True:
            try:
                if i <= constant.max_harmonic: # in the range to display
                    # set to visable which is default. nothing to do

                    # add checkbox to tabWidget_ham for harmonic selection
                    setattr(self.ui, 'checkBox_tree_harm' + str(i), QCheckBox())
                    self.ui.tabWidget_settings_settings_harm.tabBar().setTabButton(self.ui.tabWidget_settings_settings_harm.indexOf(getattr(self.ui, 'tab_settings_settings_harm' + str(i))), QTabBar.LeftSide, getattr(self.ui, 'checkBox_tree_harm' + str(i)))

                    # set signal
                    getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'checkBox_harm' + str(i)).setChecked)
                    getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked)
                    getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'frame_sp' +str(i)).setVisible)
                    getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'frame_sp' +str(i)).setVisible)
                    
                    if i in constant.default_harmonics: # in the default range 
                        # settings/control/Harmonics
                        getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(True)
                        getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(True)

                    else: # out of the default range
                        getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(False)
                        getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(False)
                        # hide spectra/sp
                        getattr(self.ui, 'frame_sp' + str(i)).setVisible(False)
                else: # to be hided
                    # settings/control/Harmonics
                    getattr(self.ui, 'checkBox_harm' + str(i)).hide()
                    getattr(self.ui, 'lineEdit_start_f' + str(i)).hide()
                    getattr(self.ui, 'lineEdit_end_f' + str(i)).hide()
                    getattr(self.ui, 'pushButton_cntr' + str(i)).hide()
                    # data/F1/checkbox
                    getattr(self.ui, 'checkBox_plt1_h' + str(i)).hide()
                    getattr(self.ui, 'checkBox_plt2_h' + str(i)).hide()
                    # spectra/sp
                    getattr(self.ui, 'frame_sp' + str(i)).setVisible(False)
                i += 2 
            except: 
                break
        
        max_gui_harmonic = i - 2 # maximum harmomic available in GUI

        # remove tabs in tabWidget_settings_settings_harm
        for i in range(constant.max_harmonic, max_gui_harmonic):
                # settings/settings/tabWidget_settings_settings_harm
                getattr(self.ui, 'tabWidget_settings_settings_harm').removeTab(int((constant.max_harmonic-1)/2)+1) # remove the same index

        # set comboBox_plt1_choice, comboBox_plt2_choice
        # dict for the comboboxes
        for key, val in constant.plt_choice.items():
            # userData is setup for geting the plot type
            # userDat can be access with itemData(index)
            self.ui.comboBox_plt1_choice.addItem(val, key)
            self.ui.comboBox_plt2_choice.addItem(val, key)
        self.ui.comboBox_plt1_choice.setCurrentIndex(2)
        self.ui.comboBox_plt2_choice.setCurrentIndex(3)
        # print(self.ui.comboBox_plt1_choice.itemData(2))

        # set time interval
        self.ui.label_actualinterval.setText(str(constant.actual_interval) + '  s')
        self.ui.lineEdit_acquisitioninterval.setText(str(constant.acquisition_interval))
        self.ui.lineEdit_refreshresolution.setText(str(constant.refresh_resolution))

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

        # set treeWidget_settings_settings_harmtree expanded
        self.ui.treeWidget_settings_settings_harmtree.expandToDepth(0)
        # set treeWidget_settings_settings_hardware expanded
        self.ui.treeWidget_settings_settings_hardware.expandToDepth(0)
        # set treeWidget_settings_settings_plots expanded
        self.ui.treeWidget_settings_settings_plots.expandToDepth(0)
        # set treeWidget_settings_data_settings expanded
        self.ui.treeWidget_settings_data_settings.expandToDepth(0)
        
        ### add combobox into treewidget
        # comboBox_fit_method
        self.create_combobox('comboBox_fit_method', constant.span_mehtod_choose, 100, 'Method', self.ui.treeWidget_settings_settings_harmtree)

        # add track_method
        self.create_combobox('comboBox_track_method', constant.track_mehtod_choose, 100, 'Tracking', self.ui.treeWidget_settings_settings_harmtree)

        # insert sample_channel
        self.create_combobox('comboBox_sample_channel', constant.sample_channel_choose, 100, 'Sample Channel', self.ui.treeWidget_settings_settings_hardware)

        # insert base_frequency
        self.create_combobox('comboBox_base_frequency', constant.base_frequency_choose, 100, 'Base Frequency', self.ui.treeWidget_settings_settings_hardware)

        # insert bandwidth
        self.create_combobox('comboBox_bandwidth', constant.bandwidth_choose, 100, 'Bandwidth', self.ui.treeWidget_settings_settings_hardware)

        # insert refernence type
        self.create_combobox('comboBox_ref_type', constant.ref_type_choose, 100, 'Type', self.ui.treeWidget_settings_data_settings)

        # move center pushButton_settings_harm_cntr to treeWidget_settings_settings_harmtree
        self.ui.treeWidget_settings_settings_harmtree.setItemWidget(self.ui.treeWidget_settings_settings_harmtree.findItems('Scan', Qt.MatchExactly | Qt.MatchRecursive, 0)[0], 1, self.ui.pushButton_settings_harm_cntr)
        # set the pushbutton width
        self.ui.pushButton_settings_harm_cntr.setMaximumWidth(50)

        
        # move center checkBox_settings_temp_sensor to treeWidget_settings_settings_hardware
        self.ui.treeWidget_settings_settings_hardware.setItemWidget(self.ui.treeWidget_settings_settings_hardware.findItems('Temperature', Qt.MatchExactly | Qt.MatchRecursive, 0)[0], 1, self.ui.checkBox_settings_temp_sensor)

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
        # set treeWidget_settings_data_settings background
        self.ui.treeWidget_settings_data_settings.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )
        
        # set treeWidget_settings_settings_plots background
        self.ui.treeWidget_settings_settings_plots.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # resize the TabBar.Button
        self.ui.tabWidget_settings_settings_harm.setStyleSheet(
            "QTabWidget::pane { border: 0px; }"
            "QTabWidget {background-color: transparent;}"
            "QTabWidget::tab-bar { left: 5px; /* move to the right by 5px */ }"
            "QTabBar::tab { border: 1px solid #9B9B9B; border-top-left-radius: 1px; border-top-right-radius: 1px;}"
            "QTabBar::tab { height: 17px; width: 38px; padding: 0px; }" 
            "QTabBar::tab:selected, QTabBar::tab:hover { background: white; }"
            "QTabBar::tab:selected { height: 19px; width: 40px; border-bottom-color: none; }"
            "QTabBar::tab:selected { margin-left: -2px; margin-right: -2px; }"
            "QTabBar::tab:first:selected { margin-left: 0; width: 40px; }"
            "QTabBar::tab:last:selected { margin-right: 0; width: 40px; }"
            "QTabBar::tab:!selected { margin-top: 2px; }"
            )


        ######### 
        # hide tableWidget_settings_mechanics_errortab
        self.ui.tableWidget_settings_mechanics_errortab.hide()
        # hide tableWidget_settings_mechanics_contoursettings
        self.ui.tableWidget_settings_mechanics_contoursettings.hide()
        # hide groupBox_settings_mechanics_simulator
        self.ui.groupBox_settings_mechanics_simulator.hide()


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


        ##################### add Matplotlib figures in to frames ##########

        # # create an empty figure and move its toolbar to TopToolBarArea of main window
        # self.ui.mpl_dummy_fig = MatplotlibWidget()
        # self.addToolBar(Qt.TopToolBarArea, self.ui.mpl_dummy_fig.toolbar)
        # self.ui.mpl_dummy_fig.hide() # hide the figure

        # add figure mpl_sp[n] into frame_sp[n]
        for i in range(1, constant.max_harmonic+2, 2):
            # add first ax
            setattr(
                self.ui, 'mpl_sp' + str(i), 
                MatplotlibWidget(
                    parent=getattr(self.ui, 'frame_sp' + str(i)), 
                    xlabel='Frequency (Hz)', 
                    ylabel='Conductance (mS)', 
                    ylabel2='Susceptance (mS)',
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
            xlabel='Conductance (mS)',
            ylabel='Susceptance (mS)',
            )
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit_polar.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit_polar))

        # add figure mpl_spectra_fit into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            xlabel='Frequency (Hz)',
            ylabel='Conductance (mS)',
            ylabel2='Susceptance (mS)',
            showtoolbar=('Back', 'Forward', 'Pan', 'Zoom')
            )
        # self.ui.mpl_spectra_fit.update_figure()
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))

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



        ####### link functions  to UI ##########
        # set RUN/STOP button
        self.ui.pushButton_runstop.clicked.connect(self.on_clicked_pushButton_runstop)

        # set pushButton_resetreftime
        self.ui.pushButton_resetreftime.clicked.connect(self.reset_reftime)

        # set label_actualinterval value
        self.ui.lineEdit_acquisitioninterval.textEdited.connect(self.set_label_actualinterval)
        self.ui.lineEdit_refreshresolution.textEdited.connect(self.set_label_actualinterval)

        # set pushButton_gotofolder
        self.ui.pushButton_gotofolder.clicked.connect(self.on_clicked_pushButton_gotofolder)

        # set arrows (la and ra) to change pages 
        self.ui.pushButton_settings_la.clicked.connect(lambda: self.set_stackedwidget_index(self.ui.stackedWidget_spectra, diret=-1)) # set index -1
        self.ui.pushButton_settings_ra.clicked.connect(lambda: self.set_stackedwidget_index(self.ui.stackedWidget_spectra, diret=1)) # set index 1
        self.ui.pushButton_data_la.clicked.connect(lambda: self.set_stackedwidget_index(self.ui.stackedWidget_data, diret=-1)) # set index -1
        self.ui.pushButton_data_ra.clicked.connect(lambda: self.set_stackedwidget_index(self.ui.stackedWidget_data, diret=1)) # set index 1

        # set QAction
        self.ui.actionLoad_Settings.triggered.connect(self.on_triggered_load_settings)
        self.ui.actionLoad_Data.triggered.connect(self.on_triggered_load_data)
        self.ui.actionNew_Data.triggered.connect(self.on_triggered_new_data)
        self.ui.actionSave.triggered.connect(self.on_triggered_actionSave)
        self.ui.actionSave_As.triggered.connect(self.on_triggered_actionSave_As)
        self.ui.actionExport.triggered.connect(self.on_triggered_actionExport)
        self.ui.actionReset.triggered.connect(self.on_triggered_actionReset)

    ########### creating functions ##############

    def create_combobox(self, name, contents, box_width, row_text='', parent=''):
        ''' this function create a combobox object with its name = name, items = contents. and  set it't width. '''
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
        obj_box.setMaximumWidth(box_width)
        # insert to the row of row_text if row_text and parent_name are not empty
        if (row_text and parent):
            # find item with row_text
            item = parent.findItems(row_text, Qt.MatchExactly | Qt.MatchRecursive, 0)
            if len(item) == 1:
                item = item[0]
            else:
                return
            # insert the combobox in to the 2nd column of row_text
            parent.setItemWidget(item, 1, obj_box)
            


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
    def openFileNameDialog(self, title, path='', filetype=constant.default_datafiletype):  
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
 
    def saveFileDialog(self, title, path='', filetype=constant.default_datafiletype):    
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
        # import subprocess
        import os
        file_path = self.ui.lineEdit_datafilestr.text()
        path = os.path.abspath(os.path.join(file_path, os.pardir))
        # print(path)
        # subprocess.Popen(f'explorer "{path}"') # every time open a new window
        os.startfile(f'{path}') # if the folder is opend, make it active

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
        fileName = self.saveFileDialog(title='Choose a file and data type', filetype=constant.export_datafiletype) # !! add path of last opened folder
        # codes for data exporting

    def on_triggered_actionReset(self):
        # reset MainWindow
        pass

    def set_frame_layout(self, widget):
        '''set a dense layout for frame with a single widget'''
        vbox = QGridLayout()
        vbox.setContentsMargins(0, 0, 0, 0) # set layout margins (left, top, right, bottom)
        vbox.addWidget(widget)
        return vbox

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





if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qcm_app = QCMApp()
    qcm_app.show()
    sys.exit(app.exec_())
    