'''
This is the main code of the QCM acquization program

'''

import sys
import datetime
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QActionGroup, QComboBox, QCheckBox, QTabBar, QTabWidget, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
# from PyQt5.uic import loadUi

# packages
from MainWindow import Ui_MainWindow
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

        # set displaying of harmonics
        self.ui.tabWidget_harm.setCurrentIndex(0)
        i = 1
        while True:
            try:
                if i <= constant.max_harmonic: # in the range to display
                    # set to visable which is default. nothing to do

                    # add checkbox to tabWidget_ham for harmonic selection
                    setattr(self.ui, 'checkBox_tree_harm' + str(i), QCheckBox())
                    self.ui.tabWidget_harm.tabBar().setTabButton(self.ui.tabWidget_harm.indexOf(getattr(self.ui, 'tab_settings_harm_' + str(i))), QTabBar.LeftSide, getattr(self.ui, 'checkBox_tree_harm' + str(i)))

                    # set signal
                    getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'checkBox_harm' + str(i)).setChecked)
                    getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked)
                    getattr(self.ui, 'checkBox_tree_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'sp' +str(i)).setVisible)
                    getattr(self.ui, 'checkBox_harm' + str(i)).clicked['bool'].connect(getattr(self.ui, 'sp' +str(i)).setVisible)
                    
                    if i in constant.default_harmonics: # in the default range 
                        # settings/control/Harmonics
                        getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(True)
                        getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(True)

                    else: # out of the default range
                        getattr(self.ui, 'checkBox_harm' + str(i)).setChecked(False)
                        getattr(self.ui, 'checkBox_tree_harm' + str(i)).setChecked(False)
                        # hide spectra/sp
                        getattr(self.ui, 'sp' + str(i)).setVisible(False)
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
                    getattr(self.ui, 'sp' + str(i)).setVisible(False)
                i += 2 
            except: 
                break
        
        max_gui_harmonic = i - 2 # maximum harmomic available in GUI

        # remove tabs in tabWidget_harm
        for i in range(constant.max_harmonic, max_gui_harmonic):
                # settings/settings/tabWidget_harm
                getattr(self.ui, 'tabWidget_harm').removeTab(int((constant.max_harmonic-1)/2)+1) # remove the same index

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
        self.ui.label_actual_interval.setText(str(constant.actual_interval) + '  s')
        self.ui.lineEdit_acquisition_interval.setText(str(constant.acquisition_interval))
        self.ui.lineEdit_refresh_resolution.setText(str(constant.refresh_resolution))

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

        # set treeWidget_harm_tree expanded
        self.ui.treeWidget_harm_tree.expandToDepth(0)
        # set treeWidget_settings_tree expanded
        self.ui.treeWidget_settings_tree.expandToDepth(0)
        # set treeWidget_data_tree_reference expanded
        self.ui.treeWidget_data_tree_reference.expandToDepth(0)
        
        ### add combobox into treewidget
        # comboBox_fit_method
        self.create_combobox('comboBox_fit_method', constant.span_mehtod_choose, 100, 'Method', self.ui.treeWidget_harm_tree)

        # add track_method
        self.create_combobox('comboBox_track_method', constant.track_mehtod_choose, 100, 'Tracking', self.ui.treeWidget_harm_tree)

        # insert sample_channel
        self.create_combobox('comboBox_sample_channel', constant.sample_channel_choose, 100, 'Sample Channel', self.ui.treeWidget_settings_tree)

        # insert base_frequency
        self.create_combobox('comboBox_base_frequency', constant.base_frequency_choose, 100, 'Base Frequency', self.ui.treeWidget_settings_tree)

        # insert bandwidth
        self.create_combobox('comboBox_bandwidth', constant.bandwidth_choose, 100, 'Bandwidth', self.ui.treeWidget_settings_tree)

        # insert refernence type
        self.create_combobox('comboBox_ref_type', constant.ref_type_choose, 100, 'Type', self.ui.treeWidget_data_tree_reference)

        # move center pushButton_settings_harm_cntr to treeWidget_harm_tree
        self.ui.treeWidget_harm_tree.setItemWidget(self.ui.treeWidget_harm_tree.findItems('Scan', Qt.MatchExactly | Qt.MatchRecursive, 0)[0], 1, self.ui.pushButton_settings_harm_cntr)
        # set the pushbutton width
        self.ui.pushButton_settings_harm_cntr.setMaximumWidth(50)

        
        # move center checkBox_settings_temp_sensor to treeWidget_settings_tree
        self.ui.treeWidget_settings_tree.setItemWidget(self.ui.treeWidget_settings_tree.findItems('Temperature', Qt.MatchExactly | Qt.MatchRecursive, 0)[0], 1, self.ui.checkBox_settings_temp_sensor)

        # set tabWidget_settings background
        self.ui.tabWidget_settings.setStyleSheet(
            # "QTabWidget, QTabWidget::pane, QTabBar { background: transparent; }"
            "QTabWidget::pane { border: 0;}"
            # "QTabWidget, QTabWidget::pane, QTabBar { border-width: 5px; border-color: red; }"
            # "QTabBar::tab-bar { background: transparent; }"
        )

        # set treeWidget_harm_tree background
        self.ui.treeWidget_harm_tree.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )
        # set treeWidget_settings_tree background
        self.ui.treeWidget_settings_tree.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )
        # set treeWidget_data_tree_reference background
        self.ui.treeWidget_data_tree_reference.setStyleSheet(
            "QTreeWidget { background: transparent; }"
        )

        # resize the TabBar.Button
        self.ui.tabWidget_harm.setStyleSheet(
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


        
        #### add widgets to status bar. from left to right
        # move label_status_coordinates to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_coordinates)
        # move progressBar_status_interval_time to statusbar
        self.ui.progressBar_status_interval_time.setAlignment(Qt.AlignCenter)
        self.ui.statusbar.addPermanentWidget(self.ui.progressBar_status_interval_time)
        # move label_status_signal_ch to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_signal_ch)
        # move label_status_reftype to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_reftype)
        # move label_status_temp_sensor to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_temp_sensor)
        # move label_status_f0BW to statusbar
        self.ui.statusbar.addPermanentWidget(self.ui.label_status_f0BW)


        ##################### add Matplotlib figures in to frames ##########

        # create an empty figure and move its toolbar to TopToolBarArea of main window
        self.ui.mpl_dummy_fig = MatplotlibWidget()
        self.addToolBar(Qt.TopToolBarArea, self.ui.mpl_dummy_fig.toolbar)
        self.ui.mpl_dummy_fig.hide() # hide the figure

        # add figure into frame_spactra_fit
        self.ui.mpl_spectra_fit = MatplotlibWidget(
            parent=self.ui.frame_spectra_fit, 
            xlabel='Frequency (Hz)',
            ylabel='Conductance (mS)',
            )
        self.ui.mpl_spectra_fit.update_figure()
        self.ui.mpl_spectra_fit.axes.plot([0, 1, 2, 3], [3,2,1,0])
        self.ui.frame_spectra_fit.setLayout(self.set_frame_layout(self.ui.mpl_spectra_fit))





        ####### link functions  to UI ##########
        # set RUN/STOP button
        self.ui.pushButton_run_stop.clicked.connect(self.on_clicked_pushButton_run_stop)

        # set pushButton_reset_reference_time
        self.ui.pushButton_reset_reference_time.clicked.connect(self.reset_reference_time)

        # set label_actual_interval value
        self.ui.lineEdit_acquisition_interval.textEdited.connect(self.set_label_actual_interval)
        self.ui.lineEdit_refresh_resolution.textEdited.connect(self.set_label_actual_interval)

        # set pushButton_new_data
        self.ui.pushButton_new_data.clicked.connect(self.on_triggered_new_data)

        # set pushButton_append_data
        self.ui.pushButton_append_data.clicked.connect(self.on_triggered_load_data)

        # set pushButton_goto_folder
        self.ui.pushButton_goto_folder.clicked.connect(self.on_clicked_pushButton_goto_folder)

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
    def on_clicked_pushButton_run_stop(self, checked):
        if checked:
            self.ui.pushButton_run_stop.setText('STOP')
        else:
            self.ui.pushButton_run_stop.setText('RUN')

    # @pyqtSlot(bool)
    def on_triggered_gropu_layout(self):
        if self.ui.group_layout.checkedAction() == self.ui.actionLayout_1: # basiview
            self.ui.stackedWidget_harmonic.setCurrentIndex(0)
            # set treewidget clapsed
            self.ui.treeWidget_harm_tree.collapseAll()
        elif self.ui.group_layout.checkedAction() == self.ui.actionLayout_2: #treeview
            self.ui.stackedWidget_harmonic.setCurrentIndex(1)
            # set treewidget all expanded
            # self.ui.treeWidget_harm_tree.expandAll()
            self.ui.treeWidget_harm_tree.expandToDepth(0)

    # @pyqtSlot()
    def reset_reference_time(self):
        ''' set time in lineEdit_reference_time '''
        current_time = datetime.datetime.now()
        self.ui.lineEdit_reference_time.setText(current_time.strftime('%Y-%m-%d %H:%M:%S'))

    # @pyqtSlot()
    def set_label_actual_interval(self):
        # get text
        acquisition_interval = self.ui.lineEdit_acquisition_interval.text()
        refresh_resolution = self.ui.lineEdit_refresh_resolution.text()
        #convert to flot
        try:
            acquisition_interval = float(acquisition_interval)
        except:
            acquisition_interval = 0
        try:
            refresh_resolution = float(refresh_resolution)
        except:
            refresh_resolution = 0
        # set label_actual_interval
        self.ui.label_actual_interval.setText(f'{acquisition_interval * refresh_resolution}  s')

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
        # change the displayed file directory in lineEdit_data_file_str
        self.ui.lineEdit_data_file_str.setText(fileName)
        # reset lineEdit_reference_time
        self.reset_reference_time()
        # set lineEdit_reference_time editable and enable pushButton_reset_reference_time
        self.ui.lineEdit_reference_time.setReadOnly(False)
        self.ui.pushButton_reset_reference_time.setEnabled(True)

    def on_triggered_load_data(self):
        fileName = self.openFileNameDialog(title='Choose an existing file to append') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_data_file_str
        self.ui.lineEdit_data_file_str.setText(fileName)
        # set lineEdit_reference_time
        # set lineEdit_reference_time read only and disable pushButton_reset_reference_time
        self.ui.lineEdit_reference_time.setReadOnly(True)
        self.ui.pushButton_reset_reference_time.setEnabled(False)

    # open folder in explorer
    # methods for different OS could be added
    def on_clicked_pushButton_goto_folder(self):
        # import subprocess
        import os
        file_path = self.ui.lineEdit_data_file_str.text()
        path = os.path.abspath(os.path.join(file_path, os.pardir))
        # print(path)
        # subprocess.Popen(f'explorer "{path}"') # every time open a new window
        os.startfile(f'{path}') # if the folder is opend, make it active

    # 
    def on_triggered_load_settings(self):
        fileName = self.openFileNameDialog('Choose a file to use its setting') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_data_file_str
        self.ui.lineEdit_data_file_str.setText(fileName)

    def on_triggered_actionSave(self):
        # save current data to file
        print('save function  to be added...')

    def on_triggered_actionSave_As(self):
        # save current data to a new file 
        fileName = self.saveFileDialog(title='Choose a new file') # !! add path of last opened folder
        # change the displayed file directory in lineEdit_data_file_str
        self.ui.lineEdit_data_file_str.setText(fileName)

    def on_triggered_actionExport(self):
        # export data to a selected form
        fileName = self.saveFileDialog(title='Choose a file and data type', filetype=constant.export_datafiletype) # !! add path of last opened folder
        # codes for data exporting

    def on_triggered_actionReset(self):
        # reset MainWindow
        pass

    def set_frame_layout(self, widget):
        '''set a dense layout for frame with a single widget'''
        vbox = QVBoxLayout()
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
    