# QCM_main script

## QCMApp Class methods

### link_tab_page(tab_idx)

### create_combobox(name, contents, box_width, row_text='', parent='')

### move_to_col2(self, obj, parent, row_text, width=[])

### find_text_item(self, parent, text)

### set_frame_layout(self, widget)

### openFileNameDialog(self, title, path='', filetype=settings_init['default_datafiletype'])

### saveFileDialog(self, title, path='', filetype=settings_init['default_datafiletype'])

### on_dateTimeChanged_dateTimeEdit_reftime

get time in dateTimeEdit_reftime and save it to self.settings  

### update_vnachannel

update vna channels (sample and reference) if ref == sample: sample = 'none'  

### on_triggered_new_data(self)

### on_triggered_load_data(self)

### on_clicked_pushButton_gotofolder(self)

### on_triggered_load_settings(self)

### on_triggered_actionSave(self)

### on_triggered_actionSave_As(self)

### on_triggered_actionExport(self)

### on_triggered_actionReset(self)

### on_changed_slider_spanctrl(self)

### span_check(self, harm, f1=None, f2=None)

### on_clicked_pushButton_spectra_fit_refresh(self)

### on_fit_lims_change(self, axes)

### update_lineedit_fit_span(self, f)

### spectra_fit_axesevent_disconnect(self, event)

### spectra_fit_axesevent_connect(self, event)

### mpl_disconnect_cid(self, mpl)

### mpl_connect_cid(self, mpl, fun)

### mpl_set_faxis(self, ax)

### on_clicked_set_temp_sensor(self, checked)

### statusbar_temp_update(self)

### temp_by_unit(self, data)

### set_stackedwidget_index(self, stwgt, idx=[], diret=[])

### update_widget(self, signal)

### update_harmwidget(self, signal)

### update_harmonic_tab(self)

### update_base_freq(self, base_freq_index)

### update_bandwidth(self, bandwidth_index)

### statusbar_f0bw_update(self)

### update_freq_range(self)

### check_freq_span(self)

### update_frequencies(self)

### update_freq_display_mode(self, signal)

### on_editingfinished_harm_freq(self)

### set_default_freqs(self)

### update_spanmethod(self, fitmethod_index)

### update_spantrack(self, trackmethod_index)

### update_harmfitfactor(self, harmfitfactor_index)

### update_samplechannel(self, samplechannel_index)

### update_refchannel(self, refchannel_index)

### update_module(self, module_text)

### update_tempsensor(self)

### update_tempdevice(self, tempdevice_index)

### update_thrmcpltype(self, thrmcpltype_index)

### set_label_temp_devthrmcpl(self)

### update_timeunit(self, timeunit_index)

### update_tempunit(self, tempunit_index)

### update_timescale(self, timescale_index)

### update_yscale(self, yscale_index)

### update_linktime(self)

### load_comboBox(self, comboBox, choose_dict, parent=None)

### update_guichecks(self, checkBox, name_in_settings)

### log_update(self)

### load_settings(self)

### check_freq_range(self, harmonic, min_range, max_range)

### smart_peak_tracker(self, harmonic, freq, conductance, susceptance, G_parameters)

### read_scan(self, harmonic)



# MathModules module

## custom functions

### datarange(data)

### num2str(A,precision=None)

### findpeaks(array, output, sortstr=None, npeaks=np.inf, minpeakheight=-np.inf, threshold=0, minpeakdistance=0, widthreference=None, minpeakwidth=0, maxpeakwidth=np.inf)

### converter_startstop_to_centerspan(f1, f2)


# AccessMyVNA module

## custom functions

### check_zero(result, func, args)

### get_hWnd(win_name)

### get_pid(hWnd):

### close_vna()

## AccessMyVNA Class methods

### Init(self)

### Close(self)

### ShowWindow(self, nValue=1)

### GetScanSteps(self)

### SetScanSteps(self, nSteps=400)

### GetScanAverage(self)

### SetScanAverage(self, nAverage=1)

### GetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9)

### SetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9, nData=[])

### Getinstrmode(self)

### Setinstrmode(self, nMode=0)

### Getdisplaymode(self)

### Setdisplaymode(self, nMode=0)

### SingleScan(self)

### EqCctRefine(self)

### SetFequencies(self, f1=4.95e6, f2=5.05e6, nFlags=1)

### GetScanData(self, nStart=0, nEnd=299, nWhata=-1, nWhatb=15)

### Autoscale(self)

### single_scan(self)

### change_settings(self, refChn=1, nMode=0, nSteps=400, nAverage=1)

### set_steps_freq(self, nSteps=300, f1=4.95e6, f2=5.00e6)

### etADCChannel(self, refChn)