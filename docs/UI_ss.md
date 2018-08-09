
# initialize UI



# Widgets functions

## groupBox_settings

### pushButton_runstop

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |click()|pushButton_resetreftime                  |setDisabled|keep settings|
| |       |pushButton_newdata                       |setDisabled|keep settings|
| |       |pushButton_appenddata                    |setDisabled|keep settings|
| |       |treeWidget_settings_settings_hardware    |setDisabled|keep settings|
| |       |stackedWidget_spectra                    |setCurrentIn|view scans  |
| |       |stackedWidget_data                       |setCurrentIn|view data   |
| |       |                                         |`run function`| collecting data... |

#### `run function`


checking myVNA connection  
if True:  
>read parameters from `self.settings`  
set timer  
initiate `accvna` module  
start timer
timer loop:  
>> harmonic loop: [1, 3, 5, ...]
>>> set vna parameter  
get time  
get temp if `checkBox_control_rectemp` is checked AND harm == 1  
`single_scan`  
`get_data` (if Failed, try again)  
set frmae_sp[n]: .l['Gpre'] = .l['G'], .l['Bpre'] = .l['B'] and .l['Ppre'] = .l['P']  
clear frame_sp[n]: .l['Gfit'], .l['Bfit'], .l['Pfit'], .l['lf'], .l['lg'] and .l['G'] , .l['B'] , l['P']  
plot frame_sp[n]: raw to .l['G'] , .l['B'] , l['P'] 
>> harmonic loop: [1, 3, 5, ...]  
>>> fit data with parameters in `self.settings`  
plot frame_sp[n]: fit to .l['Gfit'] , .l['Bfit'] , l['Pfit'] .l['lf'], .l['lg']  
plot data in `frame_plt1`: check `comboBox_plt1_choice` and plot with right data form (func: ?)  
plot data in `frame_plt2`: check `comboBox_plt2_choice` and plot with right data form (func: ?)
save data to `self.data.samp` and append raw scan to `<file name>`  
`check span` for next scan and save it to `self.settings.`
>> `if pushButton_runstop.clicked():`  
>>> waite for timer  
>>> start progressBar_status_interval_time counting  
>>> update label_status_pts
>>
>> `elif pushButton_runstop.text() == 'Resume'`: unlicked, timer paused by other function  
>>> wait for `pushButton_runstop` clicked again to resume test

if False:
> set self to toggoled(False)  
> statusbar: failure information
> 

data file structure:  
```
# data file 
samp = {
    'index',
    'time',
    'f1',
    'g1',
    'f3',
    'g3',
    'temp', # if temp module established
    'marked',
}

# test_para
samp_para = {
    'f0',
    't0',
    'tshift',
    'ref', # file name of ref type
}
# ref may have defferent rows with samp
# if the ref is from external reference
# if the ref is from the same test with samp, they should have the same idxes and time pattern
ref = {
    'index',
    'time',
    'f1',
    'g1',
    'f3',
    'g3',
    'temp', # if temp module established
    'marked'
}

# has the same rows with samp
# more columns may add 
calc = {
    'index',
    'delf1',
    'delfcal1',
    'delg1',
    'delgcal1',
    'delf3',
    'delfcal3',
    'delg3',
    'delgcal3',
    'drho',
    'Grho',
    'phi',
    'dlam',
    'lamrho',
    'delrho',
    'delfdelfsn',
    'delgdelfsn',
    'rh',
    'rlam',
}

# save as dict
calc_para = {
    'nhcal',
    'nhplot'
}
```

Raw data file structure:  
```
# raw spectra in a separate file
# n is the index in the data file
raw = {
    index: {
        't': {},
        'samp': {
            1: {
                'f': {},
                'Gp': {},
                'Bp': {},  
            },
            3: {
                'f': {},
                'Gp': {},
                'Bp': {},
            },
        },
        'ref': { # if reference measured together
            '''
            ndarray or DataFrame of data
            f, Gp1, Bp1, Gp3, Bp3,...
            '''
        },
        'temp': [] # if temp module is established
    }
}

settings = {

}

```

## tabWidget_settings  

### tab_settings_control

#### comboBox_settings_control_scanmode
| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|label_settings_control_label1  |`switch_scanmode`|switch label between "Start (MHz)" and Center (MHz)|
| | |label_settings_control_label2  |`switch_scanmode`|switch label between "End (MHz)" and Span (MHz)|
| | |lineEdit_startf\<n>  |`switch_scanmode`|switch value between start and center|
| | |lineEdit_endf\<n>  |`switch_scanmode`|switch value between end and span|
| | | |`switch_scanmode`|set the mode of displaying values in lineEdit_start\<n> and _end\<n>|

#### pushButton_cntr\<n>
| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()| |`goto_cnetering(harm)`|set UI for fitting and starts a scan|

`goto_cnetering(harm)`

> set tabWidget_settings currentIndex(1) (settings)  
set stackedWidget_spectra currentIndex(1) (page_spectra_fit)  
set stackedWidget_data currentIndex(0) (page_data_data)  
set tabWidget_settings_settings_harm currentTabName('harm')  
set treeWidget_settings_settings_harmtree values to `harm` corresponding values  
start a `single_scan` and plot the data in mpl_spectra_fit  

#### pushButton_resetreftime

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()|lineEdit_reftime |`reset_reftime`|set current time as ref. time, save it to `self.settings` and `self.samp` and show it in lineEdit_reftime|

#### lineEdit_recordinterval
| | signal|receiver|slot|note|
|-|----|----|----|----|
| |textEdited()|lineEdit_refreshresolution |`set_lineEdit_scaninterval`||
| ||lineEdit_scaninterval |`set_lineEdit_scaninterval`||

`set_lineEdit_scaninterval`
> get int value of lineEdit_recordinterval and lineEdit_refreshresolution  
set lineEdit_scaninterval value = lineEdit_recordinterval / lineEdit_refreshresolution  
save values in those three widgets to `self.settings`  

#### lineEdit_refreshresolution

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |textEdited()|lineEdit_refreshresolution |`set_lineEdit_scaninterval`||
| ||lineEdit_scaninterval |`set_lineEdit_scaninterval`||

`set_lineEdit_scaninterval`
> get int value of lineEdit_recordinterval and lineEdit_refreshresolution  
set lineEdit_scaninterval value = lineEdit_recordinterval / lineEdit_refreshresolution  
save values in those three widgets to `self.settings`  

#### checkBox_control_rectemp  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked(bool)|checkBox_settings_temp_sensor |setChecked(bool)|done in Designer|
| |clicked(bool)|label_status_temp_sensor |`on_statechanged_set_temp_sensor`|enabled(bool)|


`on_statechanged_set_temp_sensor`

> if True:  
>> get all temp sensor setting parameters  
>> if the sensor is available:
>>> setChecked(True) checkBox_settings_temp_sensor  
setEnabled(True) label_status_temp_sensor  
save values of those three to `self.settings` 
>> else:  
>>> set self setChecked(False)  
show result in statusbar  
>  
> else
>> setChecked(False) checkBox_settings_temp_sensor  
setEnabled(False) label_status_temp_sensor  
save values of those three to `self.settings`  

#### pushButton_gotofolder

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()| |`on_clicked_pushButton_gotofolder`|open data the folder (lineEdit_datafilestr.text()) in a window|

#### pushButton_newdata  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()|lineEdit_datafilestr|`on_triggered_new_data`|show the new file path|
| |clicked()|lineEdit_reftime|`on_triggered_new_data`|readOnly(False), show current time as ref time|
| |clicked()|pushButton_resetreftime|`on_triggered_new_data`|setEnabled(True)|

`on_triggered_new_data`  
> if fileName:
>> show the new file path in lineEdit_datafilestr  
rest ref time `reset_reftime`  
set lineEdit_reftime eadOnly(False)  
set pushButton_resetreftime enabled  
save lineEdit_reftime to `self.settings`  
save filename to `self.fileName`  

#### pushButton_appenddata

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()|lineEdit_datafilestr|`on_triggered_load_data`|show the appended file path|
| |clicked()|lineEdit_reftime|`on_triggered_new_data`|readOnly(True), show time in fileName|
| |clicked()|pushButton_resetreftime|`on_triggered_new_data`|setEnabled(False)|

`on_triggered_new_data`  
> if fileName:
>> show the appended file path in lineEdit_datafilestr  
rest ref time `reset_reftime`  
set lineEdit_reftime eadOnly(True)  
set pushButton_resetreftime setEnabled(False)  
load settings in fileName to `self.settings`  
save filename to `self.fileName`  

### tab_settings_settings  

#### groupBox_settings_settings_harm  

##### tabWidget_settings_settings_harm  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentChanged()|treeWidget_settings_settings_harmtree|`update_harmonic_tab`|display value|

##### treeWidget_settings_settings_harmtree

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |valueChanged()||`updatesettings_settings_harmtree`|update values to `self.settings`|


values also inclued:  
comboBox_span_method  
comboBox_span_track  
comboBox_harmfitfactor  

##### pushButton_settings_harm_cntr

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked()|  |`goto_cnetering(harm)`|set UI for fitting and starts a scan|

`goto_cnetering(harm)`  
defined above  
get harm from treeWidget_settings_settings_harmtree.currentIndex()  

#### groupBox_settings_settings_hardwares

##### treeWidget_settings_settings_hardware

###### comboBox_sample_channel

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_samp_ref_chn`|check sample and reference channel selection and save them to `self.settings`|

`updatesettings_samp_ref_chn`  
> get comboBox_sample_channel selection  
get comboBox_ref_channel selection  
if samp == ref:  
>> set comboBox_ref_channel setCurrentIndex(0) ('--')  
> 
> save values of those two to `self.settings`  

###### comboBox_ref_channel  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_samp_ref_chn`|check sample and reference channel selection and save them to `self.settings`|

`updatesettings_samp_ref_chn`  
defined above  

###### comboBox_base_frequency  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`update_base_freq`|save value to `self.settings` and update frequency display (`update_frequencies`)|

###### comboBox_base_frequency  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`update_bandwidth`|save value to `self.settings` and update frequency display (`update_frequencies`)|

###### checkBox_settings_temp_sensor  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked(bool)|checkBox_control_rectemp|setChecked(bool))|done in Designer|
| |clicked(bool)|label_status_temp_sensor |`set_temp_sensor`|enabled(bool)|

`on_statechanged_set_temp_sensor`  
defined above  

###### comboBox_settings_settings_tempmodule  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`update_thrmcpltype`|save selection to `self.settings`|

###### comboBox_thrmcpltype

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`update_thrmcpltype`|save selection to `self.settings`|

#### groupBox_settings_settings_plots

##### treeWidget_settings_settings_plots

###### comboBox_timeunit

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_timeunit`|save selection to `self.settings` and resfresh figures in stackedWidget_data|

###### comboBox_tempunit

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_tempunit`|save selection to `self.settings` and resfresh figures in stackedWidget_data, label_status_temp_sensor|

###### comboBox_timescale

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_timeunit`|save selection to `self.settings` and resfresh figures in stackedWidget_data|

###### comboBox_gammascale

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |currentIndexChanged()|  |`updatesettings_timeunit`|save selection to `self.settings` and resfresh figures in stackedWidget_data|

###### checkBox_linktime

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |clicked(bool)| | |save selection to `self.settings` and resfresh figures in stackedWidget_data|



## groupBox_spectra

### pushButton_spectra_fit_refresh

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |click()|mpl_spectra_fit    |`refesh_spectra_fit`|frefresh plot in mpl_spectra_fit|
| |click()|mpl_spectra_fit_polar    |`refesh_spectra_fit`|frefresh plot in mpl_spectra_fit_polar|

#### `refesh_spectra_fit`

> `single_scan`  
 `get_data` (if Failed, try again)  
set mpl_spectra_fit:.l['Gpre'] = .l['G'] and .l['Bpre'] = .l['B']  
set mpl_spectra_fit_polar:.l['Ppre'] = .l['P']  
clear mpl_spectra_fit: .l['Gfit'], .l['Bfit'], .l['lf'], .l['lg'] and .l['G'] , .l['B']  
clear mpl_spectra_fit_polar: .l['Pfit'] and .l['P']  
plot mpl_spectra_fit: data to .l['G'], l['B']  
plot mpl_spectra_fit_polar: data to .l['P']  

### pushButton_spectra_fit_fit  

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |click()|mpl_spectra_fit    |`fit_spectra_fit`|fit data in mpl_spectra_fit|
| |click()|mpl_spectra_fit_polar    |`fit_spectra_fit`|fit data in mpl_spectra_fit|

#### `fit_spectra_fit`  

> read data from mpl_spectra_fit ax[0]: `f`, `Gp` and `Bp`  
`fit function` return parameters  
plot mpl_spectra_fit: .l['Gfit'], .l['Bfit'], .l['lf'], .l['lg']  
plot mpl_spectra_fit_polar: .l['Pfit']  

