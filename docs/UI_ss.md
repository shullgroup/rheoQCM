
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

## groupBox_spectra

### pushButton_spectra_fit_refresh

| | signal|receiver|slot|note|
|-|----|----|----|----|
| |click()|mpl_spectra_fit    |`refesh_spectra_fit`|frefresh plot in mpl_spectra_fit|
| |click()|mpl_spectra_fit_polar    |`refesh_spectra_fit`|frefresh plot in mpl_spectra_fit_polar|

#### `refesh_spectra_fit`

`single_scan`  
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

#### `refesh_spectra_fit`  

read data from mpl_spectra_fit ax[0]: `f`, `Gp` and `Bp`  
`fit function` return parameters

plot mpl_spectra_fit: .l['Gfit'], .l['Bfit'], .l['lf'], .l['lg']  
plot mpl_spectra_fit_polar: .l['Pfit']  

