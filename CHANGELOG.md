# Changelog

## [0.21.0] - 2023-02-16

### Added

- Add auto analyzer icon change by selected device.
- Add code to convert binary to str for h5py >= 3.0.0

### Changed

- Change `__version_info__` from string to int for easier comparison.
- Change `mech_key` to the form separating haromincs with ".". e.g.: in older versions is "355"; in the new version is "3.5.5". You can use DataSaver.get_mech_key(your_mech_key) to convert it to the version compatible str.
- Change np.float and 'float64' to `float` to fit numpy >= 1.20
- Change the method share x/y in Matplotlib since join was removed since 3.6

### Fixed

- Fix bugs of combining lmfit models.

### Removed

## [0.20.1] - 2022-10-20

### Added

- Add `__main__.py` to make the program can be ran with it's name `rheoQCM'.

### Changed

- Change the number of samples (for each scan) to nSteps + 1 which is consistent with it in myVNA software.

### Fixed

- Fix PeakTracker bugs in peak_tracker() function.

### Removed

## [0.20.0] - 2022-02-11

### Added

- Add multiprocessing to property calculating to utilize multicores of cpu. Set the cores to use in `multiprocessing_cores`.

- Add `lmfit` fitting to `QCM` module. NOTE: the Error for lmfit method is different to `scipy` (default) which considers input errors from measurements.

- Add `Refresh table` button to manually update property table contents.

### Changed

- Improve peak fitting function.
- Change functions in QCM module to make it easier to use out of the GUI.

### Fixed

- Fix multiprocessing bug where the retruned variables were placed in wrong order. This bug made the calulated overlayer values (grho, phi, drho) in the wrong order (drho, grho, phi) and made the follwing calculation failed and the program would just use the results from SLA guesses.

- Fix mpl_sp_[harm] plots x/y labels when plot polor plots.  

### Removed

## [0.19.4] - 2021-05-05

### Added

- Add auto expand mechanic table when countour unchecked.
- Add "least_squares" choice for Gp/Bp peak fitting. Default is "leastsq". "leastsq" was the only choise in previous versions. "leastsq" should run faster than "least_squares"
- Add "prop_plot_ncols" in setings to customize the number of columns of property plots.

### Changed

- Clean PeakTracker code.
- Change some items' size policy in UI.
- Change time.sleep() to threading.Thread() in AccessMyVNA.py for better UI runs.

### Fixed

- Fix chrashing while exporting data with empty channel(s).
- Fix variable name errors in rheoQCM
- Fix comparability with Matplotlib >= 3.3: picker, errorbar.
- Fix the property plot scrollArea does not reset after clearing plots.

## [0.19.3] - 2020-11-06

### Added

### Changed

### Fixed

- Fix importing external QCM data (QCM-D) error.

### Removed

## [0.19.2] - 2020-08-31

### Added

- Add `Delta F_{exp}/n`, `Delta F_{cal}/n` in property table. <span style="color:red">NOTE</span>: You may need to recalculate the data from previous version to show data in these two rows.

### Changed

### Fixed

- Fix not reading user settings file bug.
- Fix file name did not clear in main class after UI reset.
- Fix nan replacing bug for reading data.

### Removed

## [0.19.1] - 2019-11-05

### Added

### Changed

### Fixed

- Fix crashing by loading file with empty channel.

### Removed

## [0.19.0] - 2019-11-05

### Added

- Add function for regenerate data from raw. This is usefull for the cases when the data structure messed up but you still keep the raw data in raw.
- Add functions to display Sauerbrey mass.

### Changed

- Change the reference data structure.
- Change the config/settings loading method and make it works for all modules

### Fixed

- Fix C++ source code bug of GetDoubleArray and make the functiion works to check available channels by checking the settings from myVNA.
- QCM module solve property guess check bug (drho check).
- Fix span < 1 Hz stopped myVNA bug by adding minimum limit to peak_tracker.
- Correct grho function in QCM module.

### Removed

## [0.18.2] - 2019-09-11

### Added
  
### Changed

- Change the way to save string in h5 file to prevent the file size increasing while repeated saving.

### Fixed

- Fix property guess out of range error from ver 0.18.1.

### Removed

## [0.18.1] - 2019-08-30

### Added

- Add format settings for property table.
- Add alert to data delete contex menu.

### Changed

- Change bulk calculation and add errors to the solutions.

### Fixed

- Correct some text in UI.
- Fix raw not find bug.
- Fix reference calculation bug.

### Removed

## [0.18.0] - 2019-08-09

### Added

- Add contour plots.
- Add slove new property to save the time from solving all.
- Add D (dissipition from QCM-D) to calculated properties.  
- Add a simple function to mark data in linear or log.
 
### Changed

- Change code to that property calculation will not remove previous data. <span style="color:red">NOTE</span>: If you want to recalculate previous data, make sure use `slove all`.
- Change exported Excel file format. The columns with repeated data are combined.
- Change HOME button of data plots to show all & auto scale.  
- Change debuging to logger. 
- Change the name `settings_ini` to `config_default` for the basic settings.

### Fixed

- Fix data plot does not change limit after ZOOM OR PAN.
- Fix issues of peak tracking condition settings with constrained conditions didn't work. <span style="color:red">NOTE</span>: When `Fix center` is set, the peak may come of the scan range!
  
### Removed

## [0.17.2] - 2019-07-05

### Added

- Add code to reset peak track mark in spectra plots before auto refitting peaks.

### Changed

- Change the mechanical column name of delf/delf_sn, delg/delf_sn. <span style="color:red">NOTE</span>: if you want the data of those two, you need to recalculate the data.

### Fixed

- Fix manual fit bug when no peak found.
- Fix bug when solving property of a single point data.

### Removed

## [0.17.1] - 2019-06-19

### Added

- Add customized "extra wait time" for AccessMyVNA.

### Changed

### Fixed

- Fix no "cal" path bug. TODO: Still needs to add the case no calibration file found.
- Fix manual peak fitting crashes when no peak is found.

### Removed

## [0.17.0] - 2019-05-29

### Added

- Add spinBox for selecting index for displaying property in table.  
- Add sample description.
- Add scipy functions to peak factors guess.
- Add viscocity to property calculation.

### Changed

- Change data structure. Copy some keys in settings_init to settings_default to keep enough information of data collecting parameter.  
- Change the way to get display data from all harmonic together to by harmonics.
- Change multiple peak fitting to use separate data ranges. The range plotted in the figures are still a connected range.
- Change variable names with "rh", reference harmonics, to "refh". This resulted in exported excel file column names change.
- Change the data reference structure to make both channels use separated references in all conditions.
- Change mechanical calculation protocal. It is still in a testing version.
- Change recording interval set up method. All three factors (record interval, refresh resolution, scan interval) can be changed.

### Fixed

- Fix property of none calculated index value which is signed to value from other rows.  
- Fix harmonic setting items update bugs.

### Removed

## [0.16.1] - 2019-03-14

### Added

- Add manual fitting report to UI
- Add export raw to right click menu in data plots.
- Add fix phi=0 for the conductance/sursceptance peak fitting.
- Add function that record data without fitting.

### Changed

- Change reference claculating strategy.
- Change property table colunms auto generated by max harmonic.

### Fixed

### Removed

## [0.16.0] - 2019-02-27

### Added

- Add window state check before resize it.
- Add text fontsize to customize settings.
- Add calibration file check.
- Add error log function.
- Add key check to user settings (Add missed key from settings_default)

### Changed

- Change the harmonic related widgets to be generated by program. (Leave only the first manually added in UI.)
- Change the harmonic related widgets settings and loading methods.
- Change DataFrame filter to regex to read data from a given harmonic.
- Change the way import data from other software (e.g.: QCM-D) to make it possible to customize the format for import and applicable for different formats (QCM-D and QCM-Z).
- Change channel selection when the same analyzer channel selected.
- Hide reference time items to simplify the setup process. The reference time will always be the first time starting the experiment.
- Improve the peak tracking method.

### Fixed

- Fix the span setting failure when peak tracking failed.
- Fix temp lines deleting bug.
- Fix delete selected raw data bug when it doesn't exist.

### Removed

## [0.15.3] - 2019-02-07

### Added

### Changed

### Fixed

- Fix the manual refit mode turning on/off bugs.
- Fix indexing bug while picking single point from data with marks.
- Fix refit not updating bug.
- Fix devided peak not plotting bug.

### Removed

## [0.15.2] - 2019-01-29

### Added

### Changed

- Change the peak tracking strategy. The changing setp sometimes might be too big.
- Change the saving protocol while testing. Save both raw and data after each data acquization.

### Fixed

- Fix the fitting range line bug when fitting is failed

### Removed

## [0.15.1] - 2019-01-18

### Added

- Add setting frequency range to myVNA for display data in it, in case it is needed to check data out of Python program.
- Add base frequency check for property claculation in UI. Now, the base frequency set in UI will be used as f1 when first harmonic data is not collected.
- Add load calibration file when switch channel

### Changed 

### Fixed

- Fix program crashes when collecting data with VNA disconnected on Windows  

### Removed

## [0.15.0] - 2019-01-15

### Added

- Add "CHANGELOG.md" file  
  
### Changed

### Fixed

### Removed