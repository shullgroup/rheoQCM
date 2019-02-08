# Changelog

## [0.15.3] - 2019-01-31

### Fixed

- Fix the manual refit mode turning on/off bugs.
- Fix indexing bug while picking single point from data with marks.
- Fix refit not updating bug.
- Fix devided peak not plotting bug.

## [0.15.2] - 2019-01-29

### Changed

- Change the peak tracking strategy. The changing setp sometimes might be too big.
- Change the saving protocol while testing. Save both raw and data after each data acquization.

### Fixed

- Fix the fitting range line bug when fitting is failed
  
## [0.15.1] - 2019-01-18

### Added

- Add setting frequency range to myVNA for display data in it, in case it is needed to check data out of Python program.
- Add base frequency check for property claculation in UI. Now, the base frequency set in UI will be used as f1 when first harmonic data is not collected.
- Add load calibration file when switch channel

### Fixed

- Fix program crashes when collecting data with VNA disconnected on Windows  
  
## [0.15.0] - 2019-01-15

### Added

- Add "CHANGELOG.md" file  
  
### Changed

### Fixed

### Removed