[![DOI](https://zenodo.org/badge/138771761.svg)](https://zenodo.org/badge/latestdoi/138771761)

# QCM Data Collection and Analysis Software

This is the Python project page for the QCM data collection and analysis software used by the Shull research group at Northwestern University. The data collection and analysis are at the testing point. Curentlly, it is using its own data format (hdf5). The data importing function with QCM-D data is useful to the community doing QCM-D tests. Some of the analysis routines are generally useful, regardless of how the QCM data were generated.

## Getting Started

The analysis portions of the software should work on Windows/Mac/Linux platforms. In all cases you'll need some familiarity with running commands from the terminal, however. It's assumed in the following that you know how to do this on your platform. The software to interface with network analyzers and collect the QCM data only runs on Windows-based computers (The analyser currently interfaced with only works on Windows).

### Capabilities

* Graphical data interface to collect QCM data with network analyzers.  
* Fast data recording without openning the dependent external software. Fewer resources are required than in previous MATLAB-based versions of the software.  
* Data collection and analysis are combined in one package.  
* Other variables (Temperature, for example) can be simultaneously recorded and saved with the QCM data. (with NI devices)

### Prerequisites

* Python 3.5+ is required. For data analysis only, it can run with both 32-bit and 64-bit Python. If you want to use the data collection portion with myVNA, 32-bit Python and Windows are required.  

* Hardware and external software for data collection: The AccessMyVNA and myVNA programs were obtained from <http://g8kbb.co.uk/html/downloads.html>.
* Python labroaries needed to run the software are listed in the `requirements.txt` file.  
  
* The Anaconda python environment is suggested.  You can  download and install the Anaconda distribution of python from [anaconda.com](https://anaconda.com/download).  

* Separated scripts works with data stored in a MATLAB-compatible .mat files (collected by our [Matlab data collecting program](https://github.com/Shull-Research-Group/QCM_Data_Acquisition_Program)).  In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):  

```bash
conda install -c conda-forge hdf5storage  
```

### Installation

To install everything you need from this repository, run the following command from a command window in the directory where you want everthing to be installed:

```bash
git clone https://github.com/zhczq/QCM_py
```

If you just need the updated analysis script, everything you need is in QCMFuncs/QCM_functions.py. In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py. You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

All the modules needed for the data collection program are in the `QCM_main/` folder. Go to that folder and run QCM_main.py will open the program.  

## Using the Stand-alone Program (exe)

A single executable file (exe file) which is precompiled from source code under 32-bit Python is located in the `stand_alone/` folder. It can be run without any additional installation on a Windows system, without the need to install a Python distribution. It is convienent for data collection in case you have 64-bit Python installed on your Windows, previously. (<span style="color:red">NOTE: You can customize the program with the  `settings_default.json`  file that comes with the executable file (See below about the way it works.)</span>)  

## Running the UI from Terminal

Go to the `QCM_mani/` folder and run `QCM_main.py` will start the UI and it will check the environment by itself.  

## Using UI with QCM-D Data

* Export the QCM-D data as .xlsx file. column names: t(s), delf1, delg1, delf3, delg3, ... The time column name could also be time(s).
* Start the UI and from the menu bar and select `File>Import QCM-D data`.  This will import the QCM-D data and save a .h5 file with the same name in * the same folder. This will save all the calculated property data for future use.  
* Now the UI can display your data and do the analysis the same as the data generated with the UI.
* Don't forget to save the data when you finish the calculation.
* Click export to export a .xlsx file with all the data in it.

## Using Analysis Code for Mat File  

If you just need the updated analysis code for .mat files, everything you really need is in `QCMFuncs/QCM_functions.py`. In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):  

```bash
conda install -c conda-forge hdf5storage  
```

In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py. You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

## Using Analysis Code for h5 File  

The `QCM_functions.py` code also works with .h5 data file collected by the UI of this project. The way to define the samples is similar as it of .mat files. Example files (`example_plot.py` and `example_sampledefs.py`) which demostrate both .mat and .h5 analysis with `QCM_functions.py` can be found in  `QCMFuncs/`.

## Documentation

The QCMnotes.pdf file has some background information on the rheometric mode of the QCM that we utilize, with some useful references included.

Modules `DataSaver` and `QCM` in `Modules/` folder are availabe for dealing with the data and doing ananlysis manually. Those modules include functions run out of the software. You can simply import those modules and do almost the same thing in the software by running your own codes. An example code of extrating data from data file can be found in `example/` folder.

The functions for Matlab version data are locoalized in `QCMFuncs/` folder.  

Export the current settings as a json file named `settings_default.json` and save in the same folder as `QCM_main.py` or `QCM_main.exe`. The UI will use the settings you saved as default after the next time you opend it. If you want the setup go back the original one, just simply delete or rename that file.  

There is a `QCM_main.bat` file in  `QCM_main/` for running the program with Python by just double clicking it. You need to change the path of python and QCM_main.py to them on your computer to make it work. Meanwhile, you can make a shortcut of this bat file and put the shortcut in a location of your choosing.

### Known Issues

* Please set MyVNA to `Basic Mode` from the left pannel of MyVNA software by selecting VNA Hardware>Configure CSD / Harmonic Mode and checking Basic Mode in Select Mode. This will make sure the time counting in the Python program fits the hardware scanning time. You will not loose any precision as far as we know.  

* The data analysis in the UI only works with `one layer` mode. Other modes, including films immersed in a liquid medium, will be added in the near future.

## To Do List (work in Progress)

* Documentation.
* Property results plotting and exporting.
* Interface with other hardware. (If you have a hardware and interested in interfacing with our software, please feel free to contact us.)

## Authors

* **Qifeng Wang**  - *Primary Developer of the current (Python) version of this project*
* **Megan Yang**  - *Developer of the current (python) version of this project*
* **Kenneth R. Shull** - *Project PI and author of some of the specific functions used in the project*

## Other Versions

If you are a MATLAB user, our previously developed MATLAB version software can be found here: <https://github.com/Shull-Research-Group/QCM_Data_Acquisition_Program>. It was developed Josh Yeh. This Python project is based on this previous MATLAB version developed by Josh.  

A MATLAB version of our data analysis software, written by Kazi Sadman can be found here: <https://github.com/sadmankazi/QCM-D-Analysis-GUI>.

## Acknowledgments

* Josh Yeh
* Diethelm Johannsmann
* Lauren Sturdy
* Ivan Makarov
