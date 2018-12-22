[![DOI](https://zenodo.org/badge/138771761.svg)](https://zenodo.org/badge/latestdoi/138771761)

# QCM Data Collection and Analysis Software

This is the Python project page for the QCM data collection and analysis software used by the Shull research group at Northwestern University. The data collection and analysis are at the testing point. Curentlly, it is using its own data format (hdf5). The data importing function will be added soon which will be useful to the community doing QCM-D tests. Some of the analysis routines are generally useful, regardless of how the QCM data were generated.


## Getting Started

The analysis portions of the software should work on Windows/Mac/Linux platforms. In all cases you'll need some familiarity with running commands from the terminal, however.  It's assumed in the following that you know how to do this on your platform.  The software to interface with network analyzers and collect the QCM data only runs on Windows-based computers.

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

* QCM data files are currently stored in a MATLAB-compatible .mat files.  In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):  

```bash
conda install -c conda-forge hdf5storage  
```

### Installation

To install everything you need from this repository, run the following command from a command window in the directory where you want everthing to be installed:

```bash
git clone https://github.com/zhczq/QCM_py
```

If you just need the updated analysis software, everything you need is in QCMFuncs/QCM_functions.py.  In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py.  You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

All the modules needed for the data collection program are in the `QCM_main/` folder. Go to that folder and run QCM_main.py will open the program.  

## Using Analysis Program

To see an example of how the program works, run example_runfile.py, which will act on some of the data in the example_data directory and generate some figures representing the analyzed data in a figures folder.  

## Using Analysis Code for Mat File  

If you just need the updated analysis code for .mat files, everything you really need is in `QCMFuncs/QCM_functions.py`. In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):  

```bash
conda install -c conda-forge hdf5storage  
```

In this same directory you will also find some example data obtained with polystyrene films at different temperatures, and a run file called PS_plots.py. You should be able to run PS_plots.py directly and end up with some property plots that illustrate how the process works, and how you need to set up the file structure to include your own data.

## Documentation

The QCMnotes.pdf file has some background information on the rheometric mode of the QCM that we utilize, with some useful references included.

Modules `DataSaver` and `QCM` in `Modules/` folder are availabe for dealing with the data and doing ananlysis manually. Those modules include functions run out of the software. You can simply import those modules and do almost the same thing in the software by running your own codes.

The functions for Matlab version data are locoalized in `QCMFuncs/` folder.  

## To Do List (work in Progress)

* Documentation.
* Property results plotting and exporting.
* Data analysis with QCM-D data.
* Compile to a single executable file for those who don't want to run programs from the command line.
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
