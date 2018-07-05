# QCM Data Collection and Analysis Software

This is the project page for the QCM data collection and analysis software used by the Shull research group at Northwestern University. The data collection part is very much under construction at the moment, and is not yet at a point where it is useful to the community.  Some of the analysis routines are potentially useful, however, and these are described below.

## Getting Started

The analysis portions of the software should work on Windows/Mac/Linux platforms.  In all cases you'll need some familiarity with running commands from the terminal, however.  It's assumed in the following that you know how to do this on your platform.  The softare to interface with the network analyzers only runs on Windows-based computers.

### Prerequisites

* Download and install the Anaconda distribution of python from [anaconda.com](https://anaconda.com/download).
* QCM data files are currently stored in a MATLAB-compatible .mat files.  In order to read and write these and get the analysis scripts to work, you need to install the hdf5storage package, which you can add with the following command (assuming you have already added the conda python distribution):
```
conda install -c conda-forge hdf5storage 
```

### Installation

To install everything you need from this repository, run the following command from a command window in the directory where you want everthing to be installed:

```
git clone https://github.com/zhczq/QCM_py
```

If you just need the updated analysis software (which currently the only software that actually works) you'll want to download QCM_functions.py

## Using the Analysis Program

To see an example of how the program works, run example_runfile.py, which will act on some of the data in the example_data directory and generate some figures representing the analyzed data in a figures folder. 

## Documentation

The QCMnotes.pdf file has some background information on the rheometric mode of the QCM that we utilize, with some useful references include.

## Authors

* **Josh Yeh** - *Initial developmen of the MATLAB version of this project*
* **Qifeng Wang**  - *Primary Developer of the current (python) version of this project*
* **Kenneth R. Shull** - *Project PI and author of some of the specific functions used in the project*

## License

We still need to figure out the right license.  https://choosealicense.com/ might help.

## Acknowledgments

* Diethelm Johannsmann
* Lauren Sturdy

