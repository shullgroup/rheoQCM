// AccessMyVNAdll.h : main header file for the AccessMyVNAdll DLL
//
// Remote access to myVNA via OLE automation
// the DLL is designed to be called from programs
// such as Dephi that can't access DLLs properly
//
// To use it, call the function MyVNAInit() first
// and make sure you call the function MyVNAClose() before
// the calling window application closes or it may hang
// the Iiit call will start up a copy of myVNA which is then closed
// by the CLose call
//
// There are a number of exported functions descibed below that
// may be used to control the applicaiton
// In general these return 0 if the call succeeded but in some cases


#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

class CAutoMyVNA : public COleDispatchDriver
{
// Attributes
public:

// Operations
public:
};


// CAccessMyVNAdllApp
// See AccessMyVNAdll.cpp for the implementation of this class
//

class CAccessMyVNAdllApp : public CWinApp
{
public:
	CAccessMyVNAdllApp();
	~CAccessMyVNAdllApp();

// Overrides
public:
	DECLARE_MESSAGE_MAP()
	virtual BOOL InitInstance();

public:
	CAutoMyVNA *m_autoMyVNAObject;
	virtual int ExitInstance();
};

// private functions used as a helper to exported functions
int MyVNAGetHelper( _TCHAR * pString, enum VARENUM, void *pnResult );
int MyVNASetHelperInt( _TCHAR * pString, enum VARENUM type, int Value );
int MyVNASetHelperDouble( _TCHAR * pString, enum VARENUM type, double Value );
int MyVNACallFunctionVoid( _TCHAR * pString );
int MyVNACallFunctionBstr( _TCHAR * pStrFunction, _TCHAR * pStrParameter );
int MyVNAGetIntArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, int *pnResult );
int MyVNASetIntArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, int *pnResult );
int MyVNAGetDoubleArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, double *pnResult );
int MyVNASetDoubleArrayHelper( _TCHAR * pString, int nWhat, int nIndex, int nArraySize, double *pnResult );
int MyVNASetStringHelper( _TCHAR * pString, int nWhat, int nIndex, _TCHAR * sWhat );
CString MyVNAGetStringHelper( _TCHAR * sWhat, int nWhat, int nIndex );

// /////////////////////////////////////////////////////////////////////////////////////////////////
// exported functions
// this is the interface provided

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAInit()
// call this function befoe trying to do anything else. It attempts to execute myVNA
// and establish it as an automation server
// OLE equivalent:
// Connect to the server using the CLSID
__declspec(dllexport) int _stdcall MyVNAInit(void);

// Make sure you call the Close function before the windows GUI application is closed
// OLE equivalent:
// simply release the interface
__declspec(dllexport) int _stdcall MyVNAClose(void);


// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASingleScan()
// attempt to single scan the VNA. On completion myVNA will post a message ( Message) to the queue of the specified
// window ( hWnd ) with the given COmmand and lParam values. See the example in AccessMyVNADlg.cpp
// OLE equivalent:
// int SingleScanAutomation(LONG Message, HWND hWnd, LONG lParam, LONG wParam );
__declspec(dllexport) int _stdcall MyVNASingleScan(int Message, HWND hWnd, int nCommand, int lParam );

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAEqCctRefine()
// attempt to refine an equivalent circuit. On completion myVNA will post a message ( Message) to the queue of the specified
// window ( hWnd ) with the given Command and lParam values. See the example in AccessMyVNADlg.cpp
// OLE equivalent:
// int RefineAutomation(LONG nEnd, LONG hWnd, LONG Command, LONG lParam);
__declspec(dllexport) int _stdcall MyVNAEqCctRefine(int Message, HWND hWnd, int nCommand, int lParam );


// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASetScanSteps()
// sets the number of steps ( min 2 max 50,000 but note this can take a huge amount of memory)
// OLE equivalent:
// Set property nScanSteps
__declspec(dllexport) int _stdcall MyVNASetScanSteps(int nSteps);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAGetScanSteps()
// read the current scan steps value
// OLE equivalent:
// Get property nScanSteps
__declspec(dllexport) int _stdcall MyVNAGetScanSteps(int *pnSteps);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASetScanAverage() and MyVNAGetScanAverage()
// set / get the current scan trace average count
// OLE equivalent:
// Set / Get property nAverage
__declspec(dllexport) int _stdcall MyVNASetScanAverage(int nAverage);
__declspec(dllexport) int _stdcall MyVNAGetScanAverage(int *pnAverage);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASetFequencies
// nFlags bits 0..3 set scan mode where f1 is first parameter / f2 is the second
// 0 = centre / span
// 1 = start / stop
// 2 = full scan
// 3 = from zero
// 4 = centre / per division
// 5 = centre / per step
// 6 = start / per step
// 7 = start / per division
// 8..15 unused
//
// return codes:
// 0 - OK
// 1 - start too low
// 2 - stop too high
// 3 - width must be > 0
// 4 - start is before end
// 5 - stop too low
// OLE equivalent:
// int SetScanDetailsAutomation(double dF1, double dF2, LONG nFlags);
__declspec(dllexport) int _stdcall MyVNASetFequencies(double f1, double f2, int nFlags);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAShowWindow()
// show or hide the current myVNA window.
// nValue set to 0 causes SW_SHOWDEFAULT
// nValue set to 1 causes SW_SHOWMINIMIZED
// all other values ignored
// OLE equivalent:
// void ShowWindowAutomation(LONG nValue);
__declspec(dllexport) int _stdcall MyVNAShowWindow(int nValue);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAGetScanData()
// get current scan data results starting from scan point nStart up to and including nEnd
// note that a scan of N steps gives N+1 data points from 0 to N
// the data type requsted( nWhat) must be one of the ones as follows. 
// It is the callers' responsibility
// to make sure that that data type is capable of being calculatd from the current scan data
//
// OLE equivalent:
// int GetScanDataAutomation(LONG nStart, LONG nEnd, LONG nWhata, LONG nWhatb, VARIANT *a, VARIANT *b );
// VARIANT a and b must encapsulate safearrays of doubles of sufficient size for the output and be zero indexed
// even if one or other ( a or b ) are not in use - i.e. DISPLAY_NOTHING is used for it, then an suitable
// variant must be provided still
__declspec(dllexport) int _stdcall MyVNAGetScanData(int nStart, int nEnd, int nWhata, int nWhatb, double *pDataA, double *pDataB );
// flags for nWhat in GetScanData
// the first is a dummy - used for nWhata or nWHatb to cause no data to be retrieved for that 
// case ( a or b) hence to retrieve just one parameter set in a call, set nWhatA or nWhatB to the
// desired value and set the other (nWhatb or nWhata) to be set to -2.
// Otherwise two separate parameter values may be retrieved at same time, for example setting
// nWhata to -1 and nWhatb to 0 would cause scan frequency data and RS data to be retrieved.
// Setting the values to 21 and 22 would cause S11 real and imaginary to be retrieved.
#define DISPLAY_NOTHING -2
#define DISPLAY_FREQ_SCALE -1
#define DISPLAY_REFL_RS 0
#define DISPLAY_REFL_XS 1
#define DISPLAY_REFL_RP 2
#define DISPLAY_REFL_XP 3
#define DISPLAY_REFL_MODZS 4
#define DISPLAY_REFL_ANGZS 5
#define DISPLAY_REFL_VSWR 6
#define DISPLAY_REFL_RL 7
#define DISPLAY_REFL_RHO 8
#define DISPLAY_REFL_ANGRHO 9
#define DISPLAY_REFL_Q 10
#define DISPLAY_REFL_CS 11
#define DISPLAY_REFL_LS 12
#define DISPLAY_REFL_CP 13
#define DISPLAY_REFL_LP 14
#define DISPLAY_REFL_GP 15
#define DISPLAY_REFL_BP 16
#define DISPLAY_REFL_MODY 17
#define DISPLAY_REFL_ANGY 18
#define DISPLAY_REFL_RHO_REAL 19
#define DISPLAY_REFL_RHO_IMAG 20
#define DISPLAY_S11_REAL 21
#define DISPLAY_S11_IMAG 22
#define DISPLAY_S11_ABS 23
#define DISPLAY_S11_ANG 24
#define DISPLAY_S11_RL 25
#define DISPLAY_S21_GAIN 26
#define DISPLAY_S21_GAIN_DB 27
#define DISPLAY_S21_GAIN_ANG 28
#define DISPLAY_S22_REAL 29
#define DISPLAY_S22_IMAG 30
#define DISPLAY_S22_ABS 31
#define DISPLAY_S22_ANG 32
#define DISPLAY_S22_RL 33
#define DISPLAY_S12_GAIN 34
#define DISPLAY_S12_GAIN_DB 35
#define DISPLAY_S12_GAIN_ANG 36
#define DISPLAY_S21_REAL 37
#define DISPLAY_S21_IMAG 38
#define DISPLAY_S12_REAL 39
#define DISPLAY_S12_IMAG 40
#define DISPLAY_TRANS_GAIN 41
#define DISPLAY_TRANS_GAINDB 42
#define DISPLAY_TRANS_GAIN_PHASE 43
#define DISPLAY_TRANS_REAL 44
#define DISPLAY_TRANS_IMAG 45
#define DISPLAY_TRANS_GROUPDELAY 46
#define DISPLAY_DISP1 47
#define DISPLAY_DISP1RE DISPLAY_DISP1
#define DISPLAY_DISP1IM DISPLAY_DISP1+1
#define DISPLAY_DISP2 49
#define DISPLAY_DISP2RE DISPLAY_DISP2
#define DISPLAY_DISP2IM DISPLAY_DISP2+1
#define DISPLAY_DISP3 51
#define DISPLAY_DISP3RE DISPLAY_DISP3
#define DISPLAY_DISP3IM DISPLAY_DISP3+1
#define DISPLAY_DISP4 53
#define DISPLAY_DISP4RE DISPLAY_DISP4
#define DISPLAY_DISP4IM DISPLAY_DISP4+1
// spectrum analyser trace - unused
#define DISPLAY_SPECTRUM_MAG 55
#define DISPLAY_SPECTRUM_DBM 56
// time domain - offsets in nDisplayTDRType
#define DISPLAY_TDR_FIRST 57
#define DISPLAY_TDR_11_V 57
#define DISPLAY_TDR_11_DB 58
#define DISPLAY_TDR_22_V 59
#define DISPLAY_TDR_22_DB 60
#define DISPLAY_TDR_21_V 61
#define DISPLAY_TDR_21_DB 62
#define DISPLAY_TDR_12_V 63
#define DISPLAY_TDR_12_DB 64
#define DISPLAY_TDR_11_ZS 65
#define DISPLAY_TDR_22_ZS 66

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASetDisplayMode() and MyVNAGetDisplayMode()
// get and set the basic display mode
__declspec(dllexport) int _stdcall MyVNASetDisplayMode(int nMode);
__declspec(dllexport) int _stdcall MyVNAGetDisplayMode(int *pnMode);
// nMode takes one of these values
#define DISP_MODE_RECT 0
#define DISP_MODE_REPORT 1
#define DISP_MODE_EQ_CCT 2
#define DISP_MODE_POLAR 3

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNASetInstrumentMode() and MyVNAGetInstrumentMode()
// get and set the basic instrument mode
// OLE equivalent:
// Set / Get property nInstrumentMode
__declspec(dllexport) int _stdcall MyVNASetInstrumentMode(int nMode);
__declspec(dllexport) int _stdcall MyVNAGetInstrumentMode(int *pnMode);
// data structure for Get/Set instrument mode
// parameter is a 32 bit unsigned integer as follows
// bits 0..3 define the mode. Not all values currently in use
// the following values should be placed into those bits as follows
#define INSTRUMENT_MODE_REFLECTION 0
#define INSTRUMENT_MODE_TRANSMISSION 1
#define INSTRUMENT_MODE_DUALMODE 2
#define INSTRUMENT_MODE_SPARAM 3
#define INSTRUMENT_MODE_SPECANAL 4
// bit 4 if set causes the program to always do a dual scan even if reflection or transmission mode set
#define ALWAYS_DO_DUAL_SCAN	
// bit 5 if set forces a reverse scan (as in S12/S22 instead of S21/S11
#define REVERSE_SCAN (1<<5)
// bit 6 if set causes RFIV mode to be selected
#define RFIV_SCAN (1<<6)
// bit 7 if set causes Reference mode to be selected
#define REFMODE_SCAN (1<<7)
// bit 8 if set causes log frequency scale mode to be selected
#define LOGF_SCAN (1<<8)

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNALoadConfiguration()
// MyVNASaveConfiguration()
// MyVNALoadCalibration()
// MyVNASaveCalibration()
// Given a filename, attempt to load or save the current program configuration
// including calibration data or just load/save the calibration data
// finally, option exists to save current trace data to a file ( s2p only supported here)
// OLE equivalents:
// int LoadConfigurationAutomation(LPCTSTR fileName);
// int SaveConfigurationAutomation(LPCTSTR fileName);
// int LoadCalibrationAutomation(LPCTSTR fileName);
// int SaveCalibrationAutomation(LPCTSTR fileName);
// int SaveTraceDataAutomation(LPCTSTR fileName);
__declspec(dllexport) int _stdcall MyVNALoadConfiguration( _TCHAR * fileName);
__declspec(dllexport) int _stdcall MyVNASaveConfiguration( _TCHAR * fileName);
__declspec(dllexport) int _stdcall MyVNALoadCalibration( _TCHAR * fileName);
__declspec(dllexport) int _stdcall MyVNASaveCalibration( _TCHAR * fileName);
__declspec(dllexport) int _stdcall MyVNASaveTraceData( _TCHAR * fileName);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// when in eq cct view mode (and only then) use these functions to establish a log file and log
// the results of a scan. The description is a string added to each log entry
// OLE equivalents:
// int SetEqCctLogFileAutomation(LPCTSTR fileName);
// int EqCctLogFileAutomation(LPCTSTR description);
__declspec(dllexport) int _stdcall MyVNASetEqCctLogFile(_TCHAR * fileName);
__declspec(dllexport) int _stdcall MyVNALogEqCctResults(_TCHAR * description);


// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// MyVNAAutoscale()
// execute the autoscale function (same as clicking the Autoscale button)
// OLE equivalent:
// int AutoscaleAutomation(void);
__declspec(dllexport) int _stdcall MyVNAAutoscale(void);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// copy whatever is currently being displayed to the clipboard
// OLE equivalent:
// int ClipboardCopyAutomation(void);
__declspec(dllexport) int _stdcall MyVNAClipboardCopy(void);

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// general purpose interface functions used to get or set various things.
// Two versions exist, one for integers and one for doubles, with a Get and a Set in each case
// The functions need a parameter to say what is to be set/got - see details below, a pointer
// to an array of sufficient size for the results and as a safeguard the number of entries
// in tht array
// OLE equivalents:
// int GetIntegerArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
// int SetIntegerArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
__declspec(dllexport) int _stdcall MyVNAGetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnResult);
__declspec(dllexport) int _stdcall MyVNASetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnData);
// options for nWhat parameter in MyVNAGetIntegerArray() and MyVNASetIntegerArray()
// all get / set an array of integers
// nIndex is required for some but not all options - as indicated below. Set to 0 when not used.
//
// case 0 - display options - array of 4 integers
// nIndex not used- set to 0
// data[0] = horizontal divisions in byte 0, vertical divisions in byte 1
// data[1] = byte 0 pen width, byte 1 marker size
// data[2] = flags as follows
//		bit 0 - graticule on
//		bit 1 - scan progress bar displayed
//		bit 2 - autoscale on display change
//		bit 3 - snap to 125 on display change
//		bit 4 - snap to 125
//		bit 5 - audio cues
//		bit 6 - force |disp| on log axes
//		bit 7 - auto refine on equivalent circuits
//		bit 8 - invert RL display
//		bit 9 - display info tips
//		bit 10 - label frequency gridlines
//		bit 11 - label vertical gridlines
//		bit 12 - show scan data
//		bit 13 - lock scan to display
//      bit 14 - 31 spare
// data[3] = more flags. Note these flags are not readable - they will always read as zero in this version
// this parameter may be omitted by setting the number of integers to 3 instead of 4.
//		bit 0 - log vertical scale if currently in rectangular display mode
//		bit 1 - log frequency scale if currently in rectangular display mode
//		bit 2 - if set, set the scan to match the current display freqeuncies (make sure scan is not locked to display - see bit 13 above)
//		bit 3 - if set, set the display to match the current scan frequencies (make sure scan is not locked to display - see bit 13 above)
//		bit 4 - if set lock the frequency axis
//		bit 5 - if set lock the left axis
//		bit 6 - if set lock the right axis
#define GETSET_DISPLAY_OPTIONS 0

// case 1 - print options - array of 4 integers
// nIndex not used- set to 0
// data[0] = unused
// data[1] = byte 0 pen width, byte 1 marker size
// data[2] = flags as follows
//		bit 0 - add print notes to clipboard copy
//		bit 1 - label markers in printout
//		bit 2 - 31 spare
// data[3] = spare
#define GETSET_PRINT_OPTIONS 1

// case 2 - get and set screen colours - array of 1 integers
// nIndex - which colour to access. 
//		0 = Border
//		1 = graticule
//		0x10 to 0x17 = trace colours 1 to 8
//		0x30 to 0x38 = marker colours 1 to 9
// data[0] - the colour. This is a DWORD passed as an int
// it is a COLORREF, which takes the form 0x00bbggrr
// and may be created with the RGB macro.
// to set a colour, populate data[0] and data[1] with the appropriate values
// to get a colour, populate data[0] with the desired target and the subroutine
// will fill in the colour value in data[1]
// if the value of data[0] is out of range, the set command has no effect
// if it is out of range on a get, the returned colour value will be -1 and the subroutine
// will return a non zero value
#define GETSET_COLOURS 2

// case 3 - get or set left axis display parameters
// case 4 - get or set right axis display parameters
// nIndex not used- set to 0
// requires an array of 4 integers
// The array contents correstpond to a 128 bit bitmap
// where the bits that are set determine which parameters are shown on the axis
// The bits correspond to the values shown for Get or Set Scan Data above
// with the exception of DISPLAY_FREQ_SCALE
// hence for example:
// DISPLAY_REFL_CS takes the value of 11 and DISPLAY_REFL_XS takes the value 1
// so the bitmap would be (1<<DISPLAY_REFL_XS) + (1<<DISPLAY_REFL_CS)
// in other words (1<<11) + (1<<1) or 0x00000000000000000000000000000802
// the value is split into four 32 bit unsigned integer values as follows
// data[0] - bits 31-0
// data[1] - bits 63-32
// data[2] = bits 95-64
// data[3] - bits 127-96
// Given the current definitions the final entry in the above list is DISPLAY_TDR_22_ZS
// which takes the value of 66, hence currently bits 67-127 will always be zero.
#define GETSETLEFTAXIS 3
#define GETSETRIGHTAXIS 4

// case 5 - get or set hardware delays & adc speed
// nIndex not used- set to 0
// data[0] - ADC speed
// data[1] - ADC step delay
// data[2] - sweep start delay
// data[3] - phase change delay
// the speed is an integer that depends on hardware; 1..10 for the N2PK other values ignored
// the time delays are integers in us
// the data may be truncated; for example setting the data length to 2 will changeADC speed and step delay only
#define GETSETHARDWARESPEEDS 5

// case 6 - set or get the hardware options
// nIndex not used- set to 0
// data[0] - CDS mode as defined below
// data[1] - flags as defined below
// data[2] - system reference
// CDS mode. This is an integer structured as follows
// bit 0 - if 0, basic mode and rest of this integer has no effect
//       - if 1, harmonic suppression mode as defined below
// bits 8-15 - harmonic where 0x01 means fundamental, 0x02 is second harmonic 
// bits 16-24 - number of samples 0x01 means 1, 0x04 means 4 etc
// limitations:
// samples must be 0x04, 0x08, 0x10 or 0x32. Other values will cause setting to be ignored
// harmonics must be 1,2,3,4 or 5. Other values will be ignored
// if harmonics is 2 or 3, sample setting 4 is not available
// if harmonics is 4 or 5, sample settings 4 and 8 are not available
// example
// to select harmonic mode 3 with 16 samples data[0] should be set to 0x00100301
// it is acceptable to issue the command with just one integer and change the CDS mode without updating the rest
// data[1] - flags
// bit 0 - if set load DDS during ADC setting
// bit 1 - if set swap detectors on reverse scan
// bit 2 - if set power down DDS when idle
// it is permitted to set the CDS mode and flags without changing system reference by passing just 2 integers
// data[2] - system reference (milli ohms) - must be > 0
#define GETSETHARDWAREOPTIONS 6

// case 7 - set or get marker configuration.
// note there are other functions to get the marker value / frequency and marker arithmetic and other settings
// nIndex - set to the marker number ( 0..(NUM_MARKERS-1) )
// function will use data[0] to determine which marker and will fill in results in array
// data[0] - source information
// data[1] - mode information
// data[2] - target
// data[3] - link
// data[4] - display flag
// the meaning of the above is as follows
//
// Source is as follows
// results if set to invalid settings are undefined.
// for example do not set more than 1 of bits 1-3
// do not set bits 8-11 out of range
// do not set bits 16-23 to invalid value ( 0..66 are valid at moment)
// bit 0; 0=>left, 1=>right
// bit 1; 1=>scan data
// bit 2; 1=>store data
// bit 3; 1=>sim data
// bit 4-7 spare
// bit 11-8; store, sim or cal index (0..15)
// bit 15-12 spare
// bit 23-16 parameter type ( not all in use - integer 0..255 )
// bits 31-24 spare
//
// mode is as follows:
// 0 = tracking
// 1 = manual
// 2 = linked
// 3 = linked; f-
// 4 = linked; f+
//
// target is as follows
// 0 = maximum
// 1 = minimum
// 2 = cross up 1st
// 3 = cross down 1st
// 4 = cross up 2nd
// 5 = cross down 2nd
// 6 = cross up 3rd
// 7 = cross down 3rd
//
// link is the other marker number ( 0 .. (NUM_MARKERS-1) )
// display flag is 0 to disable/hide and <>0 to enable & display
#define GETSETMARKERSETUP 7

// case 8 - get (not set) various program constants
// nIndex = 0 - return the following
// data[0] - number of different left/right axis parameters 
// data[1] - number of markers
// data[2] - number of calculation markers
// data[3] - number of stores for traces
// data[4] - number of annotations
// data[5] - number of separate trace colour
// data[6] - number of transverters permitted
// data[7] - limit on length of a transverter name
// data[8] - number of simulations supported
// data[9] - number of simulation structures per simulation supported
// data[10] - number of components per simulation
#define GETPROGRAMCONSTANTS 8

// case 9 - get or set equivalent circuit configuration
// nIndex not used - set to 0
// Note: this function is only supported when equivalent circuit display mode is selected
// and unless default options are desired MUST be sent each time the display mode is set to eq cct mode
// data[0] - equivalent circuit device type. Set to 0 for crystal motional parameters
// data[1] - model - set as follows
//			0 = 45 degree phase
//			1 = 3dB
//			2 = 6 term
// data[2] - set data source
//			0 = current scan data
//			1..number of stores = stored trace data
#define GETSETEQCCTCONFIG 9

// case 10 - get or set simulation configuration
// nIndex not used - set to 0
// sets overall configuration by determining the type of each block
// data[0] is simulation block 1
// data[1] is block 1 etc
// each one takes values thus
// 0 = unused
// 1 = simulation
// 2 = scan data
// 3 = store 1 
// 4 = store 2 
// etc
#define GETSETSIMULATIONCONFIG 10

// case 11 - trace calculation options
// nIndex not used - set to 0
// sets options related to trace calculation controls
// data[0] are various flags
//		bit 0 - if set show network simulation dialog
//		bit 1 - if set, set "show simulation data"
//		bit 2 - if set show marker measurements dialog
//		bit 3 - if set, set "TDR functions"
// when read, bits 0 and 2 will always read '0'
#define GETSETTRACECALCULATIONOPTIONS 11

// case 12 - switch and attenuator configuration
// nIndex not used - set to 0
// sets options related to configuration of switch and attenuator options
// data[0] are various flags
//		bit 0 - if set set "invert sense" flag for switch 1
//		bit 1 - if set set "invert sense" flag for switch 2
//		bit 8 - if set set "invert sense" flag for attenuator
// data[1] - values 0 to 7 configure the "forward scan" attenuator setting
// data[2] - values 0 to 7 configure the "reverse scan" attenuator setting
#define GETSETSWITCHATTENUATORCONFIG 12

// case 13 - switch and attenuator settings
// nIndex not used - set to 0
// sets options related to configuration of switch and attenuator settings
// data[0] are various flags
//		bit 0 - if set set "invert sense" flag for switch 1
//		bit 1 - if set set "invert sense" flag for switch 2
//		bit 8 - if set set enable switch 1 during reverse scan
//		bit 9 - if set set enable switch 2 during reverse scan
//		bit 16 - if set set enable switch 1 during scan
//		bit 17 - if set set enable switch 2 during scan
//		bit 24 - if set set enable automatic attenuator setting during scan
// data[1] - values 0 to 7 sets the attenuator
#define GETSETSWITCHATTENUATORSETTINGS 13


// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// general purpose interface functions used to get or set various things.
// Two versions exist, one for integers and one for doubles, with a Get and a Set in each case
// The functions need a parameter to say what is to be set/got - see details below, a pointer
// to an array of sufficient size for the results and as a safeguard the number of entries
// in tht array
// OLE equivalents:
// int GetDoubleArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
// int SetDoubleArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
__declspec(dllexport) int _stdcall MyVNAGetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnResult);
__declspec(dllexport) int _stdcall MyVNASetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnData);
//
// options for MyVNAGetDoubleArray and MyVNASetDoubleArray
// case 0 - get (not set) frequency data
// nIndex is not used - set to 0
//  returns a set of doubles for scan related frequency data as follows
//  data[0] = actual scan frequency start
//  data[1] = actual scan frequency end
//  data[2] = scan start (for options that use a start frequency)
//  data[3] = scan end (for options that use a start frequency)
//  data[4] = scan centre (for options that use a centre frequency)
//  data[5] = scan span (for options that use a span frequency)
//  data[6] = scan freq/step (for options that use a freq/step frequency)
//  data[7] = scan freq/division (for options that use a freq/division frequency)
//  data[8] = current scan mode (an integer as defined above returned as a double)
#define GET_SCAN_FREQ_DATA 0

// case 1 - get or set marker values
// passes an array of values as follows
// nIndex = marker number 0..(NUM_MARKERS-1)
// data[0] = marker value
// data[1] = time or frequency value
#define GETSET_MARKER_VALUES 1

// case 2 - get (not set) equivalent circuit results
// nIndex is not used
// data depends on model chosen
#define GET_EQCCT_VALUES 2

// case 3 - get/set display frequency and time settings
// nIndex is not used
// data[0] - display start frequency
// data[1] - display end frequency
// data[2] - display start time
// data[3] - display end time
// Either 2 or 4 values may be provided, either frequencies alone or frequencies and tim
#define GETSET_DISPLAY_FT_AXIS_VALUES 3

// case 4 - get/set vertical axis settings
// nIndex is 0 for left, 1 for right axis
// data[0] - axis top
// data[1] - axis bottom
#define GETSET_DISPLAY_VERTICAL_AXIS_VALUES 4

// case 5 - get/set N2PK hardware configuration
// nIndex unused
// data[0] - transmission ADV 1 or 2
// data[1] - reflection ADV 1 or 2
// may either pass 2 values or 5; if 2 only the above are affected
// data[2] - clock frequency (Hz)
// data[3] - VNA minimum frequency
// data[4] - VNA maximum frequency
#define GETSET_N2PK_HARDWARE_VALUES 5

// case 6 - get/set network simulation components
// nIndex - simulation number ( 0..{num_simulations-1} )
// data[0] - component 0 type (0=none; 1=R, 2=L, 3=C)
// data[1] - component 0 value
// data[2] - component 1 type
// etc
// so for 8 components it needs an array of 16 doubles
#define GETSET_SIMULATION_COMPONENT_VALUES 6

// /////////////////////////////////////////////////////////////////////////////////////////////////
//
// Get or set a string parameter
// The functions need a parameter to say what is to be set/got - see details below, 
// a string for the result / parameter
// and an index value
// OLE equivalents:
// int SetStringAutomation(LONG nWhat, LONG nIndex, BSTR newstring);
// BSTR GetStringAutomation(LONG nWhat, LONG nIndex);
__declspec(dllexport) int _stdcall MyVNASetString(int nWhat, int nIndex, _TCHAR * sWhat );
__declspec(dllexport) CString _stdcall MyVNAGetString(int nWhat, int nIndex );

//
// options for MyVNAGetString and MyVNASetString
// case 0 - equation
// nIndex is which one to set/get 0..3
#define GETSET_EQUATION 0
