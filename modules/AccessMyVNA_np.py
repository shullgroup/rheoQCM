
import os
import signal
import numpy as np
from ctypes import *
# from ctypes import windll, WinDLL, wintypes, WINFUNCTYPE, POINTER, c_int, c_double, byref, Array, cast, get_last_error, WinError
from ctypes.wintypes import HWND, LONG, BOOL, LPARAM, LPDWORD, DWORD, LPWSTR
# from comtypes.safearray import safearray_as_ndarray

import numpy.ctypeslib as clib
import sys, struct, time

import win32ui
import win32process

import ctypes

try: # run from main
    from modules.retrying import retry
except: # run by itself
    from retrying import retry

print(sys.version)
print(struct.calcsize('P') * 8)


# constant
WM_USER = 0x0400                 # WM_USER   0x0400
WM_COMMAND = 0x0111                # WM_COMMAND 0x0111
MESSAGE_SCAN_ENDED = WM_USER + 0x1234  # MESSAGE_SCAN_ENDED (WM_USER+0x1234)
# retry decorator
wait_fixed = 10
stop_max_attempt_number = 100
stop_max_delay = 10000

# window name
win_names = [
    u'myVNA - Reflection mode "myVNA" [Embedded] ',  # there is a space at the end
    u'myVNA - Reflection mode "myVNA"']
# win_name = 'AccessMyVNA'

# dll path
# dll_path = r'./VNA/AccessMyVNA_v0.7/release/AccessMyVNAdll.dll'
# dll_path = r'./VNA/AccessMyVNAv0.7_J/release/AccessMyVNAdll.dll'
dll_path = r'./dll/AccessMyVNAdll.dll'

vna = WinDLL(dll_path, use_last_error=False) # this only works with AccessMyVNA
# vna = OleDLL(r'AccessMyVNAdll.dll', use_last_error=True) # this only works with AccessMyVNA
# print(vars(vna))
print(vna._handle)
# print(dir(vna))


#region functions
def _check_zero(result, func, args):    
    if not result:
        err = get_last_error()
        if err:
            raise WinError(err)
    return args

def get_hWnd(win_name=win_names[0]):
    try:
        hWnd = win32ui.FindWindow(None, win_name).GetSafeHwnd()
        # pid = win32process.GetWindowThreadProcessId(hWnd)[1]
    except:
        hWnd = None
    print('hWnd', hWnd, '"' + win_name + '"') 
    return hWnd

    # hWnd = win32ui.GetMainFrame.GetSafeHwnd

def get_pid(hWnd):
    if not hWnd:
        print('PID did not find!')
        pid = None        
    else:
        pid = win32process.GetWindowThreadProcessId(hWnd)[1]

    return pid

def close_win():
    # close myVNA initiated by AccessMyVNA
    for win_name in win_names:
        hWnd = get_hWnd(win_name)
        print('hWnd', hWnd)
        pid = get_pid(hWnd)
        print('pid', pid)
        if pid:
            os.kill(pid, signal.SIGTERM)

# make function prototypes 
def wfunc(name, dll, result, *args):
    '''
    build and apply a ctypes prototype complete with   rameter flags
    
    name: string of function name in the dll, 
    dll: dll object, 
    out: type of the output, 
    *args: list of tuples(argtype, (in/out, argname[, default]) 
        argname: name of the argument, 
        argtype: type, 
        in/out: 1=input and 2=output,
        default: optional default value.
    '''
    atypes = []
    aflags = []
    for arg in args:
        atypes.append(arg[0])
        aflags.append(arg[1])
    return WINFUNCTYPE(result, *atypes)((name, dll), tuple(aflags))
    # return WINFUNCTYPE(result, *atypes)((name, dll)) #, tuple(aflags))

def get_wait_time(nPoints=200, averages=0, step_delay=0, start_delay=0, buffer=70):
    dds_load = 90 # us
    phase_points = [0, 90, 180, 270] # CDS phase points, size of list is what is important
    delay = 4000 # delay that depends on the PC CPU speed, sweep points, etc. documentation provides no further details

    # ADC conversion timing
    conversion_delay = 300 # microseconds
    fqud_pulse = 100 # microseconds
    clock_delay = 20 # microseconds
    
    usb_frame_time = 125 # USB version 0.22
    
    # align ADC conversions with USB frames grid
    # TODO adjust alignment incorporating conversion delays
    minimum_conversion_time = 500 # umicroseconds
    conversion_time = conversion_delay + fqud_pulse + clock_delay + step_delay + buffer # microseconds
    if conversion_time % usb_frame_time != 0:
        minimum_conversion_time = (conversion_time % usb_frame_time)*conversion_time + usb_frame_time # microseconds
    
    conversion_rate = 1/(conversion_time*1e-6) # get the data rate in ADC conversions/sec
    num_conversions = nPoints * len(phase_points) # get the number of conversions
    total_time = dds_load * 1e-6 + start_delay * 1e-6 + num_conversions / conversion_rate + delay * 1e-6 # get total time in seconds
    print(total_time)

    return total_time
    
#endregion

#region assign functions
#########################################
# MyVNAInit = vna[13] # MyVNAInit
# // call this function befoe trying to do anything else. It attempts to executeF myVNA
# // and establish it as an automation server
# // OLE equivalent:
# // Connect to the server using the CLSID
# __declspec(dllexport) int _stdcall MyVNAInit(void)

_MyVNAInit = wfunc(
    '?MyVNAInit@@YGHXZ', vna, c_int,
)

##########################################
# MyVNAClose =  vna[3] # MyVNAClose
# MyVNAClose = getattr(vna, '?MyVNAClose@@YGHXZ') # MyVNAClose

# // you MUST call this before the calling windows application closes in order to correctly shut down OLE
# // Make sure you call the Close function before the windows GUI application is closed
# // OLE equivalent:
# // simply release the interface	
# __declspec(dllexport) int _stdcall MyVNAClose(void);

_MyVNAClose = wfunc(
    '?MyVNAClose@@YGHXZ', vna, c_int,
)

##########################################
# MyVNAShowWindow = vna[29] # MyVNAShowWindow
# // show or hide the current myVNA window.
# // nValue set to 0 causes SW_SHOWDEFAULT
# // nValue set to 1 causes SW_SHOWMINIMIZED
# // all other values ignored
# // OLE equivalent:
# // void ShowWindowAutomation(LONG nValue);
# __declspec(dllexport) int _stdcall MyVNAShowWindow(int nValue)
_MyVNAShowWindow = wfunc(
    '?MyVNAShowWindow@@YGHH@Z', vna, c_int,
    (c_int, (1, 'nValue', 1))
)
##########################################
# MyVNAGetScanSteps = vna[11] # MyVNAGetScanSteps
# // read the current scan steps value
# // OLE equivalent:
# // Get property nScanSteps
# __declspec(dllexport) int _stdcall MyVNAGetScanSteps(int *pnSteps);
# int nTemp;
# int nRet = MyVNAGetScanSteps(&nTemp);
_MyVNAGetScanSteps = wfunc(
    '?MyVNAGetScanSteps@@YGHPAH@Z', vna, c_int,
    (POINTER(c_int), (1, 'pnSteps', 200))
)


##########################################
# MyVNASetScanSteps = vna[27] # MyVNASetScanSteps
# // sets the number of steps ( min 2 max 50,000 but note this can take a huge amount of memory)
# // OLE equivalent:
# // Set property nScanSteps
# __declspec(dllexport) int _stdcall MyVNASetScanSteps(int nSteps);
# __declspec(dllexport) int _stdcall MyVNASetScanSteps(int *pnSteps)
# int nTemp;
# int nRet = MyVNASetScanSteps(nTemp);
_MyVNASetScanSteps = wfunc(
    '?MyVNASetScanSteps@@YGHH@Z', vna, c_int,
    (c_int, (1, 'nSteps', 200))
)

##########################################
# MyVNAGetScanAverage = vna[9] # MyVNAGetScanAverage
# __declspec(dllexport) int _stdcall MyVNAGetScanAverage(int *pnAverage)
# int nTemp;
# int nRet = MyVNAGetScanAverage(&nTemp);
_MyVNAGetScanAverage = wfunc(
    '?MyVNAGetScanAverage@@YGHPAH@Z', vna, c_int,
    (POINTER(c_int), (1, 'pnAverage', 1))
)

##########################################
# MyVNASetScanAverage = vna[26] # MyVNASetScanAverage
# // set the current scan trace average count
# // OLE equivalent:
# // Set property nAverage
# __declspec(dllexport) int _stdcall MyVNASetScanAverage(int nAverage)
# int nTemp;
# int nRet = MyVNASetScanAverage(&nTemp);
_MyVNASetScanAverage = wfunc(
    '?MyVNASetScanAverage@@YGHH@Z', vna, c_int,
    (c_int, (1, 'nAverage', 1))
)

##########################################
# MyVNAGetDoubleArray = vna[6] # MyVNAGetDoubleArray
# __declspec(dllexport) int _stdcall MyVNAGetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnResult)
# double dFreqData[10];
# 	if( MyVNAGetDoubleArray( GET_SCAN_FREQ_DATA, 0, 9, &dFreqData[0] ) == 0 )
#   {
#   	dFreqStart = dFreqData[2] / 1e6;
#   	dFreqStop = dFreqData[3] / 1e6;
#   }
#
_MyVNAGetDoubleArray = wfunc(
    '?MyVNAGetDoubleArray@@YGHHHHPAN@Z', vna, c_int,
    (c_int, (1, 'nwhat')), 
    (c_int, (1, 'nIndex')), 
    (c_int, (1, 'nArraySize')),
    # (c_void_p, (1, 'npResult')) 
    (POINTER(c_double), (1, 'npResult')) 
)

##########################################
# MyVNASetDoubleArray = vna[21] # MyVNASetDoubleArray
#__declspec(dllexport) int _stdcall MyVNASetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnData)
# double dFreqData[10];
# 	if( MyVNASetDoubleArray( GET_SCAN_FREQ_DATA, 0, 9, &dFreqData[0] ) == 0 )
#   {
#   	dFreqStart = dFreqData[2] / 1e6;
#   	dFreqStop = dFreqData[3] / 1e6;
#   }

# // general purpose interface functions used to get or set various things.
# // Two versions exist, one for integers and one for doubles, with a Get and a Set in each case
# // The functions need a parameter to say what is to be set/got - see details below, a pointer
# // to an array of sufficient size for the results and as a safeguard the number of entries
# // in tht array
# // OLE equivalents:
# // int GetDoubleArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
# // int SetDoubleArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
# __declspec(dllexport) int _stdcall MyVNAGetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnResult);
# __declspec(dllexport) int _stdcall MyVNASetDoubleArray(int nWhat, int nIndex, int nArraySize, double *pnData);
# //
# // options for MyVNAGetDoubleArray and MyVNASetDoubleArray
# // case 0 - get (not set) frequency data
# // nIndex is not used - set to 0
# //  returns a set of doubles for scan related frequency data as follows
# //  data[0] = actual scan frequency start
# //  data[1] = actual scan frequency end
# //  data[2] = scan start (for options that use a start frequency)
# //  data[3] = scan end (for options that use a start frequency)
# //  data[4] = scan centre (for options that use a centre frequency)
# //  data[5] = scan span (for options that use a span frequency)
# //  data[6] = scan freq/step (for options that use a freq/step frequency)
# //  data[7] = scan freq/division (for options that use a freq/division frequency)
# //  data[8] = current scan mode (an integer as defined above returned as a double)
# #define GET_SCAN_FREQ_DATA 0

# // case 1 - get or set marker values
# // passes an array of values as follows
# // nIndex = marker number 0..(NUM_MARKERS-1)
# // data[0] = marker value
# // data[1] = time or frequency value
# #define GETSET_MARKER_VALUES 1

# // case 2 - get (not set) equivalent circuit results
# // nIndex is not used
# // data depends on model chosen
# #define GET_EQCCT_VALUES 2

# // case 3 - get/set display frequency and time settings
# // nIndex is not used
# // data[0] - display start frequency
# // data[1] - display end frequency
# // data[2] - display start time
# // data[3] - display end time
# // Either 2 or 4 values may be provided, either frequencies alone or frequencies and tim
# #define GETSET_DISPLAY_FT_AXIS_VALUES 3

# // case 4 - get/set vertical axis settings
# // nIndex is 0 for left, 1 for right axis
# // data[0] - axis top
# // data[1] - axis bottom
# #define GETSET_DISPLAY_VERTICAL_AXIS_VALUES 4

# // case 5 - get/set N2PK hardware configuration
# // nIndex unused
# // data[0] - transmission ADV 1 or 2
# // data[1] - reflection ADV 1 or 2
# // may either pass 2 values or 5; if 2 only the above are affected
# // data[2] - clock frequency (Hz)
# // data[3] - VNA minimum frequency
# // data[4] - VNA maximum frequency
# #define GETSET_N2PK_HARDWARE_VALUES 5

# // case 6 - get/set network simulation components
# // nIndex - simulation number ( 0..{num_simulations-1} )
# // data[0] - component 0 type (0=none; 1=R, 2=L, 3=C)
# // data[1] - component 0 value
# // data[2] - component 1 type
# // etc
# // so for 8 components it needs an array of 16 doubles
# #define GETSET_SIMULATION_COMPONENT_VALUES 6
_MyVNASetDoubleArray = wfunc(
    '?MyVNASetDoubleArray@@YGHHHHPAN@Z', vna, c_int,
    (c_int, (1, 'nwhat')), 
    (c_int, (1, 'nIndex')), 
    (c_int, (1, 'nArraySize')),
    # (c_void_p, (1, 'pnData')) 
    (POINTER(c_double), (1, 'pnData')) 
)

##########################################
# MyVNAGetIntegerArray = vna[8]
# # // get miscellaneous integer array based data
# # __declspec(dllexport) int _stdcall MyVNAGetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnResult)
# pass
_MyVNAGetIntegerArray = wfunc(
    '?MyVNAGetIntegerArray@@YGHHHHPAH@Z', vna, c_int,
    (c_int, (1, 'nwhat')), 
    (c_int, (1, 'nIndex')), 
    (c_int, (1, 'nArraySize')),
    # (c_void_p, (1, 'npResult')) 
    (POINTER(c_int), (1, 'npResult')) 
)

##########################################
# MyVNASetIntegerArray = vna[25]
# // set miscellaneous integer array based data
# __declspec(dllexport) int _stdcall MyVNASetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnData)
# // general purpose interface functions used to get or set various things.
# // Two versions exist, one for integers and one for doubles, with a Get and a Set in each case
# // The functions need a parameter to say what is to be set/got - see details below, a pointer
# // to an array of sufficient size for the results and as a safeguard the number of entries
# // in tht array
# // OLE equivalents:
# // int GetIntegerArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
# // int SetIntegerArrayAutomation(LONG nWhat, LONG nIndex, LONG nSize, VARIANT *a);
# __declspec(dllexport) int _stdcall MyVNAGetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnResult);
# __declspec(dllexport) int _stdcall MyVNASetIntegerArray(int nWhat, int nIndex, int nArraySize, int *pnData);
# // options for nWhat parameter in MyVNAGetIntegerArray() and MyVNASetIntegerArray()
# // all get / set an array of integers
# // nIndex is required for some but not all options - as indicated below. Set to 0 when not used.
# //
# // case 0 - display options - array of 4 integers
# // nIndex not used- set to 0
# // data[0] = horizontal divisions in byte 0, vertical divisions in byte 1
# // data[1] = byte 0 pen width, byte 1 marker size
# // data[2] = flags as follows
# //		bit 0 - graticule on
# //		bit 1 - scan progress bar displayed
# //		bit 2 - autoscale on display change
# //		bit 3 - snap to 125 on display change
# //		bit 4 - snap to 125
# //		bit 5 - audio cues
# //		bit 6 - force |disp| on log axes
# //		bit 7 - auto refine on equivalent circuits
# //		bit 8 - invert RL display
# //		bit 9 - display info tips
# //		bit 10 - label frequency gridlines
# //		bit 11 - label vertical gridlines
# //		bit 12 - show scan data
# //		bit 13 - lock scan to display
# //      bit 14 - 31 spare
# // data[3] = more flags. Note these flags are not readable - they will always read as zero in this version
# // this parameter may be omitted by setting the number of integers to 3 instead of 4.
# //		bit 0 - log vertical scale if currently in rectangular display mode
# //		bit 1 - log frequency scale if currently in rectangular display mode
# //		bit 2 - if set, set the scan to match the current display freqeuncies (make sure scan is not locked to display - see bit 13 above)
# //		bit 3 - if set, set the display to match the current scan frequencies (make sure scan is not locked to display - see bit 13 above)
# //		bit 4 - if set lock the frequency axis
# //		bit 5 - if set lock the left axis
# //		bit 6 - if set lock the right axis
# #define GETSET_DISPLAY_OPTIONS 0

# // case 1 - print options - array of 4 integers
# // nIndex not used- set to 0
# // data[0] = unused
# // data[1] = byte 0 pen width, byte 1 marker size
# // data[2] = flags as follows
# //		bit 0 - add print notes to clipboard copy
# //		bit 1 - label markers in printout
# //		bit 2 - 31 spare
# // data[3] = spare
# #define GETSET_PRINT_OPTIONS 1

# // case 2 - get and set screen colours - array of 1 integers
# // nIndex - which colour to access. 
# //		0 = Border
# //		1 = graticule
# //		0x10 to 0x17 = trace colours 1 to 8
# //		0x30 to 0x38 = marker colours 1 to 9
# // data[0] - the colour. This is a DWORD passed as an int
# // it is a COLORREF, which takes the form 0x00bbggrr
# // and may be created with the RGB macro.
# // to set a colour, populate data[0] and data[1] with the appropriate values
# // to get a colour, populate data[0] with the desired target and the subroutine
# // will fill in the colour value in data[1]
# // if the value of data[0] is out of range, the set command has no effect
# // if it is out of range on a get, the returned colour value will be -1 and the subroutine
# // will return a non zero value
# #define GETSET_COLOURS 2

# // case 3 - get or set left axis display parameters
# // case 4 - get or set right axis display parameters
# // nIndex not used- set to 0
# // requires an array of 4 integers
# // The array contents correstpond to a 128 bit bitmap
# // where the bits that are set determine which parameters are shown on the axis
# // The bits correspond to the values shown for Get or Set Scan Data above
# // with the exception of DISPLAY_FREQ_SCALE
# // hence for example:
# // DISPLAY_REFL_CS takes the value of 11 and DISPLAY_REFL_XS takes the value 1
# // so the bitmap would be (1<<DISPLAY_REFL_XS) + (1<<DISPLAY_REFL_CS)
# // in other words (1<<11) + (1<<1) or 0x00000000000000000000000000000802
# // the value is split into four 32 bit unsigned integer values as follows
# // data[0] - bits 31-0
# // data[1] - bits 63-32
# // data[2] = bits 95-64
# // data[3] - bits 127-96
# // Given the current definitions the final entry in the above list is DISPLAY_TDR_22_ZS
# // which takes the value of 66, hence currently bits 67-127 will always be zero.
# #define GETSETLEFTAXIS 3
# #define GETSETRIGHTAXIS 4

# // case 5 - get or set hardware delays & adc speed
# // nIndex not used- set to 0
# // data[0] - ADC speed
# // data[1] - ADC step delay
# // data[2] - sweep start delay
# // data[3] - phase change delay
# // the speed is an integer that depends on hardware; 1..10 for the N2PK other values ignored
# // the time delays are integers in us
# // the data may be truncated; for example setting the data length to 2 will changeADC speed and step delay only
# #define GETSETHARDWARESPEEDS 5

# // case 6 - set or get the hardware options
# // nIndex not used- set to 0
# // data[0] - CDS mode as defined below
# // data[1] - flags as defined below
# // data[2] - system reference
# // CDS mode. This is an integer structured as follows
# // bit 0 - if 0, basic mode and rest of this integer has no effect
# //       - if 1, harmonic suppression mode as defined below
# // bits 8-15 - harmonic where 0x01 means fundamental, 0x02 is second harmonic 
# // bits 16-24 - number of samples 0x01 means 1, 0x04 means 4 etc
# // limitations:
# // samples must be 0x04, 0x08, 0x10 or 0x32. Other values will cause setting to be ignored
# // harmonics must be 1,2,3,4 or 5. Other values will be ignored
# // if harmonics is 2 or 3, sample setting 4 is not available
# // if harmonics is 4 or 5, sample settings 4 and 8 are not available
# // example
# // to select harmonic mode 3 with 16 samples data[0] should be set to 0x00100301
# // it is acceptable to issue the command with just one integer and change the CDS mode without updating the rest
# // data[1] - flags
# // bit 0 - if set load DDS during ADC setting
# // bit 1 - if set swap detectors on reverse scan
# // bit 2 - if set power down DDS when idle
# // it is permitted to set the CDS mode and flags without changing system reference by passing just 2 integers
# // data[2] - system reference (milli ohms) - must be > 0
# #define GETSETHARDWAREOPTIONS 6

# // case 7 - set or get marker configuration.
# // note there are other functions to get the marker value / frequency and marker arithmetic and other settings
# // nIndex - set to the marker number ( 0..(NUM_MARKERS-1) )
# // function will use data[0] to determine which marker and will fill in results in array
# // data[0] - source information
# // data[1] - mode information
# // data[2] - target
# // data[3] - link
# // data[4] - display flag
# // the meaning of the above is as follows
# //
# // Source is as follows
# // results if set to invalid settings are undefined.
# // for example do not set more than 1 of bits 1-3
# // do not set bits 8-11 out of range
# // do not set bits 16-23 to invalid value ( 0..66 are valid at moment)
# // bit 0; 0=>left, 1=>right
# // bit 1; 1=>scan data
# // bit 2; 1=>store data
# // bit 3; 1=>sim data
# // bit 4-7 spare
# // bit 11-8; store, sim or cal index (0..15)
# // bit 15-12 spare
# // bit 23-16 parameter type ( not all in use - integer 0..255 )
# // bits 31-24 spare
# //
# // mode is as follows:
# // 0 = tracking
# // 1 = manual
# // 2 = linked
# // 3 = linked; f-
# // 4 = linked; f+
# //
# // target is as follows
# // 0 = maximum
# // 1 = minimum
# // 2 = cross up 1st
# // 3 = cross down 1st
# // 4 = cross up 2nd
# // 5 = cross down 2nd
# // 6 = cross up 3rd
# // 7 = cross down 3rd
# //
# // link is the other marker number ( 0 .. (NUM_MARKERS-1) )
# // display flag is 0 to disable/hide and <>0 to enable & display
# #define GETSETMARKERSETUP 7

# // case 8 - get (not set) various program constants
# // nIndex = 0 - return the following
# // data[0] - number of different left/right axis parameters 
# // data[1] - number of markers
# // data[2] - number of calculation markers
# // data[3] - number of stores for traces
# // data[4] - number of annotations
# // data[5] - number of separate trace colour
# // data[6] - number of transverters permitted
# // data[7] - limit on length of a transverter name
# // data[8] - number of simulations supported
# // data[9] - number of simulation structures per simulation supported
# // data[10] - number of components per simulation
# #define GETPROGRAMCONSTANTS 8

# // case 9 - get or set equivalent circuit configuration
# // nIndex not used - set to 0
# // Note: this function is only supported when equivalent circuit display mode is selected
# // and unless default options are desired MUST be sent each time the display mode is set to eq cct mode
# // data[0] - equivalent circuit device type. Set to 0 for crystal motional parameters
# // data[1] - model - set as follows
# //			0 = 45 degree phase
# //			1 = 3dB
# //			2 = 6 term
# // data[2] - set data source
# //			0 = current scan data
# //			1..number of stores = stored trace data
# #define GETSETEQCCTCONFIG 9

# // case 10 - get or set simulation configuration
# // nIndex not used - set to 0
# // sets overall configuration by determining the type of each block
# // data[0] is simulation block 1
# // data[1] is block 1 etc
# // each one takes values thus
# // 0 = unused
# // 1 = simulation
# // 2 = scan data
# // 3 = store 1 
# // 4 = store 2 
# // etc
# #define GETSETSIMULATIONCONFIG 10

# // case 11 - trace calculation options
# // nIndex not used - set to 0
# // sets options related to trace calculation controls
# // data[0] are various flags
# //		bit 0 - if set show network simulation dialog
# //		bit 1 - if set, set "show simulation data"
# //		bit 2 - if set show marker measurements dialog
# //		bit 3 - if set, set "TDR functions"
# // when read, bits 0 and 2 will always read '0'
# #define GETSETTRACECALCULATIONOPTIONS 11

# // case 12 - switch and attenuator configuration
# // nIndex not used - set to 0
# // sets options related to configuration of switch and attenuator options
# // data[0] are various flags
# //		bit 0 - if set set "invert sense" flag for switch 1
# //		bit 1 - if set set "invert sense" flag for switch 2
# //		bit 8 - if set set "invert sense" flag for attenuator
# // data[1] - values 0 to 7 configure the "forward scan" attenuator setting
# // data[2] - values 0 to 7 configure the "reverse scan" attenuator setting
# #define GETSETSWITCHATTENUATORCONFIG 12

# // case 13 - switch and attenuator settings
# // nIndex not used - set to 0
# // sets options related to configuration of switch and attenuator settings
# // data[0] are various flags
# //		bit 0 - if set set "invert sense" flag for switch 1
# //		bit 1 - if set set "invert sense" flag for switch 2
# //		bit 8 - if set set enable switch 1 during reverse scan
# //		bit 9 - if set set enable switch 2 during reverse scan
# //		bit 16 - if set set enable switch 1 during scan
# //		bit 17 - if set set enable switch 2 during scan
# //		bit 24 - if set set enable automatic attenuator setting during scan
# // data[1] - values 0 to 7 sets the attenuator
# #define GETSETSWITCHATTENUATORSETTINGS 13
_MyVNASetIntegerArray = wfunc(
    '?MyVNASetIntegerArray@@YGHHHHPAH@Z', vna, c_int,
    (c_int, (1, 'nwhat')), 
    (c_int, (1, 'nIndex')), 
    (c_int, (1, 'nArraySize')),
    # (c_void_p, (1, 'pnData')) 
    (POINTER(c_int), (1, 'pnData')) 
)

##########################################
# MyVNAGetInstrumentMode = vna[7] # MyVNAGetInstrumentMode
# __declspec(dllexport) int _stdcall MyVNAGetInstrumentMode(int *pnMode)
# int nRet = MyVNAGetInstrumentMode( &nTemp);
_MyVNAGetInstrumentMode = wfunc(
    '?MyVNAGetInstrumentMode@@YGHPAH@Z', vna, c_int,
    (POINTER(c_int), (1, 'pnMode', 0)) # 0: reflection
) 

##########################################
# MyVNASetInstrumentMode = vna[24] # MyVNASetInstrumentMode
# // get and set the basic instrument mode
# // OLE equivalent:
# // Set / Get property nInstrumentMode
# __declspec(dllexport) int _stdcall MyVNASetInstrumentMode(int nMode);
# __declspec(dllexport) int _stdcall MyVNAGetInstrumentMode(int *pnMode);
# // data structure for Get/Set instrument mode
# // parameter is a 32 bit unsigned integer as follows
# // bits 0..3 define the mode. Not all values currently in use
# // the following values should be placed into those bits as follows
# #define INSTRUMENT_MODE_REFLECTION 0
# #define INSTRUMENT_MODE_TRANSMISSION 1
# #define INSTRUMENT_MODE_DUALMODE 2
# #define INSTRUMENT_MODE_SPARAM 3
# #define INSTRUMENT_MODE_SPECANAL 4
# // bit 4 if set causes the program to always do a dual scan even if reflection or transmission mode set
# #define ALWAYS_DO_DUAL_SCAN	
# // bit 5 if set forces a reverse scan (as in S12/S22 instead of S21/S11
# #define REVERSE_SCAN (1<<5)
# // bit 6 if set causes RFIV mode to be selected
# #define RFIV_SCAN (1<<6)
# // bit 7 if set causes Reference mode to be selected
# #define REFMODE_SCAN (1<<7)
# // bit 8 if set causes log frequency scale mode to be selected
# #define LOGF_SCAN (1<<8)
_MyVNASetInstrumentMode = wfunc(
    '?MyVNASetInstrumentMode@@YGHH@Z', vna, c_int,
    (c_int, (1, 'nMode', 0))
)     # 0: reflection

##########################################
# MyVNAGetDisplayMode = vna[5] # MyVNAGetDisplayMode
# __declspec(dllexport) int _stdcall MyVNAGetDisplayMode(int *pnMode)
# int nRet = MyVNAGetDisplayMode( nInstrumentMode);
_MyVNAGetDisplayMode= wfunc(
    '?MyVNAGetDisplayMode@@YGHPAH@Z', vna, c_int,
    (POINTER(c_int), (1, 'pnMode'), 0)
)

##########################################
# MyVNASetDisplayMode = vna[20] # MyVNASetDisplayMode
# // get and set the basic display mode
# __declspec(dllexport) int _stdcall MyVNASetDisplayMode(int nMode);
# __declspec(dllexport) int _stdcall MyVNAGetDisplayMode(int *pnMode);
# // nMode takes one of these values
# #define DISP_MODE_RECT 0
# #define DISP_MODE_REPORT 1
# #define DISP_MODE_EQ_CCT 2
# #define DISP_MODE_POLAR 3
_MyVNASetDisplayMode = wfunc(
    '?MyVNASetDisplayMode@@YGHH@Z', vna, c_int,
    (c_int, (1, 'nMode', 0))
)

##########################################
# MyVNASingleScan = vna[30] # MyVNASingleScan
# .h
# // MyVNASingleScan()
# // attempt to single scan the VNA. On completion myVNA will post a message ( Message) to the queue of the specified
# // window ( hWnd ) with the given COmmand and lParam values. See the example in AccessMyVNADlg.cpp
# // OLE equivalent:
# // int SingleScanAutomation(LONG Message, HWND hWnd, LONG lParam, LONG wParam );
# __declspec(dllexport) int _stdcall MyVNASingleScan(int Message, HWND hWnd, int nCommand, int lParam );
# .cpp
# // perform (or stop if in progress) a single scan
# __declspec(dllexport) int _stdcall MyVNASingleScan(int Message, HWND hWnd, int nCommand, int lParam )
# int nRet = MyVNASingleScan(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0 );

_MyVNASingleScan = wfunc(
    '?MyVNASingleScan@@YGHHPAUHWND__@@HH@Z', vna, c_int,
    (c_int, (1, 'Message'), WM_COMMAND),
    (HWND, (1, 'hWnd'), WM_COMMAND),
    (c_int, (1, 'nCommand'), MESSAGE_SCAN_ENDED),
    (c_int, (1, 'lParam'), 0),
)

##########################################
# MyVNAEqCctRefine = vna[4] # MyVNAEqCctRefine
# .h
# // MyVNAEqCctRefine()
# // attempt to refine an equivalent circuit. On completion myVNA will post a message ( Message) to the queue of the specified
# // window ( hWnd ) with the given Command and lParam values. See the example in AccessMyVNADlg.cpp
# // OLE equivalent:
# // int RefineAutomation(LONG nEnd, LONG hWnd, LONG Command, LONG lParam);
# __declspec(dllexport) int _stdcall MyVNAEqCctRefine(int Message, HWND hWnd, int nCommand, int lParam );
# .cpp
# // perform (or stop if in progress) a scan sequence for equivalent circuit refine
# __declspec(dllexport) int _stdcall MyVNAEqCctRefine(int Message, HWND hWnd, int nCommand, int lParam )

# int nRet = MyVNAEqCctRefine(WM_COMMAND, GetSafeHwnd(), MESSAGE_SCAN_ENDED, 0 );

_MyVNAEqCctRefine = wfunc(
    '?MyVNAEqCctRefine@@YGHHPAUHWND__@@HH@Z', vna, c_int,
    (c_int, (1, 'Message'), WM_COMMAND),
    (HWND, (1, 'hWnd'), WM_COMMAND),
    (c_int, (1, 'nCommand'), MESSAGE_SCAN_ENDED),
    (c_int, (1, 'lParam'), 0),
)

##########################################
# MyVNASetFequencies = vna[23]
# // frequency set / get functions
# // this first one sets start & end and also sets frequency control mode.
# // nFlags bits 0..3 set scan mode where f1 is first parameter / f2 is the second
# // 0 = centre / span
# // 1 = start / stop
# // 2 = full scan
# // 3 = from zero
# // 4 = centre / per division
# // 5 = centre / per step
# // 6 = start / per step
# // 7 = start / per division
# // 8..15 unused
# //
# // return codes:
# // 0 - OK
# // 1 - start too low
# // 2 - stop too high
# // 3 - width must be > 0
# // 4 - start is before end
# // 5 - stop too low
# // OLE equivalent:
# // int SetScanDetailsAutomation(double dF1, double dF2, LONG nFlags);
# __declspec(dllexport) int _stdcall MyVNASetFequencies(double f1, double f2, int nFlags)
# int nRet = MyVNASetFequencies( dFreqStart*1e6, dFreqStop*1e6, 1 );

_MyVNASetFequencies = wfunc(
    '?MyVNASetFequencies@@YGHNNH@Z', vna, c_int,
    (c_double, (1, 'f1')),
    (c_double, (1, 'f2')),
    (c_int, (1, 'nflags')),
)  

##########################################
# MyVNAGetScanData = vna[10] # MyVNAGetScanData
# .h
# // MyVNAGetScanData()
# // get current scan data results starting from scan point nStart up to and including nEnd
# // note that a scan of N steps gives N+1 data points from 0 to N
# // the data type requsted( nWhat) must be one of the ones as follows. 
# // It is the callers' responsibility
# // to make sure that that data type is capable of being calculatd from the current scan data
# //
# // OLE equivalent:
# // int GetScanDataAutomation(LONG nStart, LONG nEnd, LONG nWhata, LONG nWhatb, VARIANT *a, VARIANT *b );
# // VARIANT a and b must encapsulate safearrays of doubles of sufficient size for the output and be zero indexed
# // even if one or other ( a or b ) are not in use - i.e. DISPLAY_NOTHING is used for it, then an suitable
# // variant must be provided still
# __declspec(dllexport) int _stdcall MyVNAGetScanData(int nStart, int nEnd, int nWhata, int nWhatb, double *pDataA, double *pDataB );
# // flags for nWhat in GetScanData
# // the first is a dummy - used for nWhata or nWHatb to cause no data to be retrieved for that 
# // case ( a or b) hence to retrieve just one parameter set in a call, set nWhatA or nWhatB to the
# // desired value and set the other (nWhatb or nWhata) to be set to -2.
# // Otherwise two separate parameter values may be retrieved at same time, for example setting
# // nWhata to -1 and nWhatb to 0 would cause scan frequency data and RS data to be retrieved.
# // Setting the values to 21 and 22 would cause S11 real and imaginary to be retrieved.
# .cpp
# __declspec(dllexport) int _stdcall MyVNAGetScanData(int nStart, int nEnd, int nWhata, int nWhatb, double *pDataA, double *pDataB )
# MyVNAGetScanData(0, 199, -1, 15, &dFreq[0], &dData[0]);//scan data
# nWhat -2: nothing; -1: frequency; 15: Gp; 16:Bp (see the defination in .h file)

_MyVNAGetScanData = wfunc(
    '?MyVNAGetScanData@@YGHHHHHPAN0@Z', vna, c_int,
    (c_int, (1, 'nStart')),
    (c_int, (1, 'nEnd')),
    (c_int, (1, 'nWhata')),
    (c_int, (1, 'nWthab')),
    (POINTER(c_double), (1, 'pDataA')),
    (POINTER(c_double), (1, 'pDataB')),
)

##########################################
# MyVNAAutoscale = vna[1]
# // execute the autoscale function (same as clicking the Autoscale button)
# // OLE equivalent:
# // int AutoscaleAutomation(void);
# __declspec(dllexport) int _stdcall MyVNAAutoscale()
# MyVNAAutoscale = getattr(vna, '?MyVNAAutoscale@@YGHXZ')
# MyVNAAutoscale.argtypes = None
_MyVNAAutoscale = wfunc(
    '?MyVNAAutoscale@@YGHXZ', vna, c_int
)

#endregion

#region
'''
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
'''

#endregion

class AccessMyVNA():
    '''
    the module used to comunicate with MyVNA
    '''
    def __init__(self):
        # super(AccessMyVNA, self).__init__()
        # self.scandata_a = []
        # self.scandata_b = []
        # self.narray = []
       self.vnaset = self._vnaset_int

    # use __enter__ __exit__ for with or use try finally
    def __enter__(self):
        self.Init()
        self.ShowWindow(1) # mininmize window (may not work)
        print('in')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('out1')
        self.Close()
        print('out2')


    _vnaset_int = {
        'instrmode': np.array(0, dtype=int),
        'displaymode': np.array(0, dtype=int),
        'nsteps': np.array(0, dtype=int),
        'scanaverage': np.array(0, dtype=int),
        'chn': np.array(0, dtype=int), # avtive channel
        'speed': np.array(0, dtype=int), # speed of vna
        'f': [], # start & stop frequencies [start, stop]
    }

    def Init(self):
        # close vna window
        close_win()
        ret = _MyVNAInit()
        print('MyVNAInit\n', ret)
        return ret

    def Close(self):
        ret = _MyVNAClose()
        print('MyVNAClose\n', ret) #MyVNAAutoscale

    def ShowWindow(self, nValue=1):
        '''
        nValue 0: show; 1:minimize
        '''
        ret = _MyVNAShowWindow(nValue)
        print('MyVNAShowWindow\n', ret)
        return ret

    def SetScanSteps(self, nSteps=400):
        # if not isinstance(nSteps, np.ndarray):
        #     nSteps = np.array(nSteps, dtype=int)
        ret = _MyVNASetScanSteps(nSteps)
        # ret = MyVNASetScanSteps(nSteps)
        print('MyVNASetScanSteps\n', ret, nSteps)
        return ret, nSteps

    def GetScanSteps(self):
        nSteps = np.array(0, dtype=int)
        ret = _MyVNAGetScanSteps(nSteps.ctypes.data_as(POINTER(c_int)))
        # ret = MyVNAGetScanSteps(byref(nSteps))
        print('MyVNAGetScanSteps\n', ret, nSteps)

        nStp = nSteps
        del nSteps

        return ret, nStp

    def SetScanAverage(self, nAverage=1):
        # if not isinstance(nAverage, np.ndarray):
        #     nAverage = np.array(nAverage, dtype=int)
        ret = _MyVNASetScanAverage(nAverage)
        # ret = MyVNASetScanAverage(nAverage)
        print('MyVNASetScanAverage\n', ret, nAverage)
        return ret, nAverage

    def GetScanAverage(self):
        # nAverage = c_int()
        nAverage = np.array(0, dtype=int)
        ret = _MyVNAGetScanAverage(nAverage.ctypes.data_as(POINTER(c_int)))
        print('MyVNAGetScanAverage\n', ret, nAverage)

        nAve = nAverage
        del nAverage

        return ret, nAve

        
    #region
    '''
                                        nWhat   nArraySize
    GET_SCAN_FREQ_DATA                  0       9
    GETSET_MARKER_VALUES                1       2
    GET_EQCCT_VALUES                    2       depends on model chosen
    GETSET_DISPLAY_FT_AXIS_VALUES       3       2/4
    GETSET_DISPLAY_VERTICAL_AXIS_VALUES 4       2
    GETSET_N2PK_HARDWARE_VALUES         5       2/5
    GETSET_SIMULATION_COMPONENT_VALUES  6       components * 2

    GETSETSIMULATIONCONFIG                 5
    GETSETTRACECALCULATIONOPTIONS          1
    GETSETSWITCHATTENUATORCONFIG           3
    GETSETSWITCHATTENUATORSETTINGS         2
    '''
    #endregion

    @retry(wait_fixed=wait_fixed, stop_max_attempt_number=stop_max_attempt_number, stop_max_delay=stop_max_delay, logger=True)
    def GetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9):
        '''
        Get frequency nWhat = GET_SCAN_FREQ_DATA 0
        '''
        print('MyVNAGetDoubleArray')

        nResult = np.zeros(nArraySize, dtype=np.float, order='C')
        # self.narray = nResult

        nRes_ptr = nResult.ctypes.data_as(POINTER(c_double))
        ret = _MyVNAGetDoubleArray(nWhat, nIndex, nArraySize, nRes_ptr)
        # ret = _MyVNAGetDoubleArray(nWhat, nIndex, nArraySize, c_void_p(nResult))
        ######### end pointer ##########

        # print(nRes_ptr)
        # print(nRes_ptr.contents)
        ndRes = nResult[:] 
        del nResult

        print(ret, ndRes)
        return ret, ndRes


    @retry(wait_fixed=wait_fixed, stop_max_attempt_number=stop_max_attempt_number, stop_max_delay=stop_max_delay, logger=True)
    def SetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9, nData=[]):
        '''
        Set frequency nWhat = GET_SCAN_FREQ_DATA 0
        '''
        print('MyVNASetDoubleArray')
        if len(nData) == nArraySize: # check nData size
            if not isinstance(nData, np.ndarray): # check nData type
                nData = np.array(nData)
            nData.astype(np.float64, order='C')
        # print(nData)
        # print(nData.ctypes.data)

        # cast the array into a pointer of type c_double:
        ptr = nData.ctypes.data_as(POINTER(c_double))
        ret = _MyVNASetDoubleArray(nWhat, nIndex, nArraySize, ptr)
        # ret = _MyVNASetDoubleArray(nWhat, nIndex, nArraySize, c_void_p(nData.ctypes.data))

        print(ret, nData)
        
        nD = nData
        rt = ret
        del nData, ret

        return rt, nD

    @retry(wait_fixed=wait_fixed, stop_max_attempt_number=stop_max_attempt_number, stop_max_delay=stop_max_delay, logger=True)
    def GetIntegerArray(self, nWhat=5, nIndex=0, nArraySize=4):
        '''
        get hardware setup
        Get hardware delay & adc speed nWhat = 5
        '''
        print('MyVNAGetIntegerArray')

        nResult = np.zeros(nArraySize, dtype=int, order='C')
        # self.narray = nResult

        nRes_ptr = nResult.ctypes.data_as(POINTER(c_int))
        ret = _MyVNAGetIntegerArray(nWhat, nIndex, nArraySize, nRes_ptr)
        ######### end pointer ##########

        # print(nRes_ptr)
        # print(nRes_ptr.contents)
        ndRes = nResult[:] 
        del nResult

        print(ret, ndRes)
        return ret, ndRes


    @retry(wait_fixed=wait_fixed, stop_max_attempt_number=stop_max_attempt_number, stop_max_delay=stop_max_delay, logger=True)
    def SetIntegerArray(self, nWhat=5, nIndex=0, nArraySize=4, nData=[]):
        '''
        Set frequency nWhat = GET_SCAN_FREQ_DATA 0
        '''
        print('MyVNASetIntegerArray')
        if len(nData) == nArraySize: # check nData size
            if not isinstance(nData, np.ndarray): # check nData type
                nData = np.array(nData)
            nData.astype(int, order='C')
        # print(nData)
        # print(nData.ctypes.data)

        # cast the array into a pointer of type c_int:
        ptr = nData.ctypes.data_as(POINTER(c_int))
        ret = _MyVNASetIntegerArray(nWhat, nIndex, nArraySize, ptr)

        print(ret, nData)
        
        nD = nData
        rt = ret
        del nData, ret

        return rt, nD

    def Getinstrmode(self):
        nMode = np.array(0, dtype=int)
        ret = _MyVNAGetInstrumentMode(nMode.ctypes.data_as(POINTER(c_int)))
        print('MyVNAGetInstrumentMode\n', ret, nMode)

        nM = nMode
        del nMode

        return ret, nM

    def Setinstrmode(self, nMode=0):
        '''
        nMode: 0, Reflection
        '''
        ret = _MyVNASetInstrumentMode(nMode)
        print('MyVNASetInstrumentMode\n', ret, nMode)
        return ret, nMode


    def Getdisplaymode(self):
        nMode = np.array(0, dtype=int)
        ret = _MyVNAGetDisplayMode(nMode.ctypes.data_as(POINTER(c_int)))
        print('MyVNAGetDisplayMode\n', ret, nMode)
        return ret, nMode

    def Setdisplaymode(self, nMode=0):
        ret = _MyVNASetDisplayMode(nMode)
        print('MyVNASetDisplayMode\n', ret, nMode)
        return ret, nMode
        # __declspec(dllexport) int _stdcall MyVNASetDisplayMode(int nMode)
        # int nRet = MyVNASetDisplayMode( nDisplayMode);

    def SingleScan(self):
        '''
        starts a single scan 
        '''
        # ret = MyVNASingleScan(0x0111, get_hWnd(), 0x0400 + 0x1234, 0)
        hWnd = get_hWnd()
        ret = _MyVNASingleScan(WM_COMMAND, hWnd, MESSAGE_SCAN_ENDED, 0)
        print('MyVNASingleScan\n', ret, hWnd)
        return ret, hWnd

    def EqCctRefine(self):
        hWnd = get_hWnd()
        ret = _MyVNAEqCctRefine(WM_COMMAND, hWnd, MESSAGE_SCAN_ENDED, 0)
        print('MyVNAEqCctRefine\n', ret, hWnd)
        return ret, hWnd

    def SetFequencies(self, f1=4.95e6, f2=5.05e6, nFlags=1):
        '''
        nFlags: 1 = start / stop
        '''
        ret = _MyVNASetFequencies(f1, f2, 1)
        print('MyVNASetFequencies\n', ret, f1, f2) #MyVNASetFequencies
        return ret, f1, f2
    
    @retry(wait_fixed=wait_fixed, stop_max_attempt_number=stop_max_attempt_number, stop_max_delay=stop_max_delay)
    def GetScanData(self, nStart=0, nEnd=299, nWhata=-1, nWhatb=15):

        print('MyVNAGetScanData')
        nSteps = nEnd - nStart + 1
        # nSteps = nSteps * 2
        # print('nSteps=', nSteps)


        data_a = np.zeros(nSteps, dtype=np.float, order='C')
        data_b = np.zeros(nSteps, dtype=np.float, order='C')

        ptr_a = data_a.ctypes.data_as(POINTER(c_double))
        ptr_b = data_b.ctypes.data_as(POINTER(c_double))

        # self.scandata_a.append(ptr_a)
        # self.scandata_b.append(ptr_b)

        ret = _MyVNAGetScanData(nStart, nEnd, nWhata, nWhatb, ptr_a, ptr_b)

        # ret
        #  0: 
        # -1: nSteps < 1
        #  1: crushes before l682: (errcode = MyVNAInit()) == 0
        print(ret, data_a[0], data_b[0])

        da = data_a[:nEnd]
        db = data_b[:nEnd]
        rt = ret
        del data_a, data_b, ret

        return rt, da, db
        
        # simple way
        # return MyVNAGetScanData(nStart, nEnd, nWhata, nWhatb, data_a.ctypes.data_as(POINTER(c_double)), data_b.ctypes.data_as(POINTER(c_double)))


    def Autoscale(self):
        ret = _MyVNAAutoscale()
        print('MyVNAAutoscale\n', ret) #MyVNAAutoscale
        return ret

    ################ combined functions #################
    def single_scan(self):
        print('single_scan')
        # self.Init()
        self.SingleScan()
        self.Autoscale()
        # wait for some time
        time.sleep(1)
        ret, nSteps = self.GetScanSteps()
        ret, f, G = self.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=15)
        # time.sleep(1)
        ret, f, B = self.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=16)
        # self.Close()
        return ret, f, G * 1e3, B * 1e3 # f in Hz; G & B in mS
    
    def change_settings(self, refChn=1, nMode=0, nSteps=400, nAverage=1):
        # ret =           self.Init()
        ret, nMode =    self.Setinstrmode(nMode)
        ret, nData =    self.setADCChannel(refChn)
        ret, nSteps =   self.SetScanSteps(nSteps)
        ret, nAverage = self.SetScanAverage(nAverage)
        # ret =           self.Close()

    def set_steps_freq(self, nSteps=300, f1=4.95e6, f2=5.00e6):
        # set scan parameters
        ret, nSteps =   self.SetScanSteps(nSteps)
        self.SetFequencies(f1, f2, nFlags=1)

    def setADCChannel(self, reflectchn=1):
        # switch ADV channel for test
        # nData = [transChn, reflectchn]
        if reflectchn == 1:
            nData = np.array([2., 1.])
        elif reflectchn == 2:
            nData = np.array([1., 2.])

        ret, nData = self.SetDoubleArray(nWhat=5, nIndex=0, nArraySize=2, nData=nData)
        return ret, nData

    def set_vna(self, setflg):
        '''
        set MyVNA by setflg (dict)
        setflg: {'f1', 'f2', 'steps', 'chn', 'avg', 'speed', ...}
        '''
        for flg, val in setflg.items():
            if val: # val != None
                if flg == 'f': # set frequency
                    ret, f1, f2 = self.SetFequencies(f1=val[0], f2=val[1], nFlags=1)
                elif flg == 'steps': # set scan steps
                    ret, nSteps = self.SetScanSteps(nSteps=val)
                elif flg == 'chn': # set scan channel
                    ret, nData = self.setADCChannel(reflectchn=val)
                elif flg == 'avg': # set scan average
                    ret, nAverage = self.SetScanAverage(nAverage=val)
                elif flg == 'instrmode': # set instrument mode
                    ret, nMode = self.Setinstrmode(nMode=0)
                elif flg == 'speed': # set scan speed
                    # we don't need to change it through python now
                    pass
                else:
                    # add more above else
                    pass
                
                if ret != 0: # error found during setting vna  
                    return ret
            else: # val == None
                # don't need change the default
                pass
        
        return 0

    def get_freq_span(self):
        ''' get frequency span from vna setup '''
        ret, ndResult = self.GetDoubleArray(nWhat=0, nIndex=0, nArraySize=9)
        return ret, ndResult[0:1]




            
            


# get_hWnd()

# exit(0)
if __name__ == '__main__':
    # accvna = AccessMyVNA()
    # ret = accvna.GetDoubleArray()
    # ret, f, G, B = accvna.single_scan()

    with AccessMyVNA() as accvna:
        ret = accvna.ShowWindow(1)
        accvna.GetDoubleArray(nWhat=5, nIndex=0, nArraySize=2)
        accvna.GetIntegerArray(nWhat=5, nIndex=0, nArraySize=4)
        accvna.SetIntegerArray(nWhat=5, nIndex=0, nArraySize=4, nData=[10, 1, 500, 10])
        exit(0)
        accvna.setADCChannel(reflectchn=1)
        accvna.GetDoubleArray(nWhat=5, nIndex=0, nArraySize=2)
        exit(1)
        accvna.setADCChannel(reflectchn=2)
        accvna.GetDoubleArray(nWhat=5, nIndex=0, nArraySize=2)
        exit(0)
        ret, nSteps = accvna.SetScanSteps()
        ret, nSteps = accvna.GetScanSteps() 
        ret, nAverage = accvna.SetScanAverage()
        ret, nAverage = accvna.GetScanAverage()
        ret, nMode = accvna.Setinstrmode()
        ret, nMode = accvna.Getinstrmode()
        ret, nMode = accvna.Setdisplaymode()
        ret, nMode = accvna.Getdisplaymode()
        t0 = time.time()
        ret, nMode = accvna.SingleScan()
        t1 = time.time()
        print(t1-t0)
        # exit(0)
        ret, nMode = accvna.EqCctRefine()
        ret = accvna.SetFequencies()
        # for i in range(100):
        #     print('i:', i)
        #     ret, nResult = accvna.SetDoubleArray(nWhat=5, nIndex=0, nArraySize=2, nData=[1, 2])
        ret, nResult = accvna.GetDoubleArray()
        # print('nR', nResult)
        ret, f, G, B = accvna.single_scan()
        ret, f, G = accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=15)
        # ret = accvna.SingleScan()
        print(ret)

    # vna = AccessMyVNA()
    # vna.Init()
    # vna.SingleScan()
    # vna.GetDoubleArray()                 # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: doesn't affect
    # vna.Close()
    
    
    # accvna = AccessMyVNA() 
    # # call this function before trying to do anything else
    # # Init()
    # get_hWnd()
    # # Init()
    # # ShowWindow(nValue=0)               # AccessMyVNA Closed:OK
    # # # SetScanSteps(nSteps=300)              # AccessMyVNA(Open: click NO; Closed:OK) NyVNA: set when it is closed. after restarted
    # ret, nSteps = accvna.GetScanSteps()                # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: after closed
    # # SetScanAverage(nAverage=1)                 # AccessMyVNA(Open: click NO; Closed:OK) NyVNA: set when it is closed. after restarted
    # # GetScanAverage()                   # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: after closed
    # # Init()
    # # GetDoubleArray(nWhat=5, nIndex=0, nArraySize=2)                 # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: doesn't affect
    # # SetDoubleArray(nWhat=3, nIndex=0, nArraySize=2, nData=np.array([4.9e6, 5.1e6])) # AccessMyVNA(Open: error)
    # # Init()
    # # # Setinstrmode()                      # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: need restart
    # # Getinstrmode()                     # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: need restart
    # # # Setdisplaymode()            # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: need restart
    # # Getdisplaymode()           # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: need restart
    # accvna.SetFequencies()                # AccessMyVNA(O: click NO; C:OK) MyVNA: need restart
    # # Init()
    # accvna.SingleScan()                # AccessMyVNA(O: click NO; C:OK)
    # print('pause 2 s...')
    # time.sleep(2)
    # # # Autoscale()                 # AccessMyVNA(Open: click NO; Closed:OK) MyVNA: doesn't affect
    # # # Init()
    # # # EqCctRefine()      # return 1
    # # # Init()
    # # # follow with SingleScan
    # # # Init()
    # accvna.GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=15) # open a MyVNA window with an error.
    # # Init()
    # # GetScanData(nStart=0, nEnd=nSteps-1, nWhata=-1, nWhatb=16)
    # get_hWnd()
    # # # MUST call this before the calling windows application closes
    # get_hWnd()



# to do
#region
'''
                                    nWhat   nArraySize
GETSET_DISPLAY_OPTIONS              0       4
GETSET_PRINT_OPTIONS                1       4
GETSET_COLOURS                      2       1
GETSETLEFTAXIS                      3
GETSETRIGHTAXIS                     4
GETSETHARDWARESPEEDS                5
GETSETHARDWAREOPTIONS               6
GETSETMARKERSETUP                   7
GETPROGRAMCONSTANTS                 8
GETSETEQCCTCONFIG                   9
GETSETSIMULATIONCONFIG              10
GETSETTRACECALCULATIONOPTIONS       11
GETSETSWITCHATTENUATORCONFIG        12
GETSETSWITCHATTENUATORSETTINGS      13
'''
#endregion

def MyVNAGetString():
    MyVNAGetString = vna[12]
    # // get miscellaneous double array based data
    # __declspec(dllexport) CString _stdcall MyVNAGetString(int nWhat, int nIndex )
    

def MyVNASetString():
    # // Get or set a string parameter
    # // The functions need a parameter to say what is to be set/got - see details below, 
    # // a string for the result / parameter
    # // and an index value
    # // OLE equivalents:
    # // int SetStringAutomation(LONG nWhat, LONG nIndex, BSTR newstring);
    # // BSTR GetStringAutomation(LONG nWhat, LONG nIndex);
    # __declspec(dllexport) int _stdcall MyVNASetString(int nWhat, int nIndex, _TCHAR * sWhat );
    # __declspec(dllexport) CString _stdcall MyVNAGetString(int nWhat, int nIndex );

    # //
    # // options for MyVNAGetString and MyVNASetString
    # // case 0 - equation
    # // nIndex is which one to set/get 0..3
    # #define GETSET_EQUATION 0
    MyVNASetString = vna[28]
    # // set miscellaneous double array based data
    # __declspec(dllexport) int _stdcall MyVNASetString(int nWhat, int nIndex, _TCHAR * sValue )
    

# // MyVNALoadConfiguration()
# // MyVNASaveConfiguration()
# // MyVNALoadCalibration()
# // MyVNASaveCalibration()
# // Given a filename, attempt to load or save the current program configuration
# // including calibration data or just load/save the calibration data
# // finally, option exists to save current trace data to a file ( s2p only supported here)
# // OLE equivalents:
# // int LoadConfigurationAutomation(LPCTSTR fileName);
# // int SaveConfigurationAutomation(LPCTSTR fileName);
# // int LoadCalibrationAutomation(LPCTSTR fileName);
# // int SaveCalibrationAutomation(LPCTSTR fileName);
# // int SaveTraceDataAutomation(LPCTSTR fileName);

def MyVNALoadConfiguration():
    MyVNALoadConfiguration = vna[15]
    # __declspec(dllexport) int _stdcall MyVNALoadConfiguration(_TCHAR * fileName)
    

def MyVNASaveConfiguration():
    MyVNASaveConfiguration = vna[18]
    # __declspec(dllexport) int _stdcall MyVNASaveConfiguration(_TCHAR * fileName)
    

def MyVNALoadCalibration():
    MyVNALoadCalibration = vna[14]
    # __declspec(dllexport) int _stdcall MyVNALoadCalibration(_TCHAR * fileName)
    

def MyVNASaveCalibration():
    MyVNASaveCalibration = vna[17]
    # __declspec(dllexport) int _stdcall MyVNASaveCalibration(_TCHAR * fileName)
    

def MyVNASaveTraceData():
    MyVNASaveTraceData = vna[19]
    # __declspec(dllexport) int _stdcall MyVNASaveTraceData( _TCHAR * fileName )
    

def MyVNAClipboardCopy():
    # // copy whatever is currently being displayed to the clipboard
    # // OLE equivalent:
    # // int ClipboardCopyAutomation(void);
    # __declspec(dllexport) int _stdcall MyVNAClipboardCopy()
    MyVNAClipboardCopy = vna[2]
    
# // when in eq cct view mode (and only then) use these functions to establish a log file and log
# // the results of a scan. The description is a string added to each log entry
# // OLE equivalents:
# // int SetEqCctLogFileAutomation(LPCTSTR fileName);
# // int EqCctLogFileAutomation(LPCTSTR description);

def MyVNASetEqCctLogFile():
    MyVNASetEqCctLogFile = vna[22]
    # __declspec(dllexport) int _stdcall MyVNASetEqCctLogFile(_TCHAR * fileName)
    

def MyVNALogEqCctResults():
    MyVNALogEqCctResults = vna[16]
    # __declspec(dllexport) int _stdcall MyVNALogEqCctResults(_TCHAR * description)
    

