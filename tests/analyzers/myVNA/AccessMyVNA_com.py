"""AccessMyVNA over OLE automation -- no DLL, no 32-bit Python.

Drop-in for ``AccessMyVNA.py``. Same class, same method names, same return
tuples, so nothing above this file changes:

    from analyzers.myVNA.AccessMyVNA_com import AccessMyVNA
    with AccessMyVNA() as vna:
        vna.change_settings(refChn=1, nMode=0, nSteps=400, nAverage=1)
        vna.set_steps_freq(nSteps=400, f1=4.95e6, f2=5.00e6)
        ret, f, G, B = vna.single_scan()

WHY THIS EXISTS
---------------
``MyVNAGetScanData`` fails on the Segalman rig with MSVC C++ exception
0xE06D7363 for every argument combination. Disassembling the shim shows why:
it sizes its SafeArrays ``nEnd - nStart`` (RVA 0x28a0) while myVNA writes
``nEnd - nStart + 1`` doubles into them, because nEnd is inclusive --
Appendix D, "up to and including nEnd. note that a scan of N steps gives N+1
data points from 0 to N". That is a one-element heap overrun on every call
whatever the caller passes, and no value of nEnd makes the array the right
size. An 8-byte overrun is tolerated or fatal depending on the allocation,
which is how the same binaries work in Evanston and fault in Santa Barbara.

Talking to the automation server directly avoids the shim entirely, and with
it the 32-bit requirement: myVNA is an out-of-process server, so its
automation marshals across bitness. Verified on 64-bit Python 3.13.

Measured against the DLL on one crystal, 100 scans a cell, n=3 at 14.9687
MHz: this path 100/100 at 1.767 s and 0.284 Hz f0 scatter; SaveTraceData
100/100 at 1.786 s and 0.339 Hz; MyVNAGetScanData faulted on its first sweep
with the window both minimised and restored. An 8.3 hour unattended run over
18,504 cycles on four overtones produced no failures.

DIFFERENCES FROM AccessMyVNA.py, all deliberate
-----------------------------------------------
1. ``GetScanData(nStart, nEnd)`` treats nEnd as inclusive and allocates
   ``nEnd - nStart + 1``, which is what myVNA actually writes. The existing
   call ``GetScanData(nStart=0, nEnd=int(self._nsteps))`` therefore becomes
   correct with no change at the call site -- the ``nEnd + 1`` compensation
   moves inside and cancels.
2. ``set_vna`` raises ``MyVNAError`` where the original calls ``exit(0)``.
   Killing the interpreter from library code takes the GUI down with it and
   loses the run.
3. ``ScanMessageReceiver`` is duplicated here rather than imported, because
   ``AccessMyVNA.py`` loads the 32-bit DLL at module scope (line 75) and
   cannot be imported at all under 64-bit Python. The logic is Qifeng Wang's,
   unchanged; passing the receiver's own HWND rather than myVNA's is the part
   that makes completion work, and it is right.

Requires pywin32. Nothing else.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pythoncom
    import win32com.client
    import win32con
    import win32gui
except ImportError:  # pragma: no cover - reported at connect, not at import
    pythoncom = win32com = win32con = win32gui = None

#: myVNA's automation coclass, from Appendix D of the manual and from the
#: CLSID the shim hands to COleDispatchDriver::CreateDispatch.
MYVNA_CLSID = "{4D9364D3-3187-4CEA-A28D-5DC06E343344}"

WM_COMMAND = 0x0111
WM_USER = 0x0400
MESSAGE_SCAN_ENDED = WM_USER + 0x1234

#: Seconds to wait for the scan-completion message. Upstream's default is 2
#: and its config raised it to 5; with the completion message actually
#: arriving there is no reason to run it close.
vna_timeout = 30.0

# Display types, Appendix D. 15/16 are the *reflection* forms, which is what
# INSTRUMENT_MODE_REFLECTION (0) produces -- the QCM mode.
DISPLAY_NOTHING = -2
DISPLAY_FREQ_SCALE = -1
DISPLAY_REFL_GP = 15
DISPLAY_REFL_BP = 16

INSTRUMENT_MODE_REFLECTION = 0

# Selectors for the array getters/setters.
GETSET_SCAN_FREQ = 0
GETSET_DISPLAY_FT_AXIS = 3
GETSET_ADC_CHANNEL = 5
GETSET_HARDWARE_SPEEDS = 5


class MyVNAError(RuntimeError):
    """myVNA refused a call, or could not be reached."""


def _com_detail(exc) -> str:
    """Unpack a pythoncom.com_error into something worth reading."""
    try:
        hresult, message, excepinfo, argerr = exc.args
    except (AttributeError, ValueError):
        return str(exc)
    parts = ["HRESULT 0x%08X (%s)" % (hresult & 0xFFFFFFFF, message)]
    if (hresult & 0xFFFFFFFF) == 0x80010105:
        parts.append(
            "RPC_E_SERVERFAULT -- myVNA raised a structured exception inside "
            "the call. Not a refusal; a fault in myVNA.exe."
        )
    if excepinfo:
        description = (list(excepinfo) + [None] * 6)[2]
        if description:
            parts.append("myVNA said: %s" % str(description).strip())
    if argerr is not None:
        parts.append("bad argument index: %s" % argerr)
    return " | ".join(parts)


def _array_arg(values, double=True):
    """A VARIANT* holding a typed safearray, which is what myVNA requires.

    Appendix D: "VARIANT a and b must encapsulate safearrays of doubles of
    sufficient size for the output and be zero indexed."

    The element type has to be stated. ``VARIANT(VT_ARRAY|VT_R8|VT_BYREF)``
    produces a VARIANT holding a SAFEARRAY** -- one indirection too many --
    and myVNA faults dereferencing it, on *every* array call including ones
    that plainly work. Measured against SetDoubleArray, which the DLL issues
    on every sweep and therefore cannot be broken.
    """
    from win32com.client import VARIANT

    element = pythoncom.VT_R8 if double else pythoncom.VT_I4
    inner = VARIANT(pythoncom.VT_ARRAY | element, list(values))
    return VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_BYREF, inner)


def _array_out(arg):
    """Read back what a call left in an ``_array_arg``."""
    inner = getattr(arg, "value", None)
    return list(getattr(inner, "value", inner) or [])


class ScanMessageReceiver:
    """An invisible window that catches myVNA's scan-completion message.

    Qifeng Wang's design, from AccessMyVNA.py, duplicated because that module
    cannot be imported under 64-bit Python. ``SingleScan`` is given *this*
    window rather than myVNA's, which is what makes the notification arrive
    at the caller instead of going back to myVNA.
    """

    _CLASS_NAME = "AccessMyVNAComScanReceiver"

    def __init__(self):
        if win32gui is None:
            raise MyVNAError("pywin32 is required: pip install pywin32")
        self.scan_finished = threading.Event()
        self.hwnd = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise MyVNAError("the scan-completion window did not come up")

    def _pump(self):
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self._on_message
        wc.lpszClassName = self._CLASS_NAME
        wc.hInstance = win32gui.GetModuleHandle(None)
        try:
            win32gui.RegisterClass(wc)
        except Exception:  # noqa: BLE001 - already registered is fine
            logger.debug("window class already registered")
        self.hwnd = win32gui.CreateWindow(
            self._CLASS_NAME, "AccessMyVNA COM", 0, 0, 0, 0, 0, 0, 0,
            wc.hInstance, None,
        )
        self._ready.set()
        win32gui.PumpMessages()

    def _on_message(self, hwnd, message, wparam, lparam):
        if message == WM_COMMAND and (
            int(wparam) == MESSAGE_SCAN_ENDED
            or (int(wparam) & 0xFFFF) == (MESSAGE_SCAN_ENDED & 0xFFFF)
            or int(lparam) == MESSAGE_SCAN_ENDED
        ):
            self.scan_finished.set()
            return 0
        if message == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

    def prepare_scan(self):
        self.scan_finished.clear()

    def wait(self, timeout):
        return self.scan_finished.wait(timeout)

    def self_test(self, timeout=1.0):
        self.prepare_scan()
        win32gui.PostMessage(self.hwnd, WM_COMMAND, MESSAGE_SCAN_ENDED, 0)
        if not self.wait(timeout):
            raise MyVNAError("the scan receiver is not processing WM_COMMAND")
        self.prepare_scan()

    def close(self):
        if self.hwnd and win32gui.IsWindow(self.hwnd):
            win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
        self.hwnd = None


class AccessMyVNA:
    """The same interface as AccessMyVNA.py, over automation."""

    def __init__(self):
        self._obj = None
        self._initialized = False
        self._reconnecting = False
        self._scan_receiver = None
        self._nsteps = 400
        self._naverage = 1
        self._f = [4.95e6, 5.05e6]
        self._chn = 1
        self._instrmode = INSTRUMENT_MODE_REFLECTION
        self._displaymode = 0
        self._speed = None
        self._step_delay = None
        self._start_delay = None
        self._phase_delay = None

    # ------------------------------------------------------------ lifecycle

    def __enter__(self):
        self.Init()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Deliberately does NOT close the connection.

        rheoQCM enters this context manager once per collection cycle --
        ``with self.vna:`` at rheoQCM.py:7163, around all the harmonics for
        one channel -- so closing on exit would rebuild the dispatch object
        and the message window before every scan, about 1.5 s of pure
        overhead per cycle.

        It also matches what the DLL actually did. MyVNAClose does not
        release the dispatch driver, which is a module-level global; that is
        the same leftover that makes a second MyVNAInit return -1. The app has
        always run as though the connection persists, because it did.

        Close() still works and is still the way to shut down.
        """
        return False

    def Init(self):
        """Connect. Idempotent, and never raises -- it returns a code.

        Both matter for how rheoQCM calls this. It does
        ``with AccessMyVNA() as vna:`` and then ``vna.Init()`` inside the
        block (rheoQCM.py:5753), so Init runs twice; without the early return
        the second call would build a second dispatch object and a second
        message window. And the call site is wrapped in a bare ``except:``
        that prints "Initiating MyVNA failed!" with no detail, so anything
        raised here disappears -- returning a code and logging the cause is
        the only way the reason reaches the log.
        """
        if self._initialized and self._obj is not None:
            return 0
        if win32com is None:
            logger.error("pywin32 is required: pip install pywin32")
            return -1
        try:
            self._obj = win32com.client.Dispatch(MYVNA_CLSID)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not reach myVNA: %s", _com_detail(exc))
            self._obj = None
            return -1
        try:
            self._scan_receiver = ScanMessageReceiver()
            self._scan_receiver.self_test()
            # Read back rather than assume; myVNA keeps its own state.
            _, self._nsteps = self.GetScanSteps()
            _, self._naverage = self.GetScanAverage()
        except Exception as exc:  # noqa: BLE001 - must not raise past here
            logger.exception("myVNA connected but could not be set up: %s", exc)
            self.Close()
            return -2
        self._initialized = True
        return 0

    def Close(self):
        if self._scan_receiver is not None:
            self._scan_receiver.close()
            self._scan_receiver = None
        self._obj = None
        self._initialized = False
        return 0

    def _require(self):
        """The live automation object, reconnecting if it has been closed.

        rheoQCM keeps a reference past the ``with`` block that created it --
        ``with AccessMyVNA() as vna: ... self.vna = vna`` at rheoQCM.py:5742 --
        and then calls ``single_scan()`` on it for the rest of the session.
        With the DLL that happened to keep working, because the dispatch
        driver is a module-level global that MyVNAClose does not really
        release. Reconnecting on demand keeps that call site working rather
        than making the app crash on its first scan.
        """
        if self._obj is None:
            if not self._reconnecting:
                self._reconnecting = True
                try:
                    logger.info("automation object was closed; reconnecting")
                    if self.Init() == 0:
                        return self._obj
                finally:
                    self._reconnecting = False
            raise MyVNAError("could not reach myVNA.")
        return self._obj

    def _call(self, what, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            raise MyVNAError("%s failed: %s" % (what, _com_detail(exc))) from exc

    # ------------------------------------------------------------- settings

    def ShowWindow(self, nValue=1):
        obj = self._require()
        return self._call("ShowWindow", obj.ShowWindow, int(nValue))

    def SetScanSteps(self, nSteps=400):
        obj = self._require()
        # Appendix D: min 2, max 50,000.
        obj.ScanPoints = int(nSteps)
        self._nsteps = int(obj.ScanPoints)
        return 0, self._nsteps

    def GetScanSteps(self):
        obj = self._require()
        self._nsteps = int(obj.ScanPoints)
        return 0, self._nsteps

    def SetScanAverage(self, nAverage=1):
        obj = self._require()
        obj.ScanAverage = int(nAverage)
        self._naverage = int(obj.ScanAverage)
        return 0, self._naverage

    def GetScanAverage(self):
        obj = self._require()
        self._naverage = int(obj.ScanAverage)
        return 0, self._naverage

    def Getinstrmode(self):
        obj = self._require()
        self._instrmode = int(obj.InstrumentMode)
        return 0, self._instrmode

    def Setinstrmode(self, nMode=0):
        obj = self._require()
        obj.InstrumentMode = int(nMode)
        self._instrmode = int(obj.InstrumentMode)
        return 0, self._instrmode

    def Getdisplaymode(self):
        obj = self._require()
        self._displaymode = int(obj.DisplayMode)
        return 0, self._displaymode

    def Setdisplaymode(self, nMode=0):
        obj = self._require()
        obj.DisplayMode = int(nMode)
        self._displaymode = int(obj.DisplayMode)
        return 0, self._displaymode

    def SetFequencies(self, f1=4.95e6, f2=5.05e6, nFlags=1):
        """Dispatched as ScanFreqs, not SetScanDetails.

        Appendix D documents the C entry point as SetScanDetailsAutomation,
        but the name IDispatch resolves is ScanFreqs (dispid 4).
        """
        obj = self._require()
        ret = self._call("ScanFreqs", obj.ScanFreqs, float(f1), float(f2), int(nFlags))
        if ret == 0:
            self._f = [float(f1), float(f2)]
        return ret, float(f1), float(f2)

    # ---------------------------------------------------------- array pairs

    def GetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9):
        obj = self._require()
        arg = _array_arg([0.0] * nArraySize, double=True)
        ret = self._call(
            "GetDoubleArray", obj.GetDoubleArray, int(nWhat), int(nIndex),
            int(nArraySize), arg,
        )
        return ret, np.array(_array_out(arg), dtype=np.float64)

    def SetDoubleArray(self, nWhat=0, nIndex=0, nArraySize=9, nData=None):
        obj = self._require()
        values = list(np.asarray(nData, dtype=np.float64).ravel())
        arg = _array_arg(values, double=True)
        ret = self._call(
            "SetDoubleArray", obj.SetDoubleArray, int(nWhat), int(nIndex),
            int(nArraySize), arg,
        )
        return ret, np.asarray(nData)

    def GetIntegerArray(self, nWhat=5, nIndex=0, nArraySize=4):
        obj = self._require()
        arg = _array_arg([0] * nArraySize, double=False)
        ret = self._call(
            "GetIntegerArray", obj.GetIntegerArray, int(nWhat), int(nIndex),
            int(nArraySize), arg,
        )
        return ret, np.array(_array_out(arg), dtype=int)

    def SetIntegerArray(self, nWhat=5, nIndex=0, nArraySize=4, nData=None):
        obj = self._require()
        values = [int(v) for v in np.asarray(nData).ravel()]
        arg = _array_arg(values, double=False)
        ret = self._call(
            "SetIntegerArray", obj.SetIntegerArray, int(nWhat), int(nIndex),
            int(nArraySize), arg,
        )
        return ret, np.asarray(nData)

    # ------------------------------------------------------------- scanning

    def SingleScan(self, timeout=vna_timeout):
        obj = self._require()
        if self._scan_receiver is None:
            raise MyVNAError("Scan message receiver is not initialized.")
        self._scan_receiver.prepare_scan()
        # The receiver's own HWND, not myVNA's -- see the class docstring.
        ret = self._call(
            "SingleScan", obj.SingleScan, WM_COMMAND,
            int(self._scan_receiver.hwnd), MESSAGE_SCAN_ENDED, 0,
        )
        if ret != 0:
            return ret
        if not self._scan_receiver.wait(timeout):
            raise MyVNAError(
                "myVNA started a sweep but sent no completion message within "
                "%.1f s" % timeout
            )
        return 0

    def GetScanData(self, nStart=0, nEnd=400, nWhata=-1, nWhatb=15):
        """Read the finished sweep. **nEnd is inclusive.**

        A scan of N steps gives N+1 points from 0 to N, so nStart..nEnd
        inclusive is ``nEnd - nStart + 1`` values and the arrays are sized to
        match. The DLL sizes them ``nEnd - nStart``, one short of what myVNA
        writes, which is the overrun this module exists to avoid -- so the
        familiar ``nEnd + 1`` compensation is gone and callers pass the real
        last index.
        """
        obj = self._require()
        nSamples = int(nEnd) - int(nStart) + 1
        a = _array_arg([0.0] * nSamples, double=True)
        b = _array_arg([0.0] * nSamples, double=True)
        ret = self._call(
            "GetScanData", obj.GetScanData, int(nStart), int(nEnd),
            int(nWhata), int(nWhatb), a, b,
        )
        da = np.array(_array_out(a), dtype=np.float64)
        db = np.array(_array_out(b), dtype=np.float64)
        return ret, da, db

    def Autoscale(self):
        return self._void_call("Autoscale", "Autoscale")

    def _void_call(self, what, name):
        """Invoke a no-argument automation method.

        pywin32's late binding tries DISPATCH_PROPERTYGET first, and for a
        method that takes no arguments that *is* the invocation -- the
        attribute access runs it and hands back its int. Calling the result
        then fails with "'int' object is not callable". With a type library
        pywin32 would know the difference, but myVNA's GetTypeInfo returns
        E_NOTIMPL, so there is nothing to disambiguate from and this has to
        cope with either shape.
        """
        obj = self._require()
        try:
            result = getattr(obj, name)
            if callable(result):
                result = result()
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            raise MyVNAError("%s failed: %s" % (what, _com_detail(exc))) from exc
        return 0 if result is None else int(result)

    def EqCctRefine(self):
        obj = self._require()
        hWnd = int(self._scan_receiver.hwnd) if self._scan_receiver else 0
        ret = self._call(
            "EqCctRefine", getattr(obj, "Eq Cct Refine"),
            WM_COMMAND, hWnd, MESSAGE_SCAN_ENDED, 0,
        )
        return ret, hWnd

    # ---------------------------------------------------------------- files

    def LoadCalibration(self, fileName):
        obj = self._require()
        return self._call("LoadCalibration", obj.LoadCalibration, str(fileName))

    def SaveCalibration(self, fileName):
        obj = self._require()
        return self._call("SaveCalibration", obj.SaveCalibration, str(fileName))

    def LoadConfiguration(self, fileName):
        obj = self._require()
        return self._call(
            "LoadConfiguration", obj.LoadConfiguration, str(fileName)
        )

    def SaveConfiguration(self, fileName):
        obj = self._require()
        return self._call(
            "SaveConfiguration", obj.SaveConfiguration, str(fileName)
        )

    def SaveTraceData(self, fileName):
        obj = self._require()
        return self._call("SaveTraceData", obj.SaveTraceData, str(fileName))

    # ------------------------------------------------------- the app's API

    def single_scan(self):
        """Trigger a sweep and return ``(ret, f, G, B)`` with G/B in mS."""
        self.SingleScan()
        self.setDisplayFreq()
        self.Autoscale()
        ret, f, G = self.GetScanData(
            nStart=0, nEnd=int(self._nsteps),
            nWhata=DISPLAY_FREQ_SCALE, nWhatb=DISPLAY_REFL_GP,
        )
        ret, _, B = self.GetScanData(
            nStart=0, nEnd=int(self._nsteps),
            nWhata=DISPLAY_NOTHING, nWhatb=DISPLAY_REFL_BP,
        )
        return ret, f, G * 1000, B * 1000  # f in Hz; G & B in mS

    def setDisplayFreq(self):
        """Point myVNA's own x-axis at the sweep. Cosmetic only."""
        ret, _ = self.SetDoubleArray(
            nWhat=GETSET_DISPLAY_FT_AXIS, nIndex=0, nArraySize=2,
            nData=np.array(self._f, dtype=np.float64),
        )
        return ret

    def change_settings(self, refChn=1, nMode=0, nSteps=400, nAverage=1):
        self.Setinstrmode(nMode)
        self.setADCChannel(refChn)
        self.SetScanSteps(nSteps)
        self.SetScanAverage(nAverage)

    def set_steps_freq(self, nSteps=400, f1=4.95e6, f2=5.00e6):
        self.SetScanSteps(nSteps)
        self.SetFequencies(f1, f2, nFlags=1)

    def setADCChannel(self, reflectchn=1, paths=None):
        """Select the ADC channel, and load its calibration if given."""
        nData = np.array([2.0, 1.0] if reflectchn == 1 else [1.0, 2.0])
        ret, _ = self.SetDoubleArray(
            nWhat=GETSET_ADC_CHANNEL, nIndex=0, nArraySize=2, nData=nData
        )
        paths = paths or {}
        key = "ADC" + str(reflectchn)
        if paths.get(key):
            ret += self.LoadCalibration(paths[key])
        if ret == 0:
            self._chn = reflectchn
        return ret, reflectchn

    def getADCChannel(self):
        ret, nData = self.GetDoubleArray(
            nWhat=GETSET_ADC_CHANNEL, nIndex=0, nArraySize=2
        )
        logger.info("getADCChannel %s", nData)
        if ret == 0 and len(nData) >= 2:
            self._chn = 1 if nData[0] > nData[1] else 2
        return ret, self._chn

    def set_vna(self, setflg):
        """Apply a settings dict.

        Raises rather than calling ``exit(0)``: killing the interpreter from
        library code takes the GUI with it and loses the run.
        """
        flg_list = ["f", "steps", "chn", "avg", "speed", "instrmode"]
        for flg, val in setflg.items():
            if val is None or flg not in flg_list:
                continue
            if flg == "f" and self._f != val:
                ret, self._f[0], self._f[1] = self.SetFequencies(
                    f1=val[0], f2=val[1], nFlags=1
                )
                self._fail_if(ret, "SetFequencies")
            elif flg == "steps" and self._nsteps != val:
                ret, self._nsteps = self.SetScanSteps(nSteps=val)
                self._fail_if(ret, "SetScanSteps")
            elif flg == "chn" and val != "none" and self._chn != int(val):
                ret, self._chn = self.setADCChannel(
                    reflectchn=int(val), paths=setflg.get("cal", {})
                )
                self._fail_if(ret, "setADCChannel")
            elif flg == "avg" and self._naverage != val:
                ret, self._naverage = self.SetScanAverage(nAverage=val)
                self._fail_if(ret, "SetScanAverage")
            elif flg == "instrmode" and self._instrmode != val:
                ret, self._instrmode = self.Setinstrmode(nMode=val)
                self._fail_if(ret, "Setinstrmode")
        return 0

    @staticmethod
    def _fail_if(ret, what):
        if ret != 0:
            raise MyVNAError("%s returned %s" % (what, ret))

    def get_freq_span(self):
        ret, nData = self.GetDoubleArray(
            nWhat=GETSET_SCAN_FREQ, nIndex=0, nArraySize=9
        )
        if ret == 0 and len(nData) >= 2:
            self._f = [float(nData[0]), float(nData[1])]
        return ret, self._f

    def get_speed_delays(self):
        ret, nData = self.GetIntegerArray(
            nWhat=GETSET_HARDWARE_SPEEDS, nIndex=0, nArraySize=4
        )
        if ret == 0 and len(nData) >= 4:
            (self._speed, self._step_delay,
             self._start_delay, self._phase_delay) = (int(v) for v in nData[:4])
        return ret, nData
