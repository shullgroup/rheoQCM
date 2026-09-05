"""Exercise AccessMyVNA_com against a real instrument, without the GUI.

    python tests/analyzers/myVNA/check_com_interface.py
    python tests/analyzers/myVNA/check_com_interface.py --f1 14.9587e6 --f2 14.9787e6

Runs the same sequence rheoQCM does -- connect, configure, sweep, read -- and
prints what each step returned, so a failure names the call rather than
surfacing later as a blank plot. Touches nothing but the VNA.

The last check is the one that matters: a sweep whose frequency axis spans
the range that was requested. GetScanData returns the *previous* sweep
without complaining if it is asked too early, so a plausible-looking array is
not on its own evidence of anything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from AccessMyVNA_com import AccessMyVNA, MyVNAError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1", type=float, default=4.95e6, help="start, Hz")
    parser.add_argument("--f2", type=float, default=5.05e6, help="stop, Hz")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--averages", type=int, default=1)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--calibration", default=None, help="myVNA .cal file")
    parser.add_argument(
        "--like-gui", action="store_true",
        help="drive it the way rheoQCM does -- set_vna(setflg) then "
             "single_scan() -- and report lengths and NaN counts",
    )
    args = parser.parse_args()

    import struct

    print("python: %s, %d-bit" % (sys.version.split()[0], 8 * struct.calcsize("P")))

    try:
        with AccessMyVNA() as vna:
            if not vna._initialized:
                print("FAILED: could not reach myVNA. Is it installed and "
                      "registered? Try launching it once, elevated.")
                return 1
            # Say only what is true here. The bitness is printed above,
            # and the point is that it does not matter: this passes
            # identically on 32-bit 3.7 and 64-bit 3.13.
            print("connected, with no DLL in the path.")

            print("\nsettings")
            print("  instrument mode  %s" % (vna.Setinstrmode(0)[1],))
            paths = {"ADC%d" % args.channel: args.calibration} if args.calibration else {}
            ret, chn = vna.setADCChannel(args.channel, paths)
            print("  ADC channel      %s (ret %s)%s"
                  % (chn, ret, "  + calibration" if args.calibration else ""))
            print("  steps            %s" % (vna.SetScanSteps(args.steps)[1],))
            print("  averages         %s" % (vna.SetScanAverage(args.averages)[1],))
            ret, f1, f2 = vna.SetFequencies(args.f1, args.f2, nFlags=1)
            print("  frequency        %.1f .. %.1f Hz (ret %s)" % (f1, f2, ret))

            if args.like_gui:
                # get_vna_data_no_with() builds a setflg and calls set_vna,
                # then single_scan. Anything that only breaks in the GUI
                # breaks here too, and this prints what the fitter would see.
                setflg = {
                    "f": [args.f1, args.f2],
                    "steps": args.steps,
                    "chn": args.channel,
                    "cal": {},
                }
                print("\nset_vna(%s)" % setflg)
                ret = vna.set_vna(setflg)
                print("  ret %s, _nsteps now %s, _f now %s"
                      % (ret, vna._nsteps, vna._f))

            print("\nsweeping")
            started = time.time()
            ret, f, G, B = vna.single_scan()
            elapsed = time.time() - started
            print("  single_scan      ret %s in %.2f s" % (ret, elapsed))
            print("  points           %d (expected %d)" % (len(f), args.steps + 1))
            print("  axis             %.1f .. %.1f Hz" % (f[0], f[-1]))
            print("  G                %.4f .. %.4f mS" % (G.min(), G.max()))
            print("  B                %.4f .. %.4f mS" % (B.min(), B.max()))
            print("  lengths          f=%d G=%d B=%d" % (len(f), len(G), len(B)))
            print("  NaNs             f=%d G=%d B=%d"
                  % (int(np.isnan(f).sum()), int(np.isnan(G).sum()),
                     int(np.isnan(B).sum())))
            print("  finite G         %d of %d"
                  % (int(np.isfinite(G).sum()), len(G)))

            problems = []
            if len(f) != args.steps + 1:
                problems.append(
                    "got %d points for %d steps; a scan of N steps gives N+1"
                    % (len(f), args.steps)
                )
            if not np.all(np.diff(f) > 0):
                problems.append("the frequency axis is not increasing")
            tolerance = max(abs(f2 - f1) * 0.01, 1.0)
            if abs(f[0] - f1) > tolerance or abs(f[-1] - f2) > tolerance:
                problems.append(
                    "the axis spans %.1f..%.1f but %.1f..%.1f was requested -- "
                    "this is the previous sweep" % (f[0], f[-1], f1, f2)
                )
            if not np.any(G):
                problems.append("G is all zeros")

            print()
            if problems:
                for p in problems:
                    print("FAILED: %s" % p)
                return 1
            peak = int(np.argmax(G))
            print("PASSED. Peak G at %.1f Hz." % f[peak])
            print("This is the path rheoQCM uses with vna_use_dll = False.")
            return 0

    except MyVNAError as exc:
        print("\nFAILED: %s" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
