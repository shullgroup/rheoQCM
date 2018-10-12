import numpy as np

def datarange(data):
    # find the min and max of data
    if any(data):
        return [min(data), max(data)]
    else:
        return [None, None]

def num2str(A,precision=None):
    if isinstance(A, np.ndarray):
        if A.any() and not precision:
            return A.astype(str)
        elif A.any() and precision:
            for i in range(len(A)):
                A[i] = format(float(A[i]), '.'+str(precision)+'g')
                # A[i] = '{:.6g}'.format(float(A[i]))
            return A.astype(str)
    elif isinstance(A, float) or isinstance(A, int):
        if A and not precision:
            A = str(float(A))
            return A
        elif A and precision:
            A = format(float(A), '.'+str(precision)+'g')
            # A = '{:.6g}'.format(float(A))
            return A


def converter_startstop_to_centerspan(f1, f2):
    '''convert start/stop (f1/f2) to center/span (fc/fs)'''
    fc = (f1 + f2) / 2
    fs = f2 - f1
    return [fc, fs]

def converter_centerspan_to_startstop(fc, fs):
    '''convert center/span (fc/fs) to start/stop (f1/f2)'''
    f1 = fc - fs / 2
    f2 = fc + fs / 2
    return [f1, f2]
