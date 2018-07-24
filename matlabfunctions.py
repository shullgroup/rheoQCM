import math
import numpy as np

def num2str(A,precision=None, formatSpec=None):
    if isinstance(A, np.ndarray):
        if A.any() and not precision and not formatSpec:
            for i in range(len(A)):
                A[i] = str(float(A[i]))
            print(A)
            return A
        elif A.any() and precision and not formatSpec:
            for i in range(len(A)):
                A[i] = format(float(A[i]), '.'+str(precision)+'g')
            print(A)
            return A
        elif A.any() and formatSpec:
            print('not implemented')
        else:
            print('not available')
    elif isinstance(A, float) or isinstance(A, int):
        if A and not precision and not formatSpec:
            A = str(float(A))
            print(A)
            return A
        elif A and precision and not formatSpec:
            A = format(float(A), '.'+str(precision)+'g')
            print(A)
            return A
        elif A and formatSpec:
            print('not implemented')
        else:
            print('not available')
