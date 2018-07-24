import math
import numpy as np

def num2str(A,precision):
    if type(A) == np.ndarray:
        array = A
        for i in range(len(array)):
            array[i] = format(array[i], '.'+str(precision)+'g')
        return array
