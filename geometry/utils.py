import numpy as np

def resample_linear_1d(original, targetLen):
    '''
        Similar to scipy resample but with no aberration, see:
            https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
    '''
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp
