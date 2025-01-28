import numpy as np
import polars as pl
import scipy.fftpack as fftpack

def low_pass_filter(array, fraction):
    transform = fftpack.fft(array)
    length = array.shape[0]
    end_length = ((1 - fraction)/2)
    start = round(length*end_length)
    end = round(length*(1-end_length))
    transform[start:end] = np.zeros_like(transform[start:end])
    output = fftpack.ifft(transform)
    # output.real
    return output.real