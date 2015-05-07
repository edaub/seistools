import numpy as np

def calc_ds_2d(slip, dx, mu, poisson = 0.):
    """
    Calculates stress change due to slip on a planar fault (using FFT method)
    Inputs:
    slip = displacement (1d array)
    dx = grid spacing (assumed to be uniform)
    mu = shear modulus
    poisson (optional) = poisson's ratio (if mode 2)
    Returns:
    static stress change from fault slip
    """    
    k = np.fft.fftfreq(len(slip), dx)

    f = np.fft.fft(slip)

    f *= -mu/(1.-poisson)*np.abs(k)

    return np.real(np.fft.ifft(f))    
