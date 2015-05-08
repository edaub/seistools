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

def calc_ds_3d(slip_x, slip_y, dx, dy, mu, poisson):
    """
    Calculates stress change due to slip on a planar fault (using FFT method)
    Inputs:
    slip_x = x-displacement (2d array)
    slip_y = y-displacement (2d array)
    dx = grid spacing (assumed to be uniform)
    mu = shear modulus
    poisson (optional) = poisson's ratio (if mode 2)
    Returns:
    static stress change from fault slip
    """
    k = np.fft.fftfreq(len(slip_x), dx)
    m = np.fft.fftfreq(len(slip_x[0]), dy)

    kxy, mxy = np.meshgrid(k, m, indexing='ij')

    kmag = np.sqrt(kxy**2+mxy**2)
    kmag[0,0] = 1.

    fx = np.fft.fft2(slip_x)
    fy = np.fft.fft2(slip_y)

    sx = -mu/2./kmag*(1./(1.-poisson)*(kxy**2*fx+mxy*kxy*fy)+(mxy**2*fx-mxy*kxy*fy))
    sy = -mu/2./kmag*(1./(1.-poisson)*(mxy**2*fy+mxy*kxy*fx)+(kxy**2*fy-mxy*kxy*fx))

    return np.real(np.fft.ifft2(sx)), np.real(np.fft.ifft2(sy))
