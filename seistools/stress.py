import numpy as np

def calc_ds_2d(slip, dx, mu, poisson = 0., expand = 0):
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

    newlen = len(slip) + 2*expand
    newslip = np.zeros(newlen)

    newslip[expand:expand+len(slip)] = slip[:]
    
    k = np.fft.fftfreq(newlen, dx)

    f = np.fft.fft(newslip)

    f *= -mu/(1.-poisson)*np.abs(k)

    return np.real(np.fft.ifft(f))[expand:expand+len(slip)]    

def calc_ds_3d(slip_x, slip_y, dx, dy, mu, poisson, expand = 0):
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

    newlenx = len(slip_x)+2*expand
    newleny = len(slip_x[0])+2*expand

    newslipx = np.zeros((newlenx, newleny))
    newslipy = np.zeros((newlenx, newleny))
    newslipx[expand:expand+len(slip_x),expand:expand+len(slip_x[0])] = slip_x
    newslipy[expand:expand+len(slip_x),expand:expand+len(slip_x[0])] = slip_y
    
    k = np.fft.fftfreq(newlenx, dx)
    m = np.fft.fftfreq(newleny, dy)

    kxy, mxy = np.meshgrid(k, m, indexing='ij')

    kmag = np.sqrt(kxy**2+mxy**2)
    kmag[0,0] = 1.

    fx = np.fft.fft2(newslipx)
    fy = np.fft.fft2(newslipy)

    sx = -mu/2./kmag*(1./(1.-poisson)*(kxy**2*fx+mxy*kxy*fy)+(mxy**2*fx-mxy*kxy*fy))
    sy = -mu/2./kmag*(1./(1.-poisson)*(mxy**2*fy+mxy*kxy*fx)+(kxy**2*fy-mxy*kxy*fx))

    return (np.real(np.fft.ifft2(sx))[expand:expand+len(slip_x),expand:expand+len(slip_x[0])],
            np.real(np.fft.ifft2(sy))[expand:expand+len(slip_x),expand:expand+len(slip_x[0])])
