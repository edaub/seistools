import numpy as np
from scipy.integrate import simps

def generate_profile(npoints, length, alpha, window, seed=None):
    """
    Generates fractal fault profile with zero mean and a given amplitude/wavelength ratio
    Inputs:
    npoints = number of grid points
    length = length of fault in physical domain
    alpha = amplitude to wavelength ratio
    window = length of minimum wavelength in grid points
    seed = seed for random number generator
    Returns:
    heights of fault profile (array-like)
    """
    prng = np.random.RandomState(seed)
    phase = 2.*np.pi*prng.rand(npoints+window-1)
    k = np.fft.fftfreq(npoints+window-1,length/float(npoints+window-2))
    amp = np.zeros(npoints+window-1)
    amp[1:] = (2.*np.pi)**0.5*alpha*prng.rand(npoints+window-2)/np.abs(k[1:])**1.5/np.sqrt(length)
    f = amp*np.exp(np.complex(0., 1.)*phase)
    f = np.real(np.fft.fft(f))
    f = np.convolve(f,np.ones(window),'valid')/float(window)
    return f-f[0]-(f[-1]-f[0])/length*np.linspace(0., length, npoints)

def calc_diff(f, dx):
    """
    Calculates derivative using 4th order finite differences
    Inputs:
    f = function
    dx = grid spacing
    Returns:
    derivative (array-like)
    """
    
    df = (np.roll(f,-3)/60.-np.roll(f,-2)*3./20.+np.roll(f,-1)*3./4.-np.roll(f,1)*3./4.+np.roll(f,2)*3./20.-np.roll(f,3)/60.)/dx
    df[0] = (-21600./13649.*f[0]+81763./40947.*f[1]+131./27298.*f[2]-9143./13649.*f[3]+20539./81894.*f[4])/dx
    df[1] = (-81763./180195.*f[0]+7357./36039.*f[2]+30637./72078.*f[3]-2328./12013.*f[4]+6611./360390.*f[5])/dx
    df[2] = (-131./54220.*f[0]-7357./16266.*f[1]+645./2711.*f[3]+11237./32532.*f[4]-3487./27110.*f[5])/dx
    df[3] = (9143./53590.*f[0]-30637./64308.*f[1]-645./5359.*f[2]+13733./32154.*f[4]-67./4660.*f[5]+72./5359.*f[6])/dx
    df[4] = (-20539./236310.*f[0]+2328./7877.*f[1]-11237./47262.*f[2]-13733./23631.*f[3]+89387./118155.*f[5]-1296./7877.*f[6]+144./7877.*f[7])/dx
    df[5] = (-6611./262806.*f[1]+3487./43801.*f[2]+1541./87602.*f[3]-89387./131403.*f[4]+32400./43801.*f[6]-6480./43801.*f[7]+720./43801.*f[8])/dx
    df[-1] = -(-21600./13649.*f[-1]+81763./40947.*f[-2]+131./27298.*f[-3]-9143./13649.*f[-4]+20539./81894.*f[-5])/dx
    df[-2] = -(-81763./180195.*f[-1]+7357./36039.*f[-3]+30637./72078.*f[-4]-2328./12013.*f[-5]+6611./360390.*f[-6])/dx
    df[-3] = -(-131./54220.*f[-1]-7357./16266.*f[-2]+645./2711.*f[-4]+11237./32532.*f[-5]-3487./27110.*f[-6])/dx
    df[-4] = -(9143./53590.*f[-1]-30637./64308.*f[-2]-645./5359.*f[-3]+13733./32154.*f[-5]-67./4660.*f[-6]+72./5359.*f[-7])/dx
    df[-5] = -(-20539./236310.*f[-1]+2328./7877.*f[-2]-11237./47262.*f[-3]-13733./23631.*f[-4]+89387./118155.*f[-6]-1296./7877.*f[-7]+144./7877.*f[-8])/dx
    df[-6] = -(-6611./262806.*f[-2]+3487./43801.*f[-3]+1541./87602.*f[-4]-89387./131403.*f[-5]+32400./43801.*f[-7]-6480./43801.*f[-8]+720./43801.*f[-9])/dx

    return df

def calc_hrms(x,y):
    """
    Returns RMS height of profile y(x)
    Assumes x is evenly spaced
    """
    l = x[-1]-x[0]
    return np.sqrt(1./l*simps(y**2,x))/l
