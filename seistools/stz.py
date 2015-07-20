import numpy as np
import seistools.stress as stress
import seistools.integration as integration

class stzparams:
    "class holding parameters for STZ equations"
    def __init__(self, epsilon = 10, t0 = 1.e-13, E0 = 2.424, T = 293., V = 1.98e-28, sy = 1.,
                 c0 = 2.e4, R = 1.e6, beta = 0.9, E1 = 0.16, chiw = 1.5, chi0 = 0.07, chi1 = 1.e-1,
                 q0 = 1.e-4, k = 0.00015, v0 = 1.e-3, mu = 32., cs = 3.464, poisson = 0., dx = 0.01,
                 nload = 0, vf = False):
        self.epsilon = epsilon
        self.t0 = t0
        self.E0 = E0
        self.T = T
        self.V = V
        self.sy = sy
        self.c0 = c0
        self.R = R
        self.beta = beta
        self.E1 = E1
        self.chiw = chiw
        self.chi0 = chi0
        self.chi1 = chi1
        self.q0 = q0
        self.k = k
        self.v0 = v0
        self.mu = mu
        self.cs = cs
        self.coeff_v = 0.5*self.mu/self.cs
        self.poisson = poisson
        self.dx = dx
        self.nload = nload
        self.vf = vf

def chihat(v, params):
    "calculate chihat for a given strain rate v"
    if (v > params.q0/params.t0):
        raise ValueError("strain rate exceeds threshhold")
    if (v == 0.):
        return params.chi0
    chihat1 = params.chiw/np.log(params.q0/params.t0/v)
    chihat2 = params.chi0+(params.chiw+params.chi1)/np.log(params.q0/params.t0/v)
    chihat3 = 4.*params.chiw*params.chi0/np.log(params.q0/params.t0/v)
    if params.vf:
        return 0.5*(chihat2+np.sqrt(chihat2**2-chihat3))
    else:
        return chihat1

def vpl(s, chi, params):
    "calculates plastic strain rate given parameters"
    if s < params.sy:
        return 0.
    else:
        return params.epsilon/params.t0*np.exp(-(params.E0/7.69e-5-s*params.V*8.13e31)/params.T-1./chi)*(1.-params.sy/s)

def dvplds(v, s, chi, params):
    "derivative of plastic strain rate wrt s"
    return vpl(s, chi, params)*params.V*8.13e31/params.T+params.sy/s/(s-params.sy)

def dvpldchi(v, s, chi, params):
    "derivative of plastic strain rate wrt chi"
    return vpl(s, chi, params)/chi/chi

def dchi(v, s, chi, params):
    "calculates time derivative of effective temperature given v, s, chi, and parameter values"
    return (v*s/params.c0/params.sy*(1.-chi/chihat(v, params))-params.R*np.exp(-params.beta/chi-params.E1/7.69e-5/params.T))

def dchidt(s, chi, params):
    "calculates time derivative of effective temperature given s, chi, and parameters"
    return dchi(vpl(s, chi, params), s, chi, params)

def dsdt_2d(s, chi, f, params):
    "derivative of stress at a given point given stress, chi, stress change (from convolution), and parameters"
    return ((f-params.coeff_v*dvpldchi(vpl(s, chi, params), s, chi, params)*dchidt(s, chi, params))/
            (1.+params.coeff_v*dvplds(vpl(s, chi, params), s, chi, params)))

def stz_2d_der(x, params):
    "wrapper function for calculating derivatives for adaptive rk integration of 2d continuum stz model"

    # calculate stressing term

    xsize = len(x)//2

    dx = np.empty(len(x))

    v = np.empty(xsize)
    s = x[:xsize]
    chi = x[xsize:]

    for i in range(xsize):
        v[i] = vpl(s[i], chi[i], params)

    # set loading points to have v0

    v[0:nload] = params.v0
    v[-nload:-1] = params.v0

    f = stress.calc_ds_2d(v, params.dx, params.mu, params.poisson)

    for i in range(xsize):
        dx[i] = dsdt_2d(s[i], chi[i], f[i], params)
        dx[xsize+i] = dchidt(s[i], chi[i], params)

    return dx

def integrate_stz_2d(u0, chi0, params, ttot, tol = 1.e-6, maxsteps = 1000000):
    """
    integrate STZ equations coupled to a 2d quasidynamic elastic continuum
    s0 (array) is initial slip
    chi0 (array) is initial effective temperature
    params is class holding STZ parameter values
    ttot is total integration time
    tol (optional) is tolerance level for adaptive RK method
    maxsteps (optional) is maximum number of steps
    returns number of time steps, time, stress, and effective temperature arrays
    """

    # set up array with initial conditions

    x = np.empty(2, u0.size)

    x[:u0.size] = stress.calc_ds_2d(u0, params.dx, params.mu, params.poisson)
    x[u0.size:] = chi0

    nt, t, x = integration.rk54(x, stz_2d_der, params, ttot, tol, maxsteps)

    return nt, t, x[:u0.size], x[u0.size:]
    
