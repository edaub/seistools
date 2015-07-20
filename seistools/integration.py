import numpy as np

class rkcoeff:
    "class holding adaptive RK coefficients"
    def __init__(self):
        self.nstages = 6
        self.a = np.array([0.2, 0.3, 0.6, 1., 0.875])
        self.b = np.array([[0., 0., 0., 0., 0.], [0.2, 0., 0., 0., 0.], [3./40., 9./40, 0., 0., 0.],
                           [0.3, -0.9, 1.2, 0., 0.], [-11./54., 2.5, -70./27., 35./27., 0.],
                           [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
        self.c = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
        self.cerr = np.array([37./378.-2825./27648., 0., 250./621.-18575./48384.,
                              125./594.-13525./55296., -277./14336, 512./1771.-0.25])

def rk_time_step(x, dt, xfunc, params):
    """
    takes a 4th order Cash-Karp time step for a multi variable ODE
    x is a numpy array of current values
    xfuncs is a function for updating values (returns array-like derivative)
    params holds parameter values (passed to function)
    returns new values of variables and error estimate
    """

    rk = rkcoeff()

    kx = np.zeros((rk.nstages,x.size))

    # integrate

    for i in range(rk.nstages):
        xtemp = x
        for j in range(rk.nstages-1):
            xtemp += rk.b[i, j]*kx[j,:]
        kx[i,j] = dt*xfunc(xtemp, params)

    # calculate new values and error estimate

    xnew = x
    xerr = np.zeros(x.size)

    for i in range(rk.nstages):
        xnew += rk.c[i]*kx[i,:]
        xerr += rk.cerr[i]*kx[i,:]

    maxerr = np.amax(np.abs(xerr/x))

    return xnew, maxerr


def rk54(xvals, xfunc, params, ttot, tol = 1.e-6, maxsteps = 1000000):
    """
    integrates a system of ODEs using the Cash-Karp 5/4 adaptive method
    xvals is initial values
    xfunc is a function for variable derivatives
    params is a class holding parameters
    ttot is total integration time
    tol is desired tolerance (optional)
    maxsteps is maximum number of time steps (optional)
    returns time steps and array holding values at each time step
    """

    # flatten array, will reshape again before returning

    xshape = xvals.shape

    t = np.empty(maxsteps)
    x = np.empty(maxsteps, xvals.size)

    # set initial conditions

    t[0] = 0.
    x[0,:] = xvals.flatten()

    dt = 1.e-6
    i = 0

    # integrate forward in time

    while (t[i] < ttot and i < maxsteps):
        xnew, error = rk_time_step(x[i,:], dt, xfuncs, params)

        if (error > tol):
            # reject
            dtnew = 0.95*dt*np.abs(tol/error)**0.25
        else:
            # accept
            i += 1
            x[i,:] = xnew
            t[i] = t[i-1]+dt
            dtnew = 0.95*dt*np.abs(tol/error)**0.2

        if (dtnew/dt < 0.1):
            dtnew = 0.1*dt
        if (dtnew/dt > 10.):
            dtnew = 10.*dt

        dt = dtnew

    # return arrays and number of time steps

    if (t[i] < ttot):
        print("warning: reached maximum number of time steps")
    return i, t[:i+1], x[:i+1,:]
