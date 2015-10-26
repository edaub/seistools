import numpy as np

class rk_ls(object):
    "Class holding low storage RK integration variables, initialize with order (integer 1-4)"
    def __init__(self, order):
            assert(order == 1 or order == 2 or order == 3 or order == 4)
            if order == 1:
                    self.nstages = 1
                    self.A = np.array([0.])
                    self.B = np.array([1.])
                    self.C = np.array([0., 1.])
            elif order == 2:
                    self.nstages = 2
                    self.A = np.array([0., -1.])
                    self.B = np.array([1., 0.5])
                    self.C = np.array([0., 1., 1.])
            elif order == 3:
                    self.nstages = 3;
                    self.A = np.array([0., -5./9., -153./128.])
                    self.B = np.array([1./3., 15./16., 8./15.])
                    self.C = np.array([0., 1./3., 3./4., 1.])
            else:
                    self.nstages = 5
                    self.A = np.array([0., -567301805773./1357537059087.,
                                       -2404267990393./2016746695238.,-3550918686646./2091501179385.,
                                       -1275806237668./842570457699. ])
                    self.B = np.array([1432997174477./9575080441755.,
                                       5161836677717./13612068292357., 1720146321549./2090206949498.,
                                       3134564353537./4481467310338., 2277821191437./14882151754819.])
                    self.C = np.array([0., 1432997174477./9575080441755.,
                                       2526269341429./6820363962896., 2006345519317./3224310063776.,
                                       2802321613138./2924317926251., 1.])

    def get_nstages(self):
        "returns number of internal stages"
        return self.nstages

    def get_A(self, stage):
        "returns A coefficient for a given stage"
        assert(stage >= 0 and stage < self.nstages)
        return self.A[stage]

    def get_B(self, stage):
        "returns B coefficient for a given stage"
        assert(stage >= 0 and stage < self.nstages)
        return self.B[stage]

    def get_C(self, stage):
        "returns C coefficient for a given stage"
        assert(stage >= 0 and stage <= self.nstages)
        return self.C[stage]

    def __str__(self):
        return "RK Low Storage class with "+str(self.nstages)+" stages"

class rk54coeff(object):
    "class holding adaptive RK coefficients"
    def __init__(self):
        self.nstages = 6
        self.A = np.array([[0., 0., 0., 0., 0.], [0.2, 0., 0., 0., 0.], [3./40., 9./40, 0., 0., 0.],
                           [0.3, -0.9, 1.2, 0., 0.], [-11./54., 2.5, -70./27., 35./27., 0.],
                           [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
        self.B = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
        self.Berr = np.array([37./378.-2825./27648., 0., 250./621.-18575./48384.,
                              125./594.-13525./55296., -277./14336, 512./1771.-0.25])
        self.C = np.array([0., 0.2, 0.3, 0.6, 1., 0.875])

    def get_nstages(self):
        "returns number of stages"
        return self.nstages        

    def get_A(self, stage, step):
        "returns A coefficient for a given stage and step"
        assert(stage >= 0 and stage < self.nstages)
        assert(step >= 0 and step < self.nstages-1)
        return self.A[stage, step]

    def get_B(self, stage):
        "returns B coefficient for a given stage"
        assert(stage >= 0 and stage < self.nstages)
        return self.B[stage]

    def get_Berr(self, stage):
        "returns Berr coefficient for a given stage"
        assert(stage >= 0 and stage < self.nstages)
        return self.Berr[stage]

    def get_C(self, stage):
        "returns C coefficient for a given stage"
        assert(stage >= 0 and stage < self.nstages)
        return self.C[stage]

    def __str__(self):
        return "RK adaptive 5/4 coefficients"

def rk54_time_step(x, t, dt, xfunc, params, errnorm = None):
    """
    takes a 4th order Cash-Karp time step for a multi variable ODE
    x is a numpy array of current values
    t is current time
    dt is time step
    xfuncs is a function for updating values (returns array-like derivative of same length as x)
    params holds parameter values (passed to function)
    errnorm (optional) is a scalar or vector of values used to normalize error
    by default, errnorm uses the current values of x
    returns new values of variables and error estimate (unscaled)
    """
    assert(dt > 0.)
    assert(x.size == (xfunc(t, x, params)).size)

    if errnorm is None:
        errnorm = np.abs(x)+dt*np.abs(xfunc(t, x, params))+1.e-30

    rk = rk54coeff()

    kx = np.zeros((rk.get_nstages(),x.size))

    # integrate

    for i in range(rk.get_nstages()):
        xtemp = np.copy(x)
        for j in range(rk.get_nstages()-1):
            xtemp += rk.get_A(i, j)*kx[j,:]
        kx[i,:] = dt*xfunc(t+rk.get_C(i)*dt, xtemp, params)

    # calculate new values and error estimate

    xnew = np.copy(x)
    xerr = np.zeros(x.size)

    for i in range(rk.nstages):
        xnew += rk.get_B(i)*kx[i,:]
        xerr += rk.get_Berr(i)*kx[i,:]

    maxerr = np.amax(np.abs(xerr/errnorm))

    return xnew, maxerr


def rk54(xvals, xfunc, params, ttot, tol = 1.e-6, errnorm = None, maxsteps = 1000000, outstride = 1000000):
    """
    integrates a system of ODEs using the Cash-Karp 5/4 adaptive method
    xvals is initial values (array is flattened if it has a shape)
    xfunc is a function for variable derivatives
    params is a class holding parameters
    ttot is total integration time
    tol is desired tolerance (optional)
    maxsteps is maximum number of time steps (optional)
    returns time steps and array holding values at each time step
    """

    t = np.empty(maxsteps+1)
    x = np.empty((maxsteps+1, xvals.size))

    # set initial conditions

    t[0] = 0.
    x[0,:] = xvals

    dt = 1.e-6
    i = 0

    # integrate forward in time

    while (t[i] < ttot and i < maxsteps):
        xnew, error = rk54_time_step(x[i,:], t[i], dt, xfunc, params, errnorm)

        if (error > tol):
            # reject
            dtnew = 0.95*dt*np.abs(tol/error)**0.25
        else:
            # accept
            i += 1
            x[i,:] = xnew
            t[i] = t[i-1]+dt
            if error > 0.:
                dtnew = 0.95*dt*np.abs(tol/error)**0.2
            else:
                dtnew = 10.*dt

        if (dtnew/dt < 0.1):
            dtnew = 0.1*dt
        if (dtnew/dt > 10.):
            dtnew = 10.*dt

        dt = dtnew

        if (i%outstride == 0):
            print("Time (%) = "+str(t[i]/ttot)+", Step "+str(i))

    # return arrays and number of time steps

    if (t[i] < ttot):
        print("warning: reached maximum number of time steps")
        
    return i+1, t[:i+1], x[:i+1,:]
