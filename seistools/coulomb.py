import numpy as np

def coulomb_2d(sxx, sxy, syy, n, mu, orientation=None):
    """
    Calcualtes coulomb failure criteria for a given normal direction and friction coefficient
    Inputs:
    stress components sxx, sxy, syy (negative in compression), can be arrays
    n is normal vector for receiver fault (must have length 2)
    mu (> 0) is friction coefficient
    orientation (optional) is a string indicating how you would like to compute the tangent vector
                    In all cases, the hypothetical second tangent vector is in the +z direction
                    "x" indicates that the tangent vector is defined by n x t1 = z (x is right handed cross product,
                          chosen so that faults with a purely +x normal will give a tangent in the +y direction).
                          This means that a positive Coulomb function leads to left lateral slip on the fault
                    "y" indicates that the tangent vector is defined by t1 x n = z (so that a normal in the +y
                          direction will give a tangent vector in the +x direction). This means that a positive
                          Coulomb function leads to right lateral slip on the fault
                    None indicates that the "y" convention is used (default)
    Returns:
    coulomb failure function
    """
    assert len(n) == 2, "normal vector must have length 2"
    assert np.isclose(np.sqrt(n[0]**2+n[1]**2),1.), "normal vector must be normalized"
    assert mu >= 0., "mu must be positive"
    assert (orientation == "x" or orientation == "y" or orientation == None)
    
    sn, st = rotate_xy2nt_2d(sxx, sxy, syy, n, orientation)

    return st+mu*sn

def rotate_xy2nt_2d(sxx, sxy, syy, n, orientation=None):
    """
    Rotates stress components from xy to normal/tangential to given normal vector
    Inputs:
    stress components sxx, sxy, syy (negative in compression), can be arrays
    n is normal vector for receiver fault (must have length 2)
    Returns:
    normal and shear stress in rotated coordinates
    """
    assert len(n) == 2, "normal vector must have length 2"
    assert np.isclose(np.sqrt(n[0]**2+n[1]**2),1.), "normal vector must be normalized"
    assert (orientation == "x" or orientation == "y" or orientation == None)
    
    m = tangent_2d(n)

    sn = n[0]**2*sxx+2.*n[0]*n[1]*sxy+n[1]**2*syy
    st = n[0]*m[0]*sxx+(m[0]*n[1]+n[0]*m[1])*sxy+n[1]*m[1]*syy

    return sn, st


def tangent_2d(n, orientation=None):
    "Creates returns vector orthogonal to input vector (of length 2)"
    assert len(n) == 2, "normal vector must be of length 2"
    assert np.isclose(np.sqrt(n[0]**2+n[1]**2),1.), "normal vector must be normalized"
    assert (orientation == "x" or orientation == "y" or orientation == None)
    
    m = np.empty(2)
    
    if orientation == "x":
        m[1] = n[0]/np.sqrt(n[0]**2+n[1]**2)
        m[0] = -n[1]/np.sqrt(n[0]**2+n[1]**2)
    else:
        m[0] = n[1]/np.sqrt(n[0]**2+n[1]**2)
        m[1] = -n[0]/np.sqrt(n[0]**2+n[1]**2)

    return m
