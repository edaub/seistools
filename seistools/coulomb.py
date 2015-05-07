import numpy as np

def coulomb_2d(sxx, sxy, syy, n, mu):
    """
    Calcualtes coulomb failure criteria for a given normal and friction coefficient
    Inputs:
    stress components sxx, sxy, syy (negative in compression), can be arrays
    n is normal vector for receiver fault (must have length 2), points towards
    mu (> 0) is friction coefficient
    Returns:
    coulomb failure function
    """
    assert mu > 0., "mu must be positive"
    
    sn, st = rotate_xy2nt_2d(sxx, sxy, syy, n)

    return st+mu*sn

def rotate_xy2nt_2d(sxx, sxy, syy, n):
    """
    Rotates stress components from xy to normal/tangential to given normal vector
    Inputs:
    stress components sxx, sxy, syy (negative in compression), can be arrays
    n is normal vector for receiver fault (must have length 2)
    Returns:
    normal and shear stress in rotated coordinates
    """
    m = tangent_2d(n)

    sn = n[0]**2*sxx+2.*n[0]*n[1]*sxy+n[1]**2*syy
    st = n[0]*m[0]*sxx+(m[0]*n[1]+n[0]*m[1])*sxy+n[1]*m[1]*syy

    return sn, st


def tangent_2d(n):
    "Creates returns vector orthogonal to input vector (of length 2)"
    assert len(n) == 2, "normal vector must be of length 2"
    assert np.abs(np.sqrt(n[0]**2+n[1]**2)-1.) < 1.e-14, "normal vector must be normalized"
    
    m = np.empty(2)
    if np.abs(n[0]) > np.abs(n[1]):
        m[1] = np.sign(n[0])/np.sqrt(1.+(n[1]/n[0])**2)
        m[0] = -n[1]/n[0]*m[1]
    else:
        m[0] = -np.sign(n[1])/np.sqrt(1.+(n[0]/n[1])**2)
        m[1] = -n[0]/n[1]*m[0]

    return m
