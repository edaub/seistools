import numpy as np

def load_biax(filename):
    "loads data, with data columns specified via list columns"

    f = open(filename, 'r')

    nlines = 0

    for line in f:
        nlines += 1

    f.seek(0)

    a = f.readline()
    a = a.split()

    ncol = len(a)

    all_dat = np.empty((ncol, nlines))

    f.seek(0)

    for i in range(nlines):
        line = f.readline()
        a = line.split()
        all_dat[:,i] = np.array([float(x) for x in a])

    dataout = []

    for i in range(ncol):
        dataout.append(all_dat[i,:])

    return dataout

def ss_analysis(u, t, h, s, s_thresh):
    "analyzes stick-slip data, returning a list containing various calculated quantities"

    assert (len(u) == len(t)), "vectors must have the same length"
    assert (len(u) == len(h)), "vectors must have the same length"
    assert (len(u) == len(s)), "vectors must have the same length"
    assert (s_thresh > 0.), "stress drop must be positive"

    v0 = np.polyfit(t, u, 1)[0]

    nds = 0
    flag = False
    nt = len(u)

    ds = []
    dh = []
    dil = []
    ids = []
    startds = []
    stiff = []
    g = []
    dhdt = []

    for i in range(nt-1):
        if s(i+1) < s(i):
            # stress is dropping
            if not flag:
                flag = True
                start = i
        else:
            # stress is increasing
            if flag:
                flag = False
                if s[start]-s[i] > s_thresh:
                    nds += 1
                    dh.append(h[start]-h[i])
                    ids.append(i)
                    startds.append(start)
                    ds.append(s[start]-s[i])

    # determine stiffness

    npoints = 15
    delay = 2

    for i in range(nds):
        start = ids[i]+delay
        utemp = u[start:start+npoints]
        stemp = s[start:start+npoints]
        stiff.append(np.polyfit(utemp,stemp,1)[0])

    v = np.empty(nt)
    slip = np.empty(nt)
    kindex = 0

    for i in range(nt-1):
        if kindex == nds:
            pass
        elif i > ids[kindex+1]:
            kindex += 1
        v[i+1] = v0-(s[i+1]-s[i])/(t[i+1]-t[i])/stiff[kindex]
        slip[i+1] = slip[i]+0.5*(v[i+1]+v[i])/(t[i+1]-t[i])

    # determine dilation rate with slip

    for 
    return nds, ds, dh, stiff, ids, startds
