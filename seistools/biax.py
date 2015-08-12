import numpy as np

class biaxdata(object):
    "class representing biax data"

    def __init__(self, filename, ds_thresh = 0.1):
        "initialize data and calculate indices for stick-slip events"

        columns, all_dat = self._load_biax(filename)

        self.nt = len(all_dat[0,:])

        self.u = np.zeros(self.nt)
        self.s = np.zeros(self.nt)
        self.h = np.zeros(self.nt)
        self.sigma = np.zeros(self.nt)
        self.t = np.zeros(self.nt)
        self.mu = np.zeros(self.nt)
        self.ecu = np.zeros(self.nt)
        self.sstr = np.zeros(self.nt)
        self.acc = np.zeros(self.nt)

        for i in range(len(columns)):

            if columns[i] == "u":
                self.u = all_dat[i,:]
            elif columns[i] == "s":
                self.s = all_dat[i,:]
            elif columns[i] == "h":
                self.h = all_dat[i,:]
            elif columns[i] == "sigma":
                self.sigma = all_dat[i,:]
            elif columns[i] == "t":
                self.t = all_dat[i,:]
            elif columns[i] == "mu":
                self.mu = all_dat[i,:]
            elif columns[i] == "ecu":
                self.ecu = all_dat[i,:]
            elif columns[i] == "sstr":
                self.sstr = all_dat[i,:]
            elif columns[i] == "acc":
                self.acc = all_dat[i,:]

        self.nds, self.startds, self.ids = self._calc_stickslip(self.s, ds_thresh)

    def _load_biax(self, filename):
        "loads biax data, returning list of column names and full data array"

        f = open(filename, 'r')

        a = f.readline()

        columns = a.split()

        nlines = 0

        for line in f:
            nlines += 1

        ncol = len(columns)

        all_dat = np.empty((ncol, nlines))

        f.seek(0)

        f.readline()

        for i in range(nlines):
            line = f.readline()
            a = line.split()
            all_dat[:,i] = np.array([float(x) for x in a])

        return columns, all_dat

    def _calc_stickslip(self, s, s_thresh):
        "analyzes stick-slip data, returning a list containing indices of starts and ends of events"

        assert (s_thresh > 0.), "stress drop must be positive"

        nds = 0
        flag = False
        nt = len(s)

        ids = []
        startds = []

        for i in range(nt-1):
            if s[i+1] < s[i]:
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
                        ids.append(i)
                        startds.append(start)

        return nds, np.array(startds), np.array(ids)

    def get_nt(self):
        "returns number of time points"

        return self.nt

    def get_u(self):
        "returns load point displacement"

        return self.u

    def get_t(self):
        "returns time"

        return self.t

    def get_s(self):
        "returns shear stress"

        return self.s

    def get_h(self):
        "returns layer thickness"

        return self.h

    def get_sigma(self):
        "returns load point displacement"

        return self.sigma

    def get_mu(self):
        "returns friction coefficient"

        return self.mu

    def get_ecu(self):
        "returns corrected load point displacement"

        return self.ecu

    def get_sstr(self):
        "returns shear strain"

        return self.sstr

    def get_acc(self):
        "returns detector acceleration"

        return self.acc

    def get_nds(self):
        "returns number of stick-slip events"

        return self.nds

    def get_startds(self):
        "returns array of indices marking start of each stick-slip event"

        return self.startds

    def get_ids(self):
        "returns array of indices marking end of each stick-slip event"

        return self.ids

    def get_trecur(self):
        "calculates recurrence times for biax data"

        return np.diff(self.t[self.ids])

    def get_ds(self):
        "calculates stress drop for biax data"

        return self.s[self.startds] - self.s[self.ids]

    def get_dh(self):
        "calculates layer compaction (positive) for biax data"

        return self.h[self.startds] - self.h[self.ids]

    def get_dil(self, delay = 0):
        "calculates layer dilation for biax data"

        return self.h[self.startds[1:]] - self.h[self.ids[:-1]+delay]

    def get_stiff(self, npoints = 15, delay = 2):
        "calculates stiffness for each stick-slip event with indices in startds and ids, returns array of stiffnesses"

        stiff = []

        # determine stiffness

        for i in range(self.nds):
            start = self.ids[i]+delay
            utemp = self.u[start:start+npoints]
            stemp = self.s[start:start+npoints]
            stiff.append(np.polyfit(utemp,stemp,1)[0])

        return np.array(stiff)

    def get_g(self, npoints = 15, delay = 2):
        "calculates modulus for each stick-slip event (stiffness*thickness)"

        return self.get_stiff(npoints, delay)*(self.get_h()[self.get_ids()])

    def get_v0(self, npoints = 100):
        "calculates driving velocity from load point displacement, using a fit over npoints points"

        v0 = np.empty(self.nt)

        for i in range(self.nt):
            if (i < npoints//2):
                # average forward
                v0[i] = np.polyfit(self.t[:npoints], self.u[:npoints], 1)[0]
            elif (i > self.nt-1-npoints//2):
                # average backwards
                v0[i] = np.polyfit(self.t[-npoints:], self.u[-npoints:], 1)[0]
            else:
                # average from center
                if npoints%2 == 0:
                    v0[i] = np.polyfit(self.t[i-npoints//2:i+npoints//2], self.u[i-npoints//2:i+npoints//2], 1)[0]
                else:
                    v0[i] = np.polyfit(self.t[i-npoints//2:i+npoints//2+1], self.u[i-npoints//2:i+npoints//2+1], 1)[0]

        return v0

    def get_v(self, npoints = 100, npstiff = 15, delay = 2):
        "calculates slip velocity"

        stiff = self.get_stiff(npstiff, delay)
        v0 = self.get_v0(npoints)
        v = np.zeros(self.nt)
        kindex = 0

        for i in range(self.nt-1):
            if kindex == self.nds-1:
                pass
            elif i > self.ids[kindex+1]:
                kindex += 1
            v[i+1] = v0[i+1]-(self.s[i+1]-self.s[i])/(self.t[i+1]-self.t[i])/stiff[kindex]

        return v

    def get_slip(self, npoints = 100):
        "calculates slip by integrating slip velocity"

        v = self.get_v(npoints)
        slip = np.zeros(self.nt)

        for i in range(self.nt-1):
            slip[i+1] = slip[i]+0.5*(v[i+1]+v[i])*(self.t[i+1]-self.t[i])

        return slip

    def get_dhdu(self, npoints = 100, delay = 0):
        "calculates dilation rate with slip"

        slip = self.get_slip(npoints)
        dhdu = []

        for i in range(self.nds-1):
            npfit = self.startds[i+1]-self.ids[i]-delay
            print(npfit)
            dhdu.append(np.polyfit(slip[self.ids[i]+delay:self.ids[i]+delay+npfit], self.get_h()[self.ids[i]+delay:self.ids[i]+delay+npfit], 1)[0])

        return np.array(dhdu)

    def __str__(self):
        "returns string representation of biax data"

        return ("Bi-ax data with "+str(self.nt)+" data points\nt = "+str(self.t)+"\nu = "+str(self.u)+
                "\ns = "+str(self.s)+"\nh = "+str(self.h)+"\nsigma = "+str(self.sigma))
