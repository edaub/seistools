import numpy as np
import re

class sourcemodel(object):
    "class representing finite fault model for an earthquake"
    def __init__(self, filename):
        """
        Reads data from an fsp file
        (see Martin Mai's database online at equake-rc.info/SRCMOD/fileformats/)
        """
        f = open(filename, 'r')

        for line in f:

            matchobjnx1 = re.search(r'Nx += +', line)
            matchobjnz1 = re.search(r'Nz += +', line)
            matchobjnx2 = re.search(r'Nx += +[0-9]+', line)
            matchobjnz2 = re.search(r'Nz += +[0-9]+', line)

            if matchobjnx2 and matchobjnz2:
                self.nx = int(line[matchobjnx1.end():matchobjnx2.end()])
                self.nz = int(line[matchobjnz1.end():matchobjnz2.end()])

            matchobjdx1 = re.search(r'Dx += +', line)
            matchobjdz1 = re.search(r'Dz += +', line)
            matchobjdx2 = re.search(r'Dx += +[0-9]+\.[0-9]+', line)
            matchobjdz2 = re.search(r'Dz += +[0-9]+\.[0-9]+', line)

            if matchobjdx2 and matchobjdz2:
                self.dx = float(line[matchobjdx1.end():matchobjdx2.end()])
                self.dz = float(line[matchobjdz1.end():matchobjdz2.end()])

            matchobjns1 = re.search(r'Nsbfs += +', line)
            matchobjns2 = re.search(r'Nsbfs += +[0-9]+', line)

            if matchobjns2:
                self.nsubfaults = int(line[matchobjns1.end():matchobjns2.end()])
                assert (self.nsubfaults == self.nx*self.nz), "number of subfaults does not match"
                break

        try:
            self.nsubfaults
            self.nx
            self.nz
            self.dx
            self.dz
        except NameError:
            raise NameError("Error loading inversion information from file")

        self.lat = np.empty((self.nx, self.nz))
        self.lon = np.empty((self.nx, self.nz))
        self.x = np.empty((self.nx, self.nz))
        self.y = np.empty((self.nx, self.nz))
        self.depth = np.empty((self.nx, self.nz))
        self.slip = np.empty((self.nx, self.nz))
        self.rake = np.empty((self.nx, self.nz))
        self.rise = np.empty((self.nx, self.nz))
        self.trup = np.empty((self.nx, self.nz))

        for line in f:
            if re.search(r'LAT +LON +X==EW', line):
                break

        f.readline()

        for i in range(self.nz):
            for j in range(self.nx):
                a = f.readline()
                a = a.split()
                self.lat[j,i] = float(a[0])
                self.lon[j,i] = float(a[1])
                self.x[j,i] = float(a[2])
                self.y[j,i] = float(a[3])
                self.depth[j,i] = float(a[4])
                self.slip[j,i] = float(a[5])
                try:
                    self.rake[j,i] = float(a[6])
                except IndexError:
                    if (i == 0 and j == 0):
                        print("No rake information")
                    elif (i == self.nz-1 and j == self.nx-1):
                        self.rake = None
                try:
                    self.rise[j,i] = float(a[7])
                except IndexError:
                    if (i == 0 and j == 0):
                        print("No rise time information")
                    elif (i == self.nz-1 and j == self.nx-1):
                        self.rise = None
                try:
                    self.trup[j,i] = float(a[8])
                except IndexError:
                    if (i == 0 and j == 0):
                        print("No rupture time information")
                    elif (i == self.nz-1 and j == self.nx-1):
                        self.trup = None

        f.close()

        xm = -np.sqrt(self.x[0,:]**2+self.y[0,:]**2)
        xp = np.sqrt(self.x[-1,:]**2+self.y[-1,:]**2)
        self.strike = np.transpose(np.array([np.linspace(dist1,dist2,self.nx) for (dist1, dist2) in zip(xm, xp)]))


    def get_nx(self):
        "Returns number of subfaults in x direction"
        return self.nx

    def get_nz(self):
        "Returns number of subfaults in z direction"
        return self.nz

    def get_dx(self):
        "Returns discretization in x direction"
        return self.dx

    def get_dz(self):
        "Returns discretization in z direction"
        return self.dz

    def get_nsubfaults(self):
        "Returns total number of subfaults"
        return self.nsubfaults

    def get_slip(self, index = None):
        "Returns slip array, index is optional tuple of indices"
        if index is None:
            return self.slip
        else:
            return self.slip[index]

    def get_rake(self, index = None):
        "Returns rake array, index is optional tuple of indices"
        if self.rake is None:
            return None
        elif index is None:
            return self.rake
        else:
            return self.rake[index]

    def get_lat(self, index = None):
        "Returns latitude array, index is optional tuple of indices"
        if index is None:
            return self.lat
        else:
            return self.lat[index]

    def get_lon(self, index = None):
        "Returns longitude array, index is optional tuple of indices"
        if index is None:
            return self.lon
        else:
            return self.lon[index]

    def get_x(self, index = None):
        "Returns x array, index is optional tuple of indices"
        if index is None:
            return self.x
        else:
            return self.x[index]

    def get_y(self, index = None):
        "Returns y array, index is optional tuple of indices"
        if index is None:
            return self.y
        else:
            return self.y[index]

    def get_depth(self, index = None):
        "Returns depth array, index is optional tuple of indices"
        if index is None:
            return self.depth
        else:
            return self.depth[index]

    def get_strike(self, index = None):
        "Returns strike array, index is optional tuple of indices"
        if index is None:
            return self.strike
        else:
            return self.strike[index]

    def get_trup(self, index = None):
        "Returns rupture time array, index is optional tuple of indices"
        if self.trup is None:
            return None
        elif index is None:
            return self.trup
        else:
            return self.trup[index]

    def get_rise(self, index = None):
        "Returns rise time array, index is optional tuple of indices"
        if self.rise is None:
            return None
        elif index is None:
            return self.rise
        else:
            return self.rise[index]

    def get_slip_x(self, index = None):
        "Returns horizontal slip component array, index is optional tuple of indices"
        if self.rake is None:
            return None
        elif index is None:
            return -self.slip*np.cos(np.pi/180.*self.rake)
        else:
            return -self.slip[index]*np.cos(np.pi/180.*self.rake[index])

    def get_slip_z(self, index = None):
        "Returns vertical slip component array, index is optional tuple of indices"
        if self.rake is None:
            return None
        elif index is None:
            return self.slip*np.sin(np.pi/180.*self.rake)
        else:
            return self.slip[index]*np.sin(np.pi/180.*self.rake[index])

    def find_strike(self, lat, lon, depth):
        "Converts latitude, longitude, and depth into strike coordinates using linear interpolation"

        return ((lat-self.get_lat((0,0)))/(self.get_lat((-1,0))-self.get_lat((0,0)))*
             (self.get_strike((-1,0))-self.get_strike((0,0))))+self.get_strike((0,0))
