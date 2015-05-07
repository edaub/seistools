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
            matchobj = re.search(r'Nsbfs = [0-9]+ subfaults', line)

            if matchobj:
                self.nsubfaults = int(line[matchobj.start()+8:matchobj.end()-10])
                break

        self.lat = np.empty(self.nsubfaults)
        self.lon = np.empty(self.nsubfaults)
        self.x = np.empty(self.nsubfaults)
        self.y = np.empty(self.nsubfaults)
        self.depth = np.empty(self.nsubfaults)
        self.slip = np.empty(self.nsubfaults)
        self.rake = np.empty(self.nsubfaults)
        self.rise = np.empty(self.nsubfaults)
        self.trup = np.empty(self.nsubfaults)

        for line in f:
            if re.search(r'LAT +LON +X==EW', line):
                break

        for i in range(2):
            f.readline()

        for i in range(self.nsubfaults):
            a = f.readline()
            a = a.split()
            self.lat[i] = float(a[0])
            self.lon[i] = float(a[1])
            self.x[i] = float(a[2])
            self.y[i] = float(a[3])
            self.depth[i] = float(a[4])
            self.slip[i] = float(a[5])
            self.rake[i] = float(a[6])
            self.rise[i] = float(a[7])
            self.trup[i] = float(a[8])

        f.close()
