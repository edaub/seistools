import numpy as np
from datetime import datetime
import re

class catalog(object):
    "Class representing a seismic catalog"
    def __init__(self, *args):
        """
        Initialize catalog -- can be done in two different ways:
        (1) pass filename containing text file formatted with
        time (as datetime64) latitude longitude depth magnitude
        (2) nevents, plus arrays holding time (as datetime64), latitude, longitude, depth, magnitude
        """
        def _read_catalog(filename):
            """
            reads catalog from formatted text file
            format is time (datetime64) latitude longitude depth magnitude
            returns number of events and arrays for each data type
            """
            f = open(filename, 'r')

            nevents = 0

            for line in f:
                nevents += 1

            time = np.empty(nevents, dtype='datetime64[ms]')
            lat = np.empty(nevents)
            lon = np.empty(nevents)
            depth = np.empty(nevents)
            mag = np.empty(nevents)

            f.seek(0)

            for i in range(nevents):
                event = f.readline()
                event = event.split()
                time[i] = event[0]
                lat[i] = float(event[1])
                lon[i] = float(event[2])
                depth[i] = float(event[3])
                mag[i] = float(event[4])

            f.close()

            return nevents, time, lat, lon, depth, mag

        # get attributes from file or as read in
        if len(args) == 1:
            nevents, time, lat, lon, depth, mag = _read_catalog(args[0])
        else:
            nevents, time, lat, lon, depth, mag = args
        
        assert len(time) == nevents
        assert len(mag) == nevents
        assert len(lat) == nevents
        assert len(lon) == nevents
        assert len(depth) == nevents
        self.nevents = nevents
        self.time = np.array(time, dtype='datetime64')
        self.mag = np.array(mag)
        self.lat = np.array(lat)
        self.lon = np.array(lon)
        self.depth = np.array(depth)
        self.declustered = False
        self.aftershock = np.zeros(nevents, dtype = int)

    def get_nevents(self):
        "Returns number of events"
        return self.nevents

    def get_time(self, index = None):
        "Returns event times (can optionally include index)"
        if index is None:
            return self.time
        else:
            return self.time[index]

    def get_mag(self, index = None):
        "Returns event magnitudes (can optionally include index)"
        if index is None:
            return self.mag
        else:
            return self.mag[index]

    def get_lat(self, index = None):
        "Returns event latitudes (can optionally include index)"
        if index is None:
            return self.lat
        else:
            return self.lat[index]

    def get_lon(self, index = None):
        "Returns event longitudes (can optionally include index)"
        if index is None:
            return self.lon
        else:
            return self.lon[index]

    def get_depth(self, index = None):
        "Returns event depths (can optionally include index)"
        if index is None:
            return self.depth
        else:
            return self.depth[index]

    def get_mainshocks(self):
        "returns array indicating which events are mainshocks"
        if self.declustered == False:
            print("Must decluster catalog to get mainshocks")
        else:
            return -self.aftershock.astype(bool)

    def get_aftershocks(self):
        "returns array indicating which events are aftershocks"
        if self.declustered == False:
            print("Must decluster catalog to get aftershocks")
        else:
            return self.aftershock.astype(bool)

    def decluster(self, timefunc = None, distfunc = None):
        """
        Decluster catalog using Gardner and Knopoff method
        Can supply functions for time (in days) and distance (in km) as a function of magnitude, otherwise
        uses G&K default values for Southern California
        """

        def latlondist(lat1, lon1, depth1, lat2, lon2, depth2):
            if (np.sin(lat1*np.pi/180.)*np.sin(lat2*np.pi/180.)+np.cos(lat1*np.pi/180.)*np.cos(lat2*np.pi/180.)*np.cos((lon1-lon2)*np.pi/180.)>1.):
                dist = np.abs(depth1-depth2)
            else:
                alpha = np.arccos(np.sin(lat1*np.pi/180.)*np.sin(lat2*np.pi/180.)+np.cos(lat1*np.pi/180.)*np.cos(lat2*np.pi/180.)*np.cos((lon1-lon2)*np.pi/180.))
                dist = np.sqrt(((6371.-depth1)-(6371.-depth2)*np.cos(alpha))**2.+((6371.-depth2)*np.sin(alpha))**2.)
            dist = np.arccos(np.sin(lat1*np.pi/180.)*np.sin(lat2*np.pi/180.)+np.cos(lat1*np.pi/180.)*np.cos(lat2*np.pi/180.)*np.cos((lon1-lon2)*np.pi/180.))*6371.
            return dist

        if timefunc is None:
            timefunc = lambda x: 10.**(0.5505*x-0.5647)
        if distfunc is None:
            distfunc = lambda x: 10.**(0.1274*x+0.9692)

        for i in range(0,self.nevents-1):
            for j in range(i+1,self.nevents):
                dist = latlondist(self.lat[i], self.lon[i], self.depth[i], self.lat[j], self.lon[j], self.depth[j])
                dt = (self.time[j]-self.time[i]).astype(float)/1000./3600./24.
                if (dt < timefunc(self.mag[i]) and dist < distfunc(self.mag[i])):
                    if (self.aftershock[i] == 0):
                        self.aftershock[j] = i+1
                    else:
                        self.aftershock[j] = self.aftershock[i]

        declustind = np.bincount(self.aftershock)
        declustind[0] = 0
        clusters = np.nonzero(declustind)[0]

        for i in range(0,len(clusters)):
            clustind = np.nonzero(self.aftershock == clusters[i])[0]
            clustind = np.insert(clustind,0,clusters[i]-1)
            self.aftershock[clustind] = 0
            clustind = np.delete(clustind,np.argmax(self.mag[clustind]))
            self.aftershock[clustind] = 1

        print('Declustering Information:\nTotal number of events = '+str(self.nevents)+
              '\nNumber of main shocks = '+str(self.nevents-np.sum(self.aftershock))+
              '\nNumber of aftershocks = '+str(np.sum(self.aftershock)))
        self.declustered = True

    def get_subcatalog(self, indices):
        "Returns a subcatalog given a list of indices"
        if len(indices) == self.nevents:
            nevents = np.sum(indices)
        else:
            assert min(indices) >= 0
            assert max(indices) < self.nevents
            nevents = len(indices)
        return catalog(nevents, self.get_time(indices), self.get_lat(indices),
                             self.get_lon(indices), self.get_depth(indices), self.get_mag(indices))
    
    def __str__(self):
        "Returns a string representation of a catalog"
        return ("Catalog with "+str(self.nevents)+" events\nTime = "+str(self.get_time())
                    +"\nMagnitude = "+str(self.get_mag())+"\nLatitude = "+str(self.get_lat())
                    +"\nLongitude = "+str(self.get_lon())+"\nDepth = "+str(self.get_depth()))

    
def calc_b(mag, complete = None):
    "returns b-value calculated via the maximum entropy method"

    if complete is None:
        mag_copy = mag[:]
    else:
        mag_copy = mag[mag >= complete]

    mavg = np.mean(mag_copy)
    mc = np.min(mag_copy)-0.05
    return 1./(mavg-mc)/np.log(10)

def mc_maxcurv(mag):
    """
    determines completeness magnitude with maximum curvature method
    returns completeness magnitude and b-value
    """

    mc = mode(mag)[0][0]
    b = calc_b(mag, mc)

    return mc, b

def delta_b(mag, complete = None):
    "returns uncertainty in b-value"

    if complete is None:
        mag_copy = mag[:]
    else:
        mag_copy = mag[mag >= complete]
    
    b = calc_b(mag)
    return 2.3*b**2*np.std(mag)/np.sqrt(float(len(mag)-1))

def mc_bvalstab(mag):
    """
    determines completeness magnitude using b-value stability method
    returns completeness magnitude and b-value
    """

    mc = min(mag)-0.1
    bavg = 1.5
    bval = 1.
    db = 0.

    while np.abs(bavg-bval) > db:
        mc += 0.1
        bval = calc_b(mag, mc)
        db = delta_b(mag, mc)
        bavg = 0.
        for j in range(6):
            mctemp = mc+float(j)*0.1
            bavg += calc_b(mag, mctemp)/6.

    return mc, b
