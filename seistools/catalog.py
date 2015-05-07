import numpy as np
from datetime import datetime

class catalog(object):
    "Class representing a seismic catalog"
    def __init__(self, nevents, time, mag, lat, lon, depth):
        "Initialize catalog with arrays holding time, magnitude, latitude, longitude, and depth"
        assert len(time) == nevents
        assert len(mag) == nevents
        assert len(lat) == nevents
        assert len(lon) == nevents
        assert len(depth) == nevents
        self.nevents = nevents
        self.time = time
        self.mag = mag
        self.lat = lat
        self.lon = lon
        self.depth = depth

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

    def get_subcatalog(self, indices):
        "Returns a subcatalog given a list of indices"
        assert min(indices) >= 0
        assert max(indices) < self.nevents
        return catalog(self, len(indices), self.get_time(indices), self.get_mag(indices),
                       self.get_lat(indices), self.get_lon(indices), self.get_depth(indices))
    
    def __str__(self):
        "Returns a string representation of a catalog"
        return ("Catalog with "+str(self.nevents)+" events\nTime = "+str(self.get_time())
                    +"\nMagnitude = "+str(self.get_mag())+"\nLatitude = "+str(self.get_lat())
                    +"\nLongitude = "+str(self.get_lon())+"\nDepth = "+str(self.get_depth()))
