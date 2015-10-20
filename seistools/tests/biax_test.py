from unittest import TestCase
from tempfile import NamedTemporaryFile

import numpy as np
from seistools.biax import biaxdata

class TestBiax(TestCase):

    def setUp(self):
        u = np.linspace(0., 100., 1000)
        
        fp = NamedTemporaryFile()
    
    def test_biax(self):
        pass
        
