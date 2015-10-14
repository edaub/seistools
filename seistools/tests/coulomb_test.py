from unittest import TestCase

import numpy as np
import seistools.coulomb

class TestCoulomb(TestCase):
    
    def test_tangent_2d(self):
        self.assertIs(type(seistools.coulomb.tangent_2d((1., 0.))), np.ndarray)
        np.testing.assert_array_almost_equal_nulp(seistools.coulomb.tangent_2d((1., 0.)), np.array([0., 1.]))
        np.testing.assert_array_almost_equal_nulp(seistools.coulomb.tangent_2d((0., 1.)), np.array([-1., 0.]))
        np.testing.assert_array_almost_equal_nulp(seistools.coulomb.tangent_2d((1./np.sqrt(2.), 1./np.sqrt(2.))),
                                       np.array([-1./np.sqrt(2.), 1./np.sqrt(2.)]))
        np.testing.assert_array_almost_equal_nulp(seistools.coulomb.tangent_2d((1./np.sqrt(2.), -1./np.sqrt(2.))),
                                       np.array([1./np.sqrt(2.), 1./np.sqrt(2.)]))
        self.assertRaises(AssertionError, seistools.coulomb.tangent_2d, (1.,))
        self.assertRaises(AssertionError, seistools.coulomb.tangent_2d, (1., 0., 0.))
        self.assertRaises(AssertionError, seistools.coulomb.tangent_2d, (0.5, 0.))
        
    def test_rotate_xy2nt_2d(self):
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.rotate_xy2nt_2d(1., 1., 0., (1., 0.))),
                                       np.array((1., 1.)))
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.rotate_xy2nt_2d(0., 1., 1., (0., 1.))),
                                       np.array((1., -1.)))
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.rotate_xy2nt_2d(-1., 0.5, -1.,
                                                                                  (1./np.sqrt(2.), 1./np.sqrt(2.)))),
                                       np.array((-0.5, 0.)))
        self.assertRaises(AssertionError, seistools.coulomb.rotate_xy2nt_2d, 1., 0., 0., (1., 0., 0.))
        self.assertRaises(AssertionError, seistools.coulomb.rotate_xy2nt_2d, 1., 0., 0., (1., 1.))
        self.assertRaises(AssertionError, seistools.coulomb.rotate_xy2nt_2d, 1., 0., 0., (1.,))


    def test_coulomb_2d(self):
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.coulomb_2d(1., 1., 0., (1., 0.), 0.5)),
                                       np.array(1.5))
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.coulomb_2d(0., 1., 1., (0., 1.), 0.2)),
                                       np.array((-0.8)))
        np.testing.assert_array_almost_equal_nulp(np.array(seistools.coulomb.coulomb_2d(-1., 0.5, -1.,
                                                                                  (1./np.sqrt(2.), 1./np.sqrt(2.)), 1.)), np.array(-0.5))
        self.assertRaises(AssertionError, seistools.coulomb.coulomb_2d, 1., 0., 0., (1., 0., 0.), 0.)
        self.assertRaises(AssertionError, seistools.coulomb.coulomb_2d, 1., 0., 0., (1., 1.), 0.)
        self.assertRaises(AssertionError, seistools.coulomb.coulomb_2d, 1., 0., 0., (1.,), 0.)
        self.assertRaises(AssertionError, seistools.coulomb.coulomb_2d, 1., 0., 0., (1., 0.), -1.)
