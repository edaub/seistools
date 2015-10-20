from unittest import TestCase

import numpy as np
import seistools.integration

class test_rk_ls(TestCase):

    def setUp(self):
        self.rkls1 = seistools.integration.rk_ls(1)
        self.rkls2 = seistools.integration.rk_ls(2)
        self.rkls3 = seistools.integration.rk_ls(3)
        self.rkls4 = seistools.integration.rk_ls(4)

    def tearDown(self):
        self.rkls1 = None
        self.rkls2 = None
        self.rkls3 = None
        self.rkls4 = None

    def test_init(self):
        self.assertRaises(AssertionError, seistools.integration.rk_ls, 0)
        self.assertRaises(AssertionError, seistools.integration.rk_ls, -1)
        self.assertRaises(AssertionError, seistools.integration.rk_ls, 1.5)
        self.assertRaises(AssertionError, seistools.integration.rk_ls, 10000)

    def test_get_nstages(self):
        self.assertEqual(self.rkls1.get_nstages(), 1)
        self.assertEqual(self.rkls2.get_nstages(), 2)
        self.assertEqual(self.rkls3.get_nstages(), 3)
        self.assertEqual(self.rkls4.get_nstages(), 5)

    def test_get_A(self):
        self.assertEqual(self.rkls1.get_A(0), 0.)
        self.assertEqual(self.rkls2.get_A(0), 0.)
        self.assertEqual(self.rkls2.get_A(1), -1.)
        self.assertEqual(self.rkls3.get_A(0), 0.)
        self.assertEqual(self.rkls3.get_A(1), -5./9.)
        self.assertEqual(self.rkls3.get_A(2), -153./128.)
        self.assertEqual(self.rkls4.get_A(0), 0.)
        self.assertEqual(self.rkls4.get_A(1), -567301805773./1357537059087.)
        self.assertEqual(self.rkls4.get_A(2), -2404267990393./2016746695238.)
        self.assertEqual(self.rkls4.get_A(3), -3550918686646./2091501179385.)
        self.assertEqual(self.rkls4.get_A(4), -1275806237668./842570457699.)
        self.assertRaises(AssertionError, self.rkls1.get_A, -1)
        self.assertRaises(AssertionError, self.rkls1.get_A, 1)
        self.assertRaises(AssertionError, self.rkls2.get_A, -1)
        self.assertRaises(AssertionError, self.rkls2.get_A, 2)
        self.assertRaises(AssertionError, self.rkls3.get_A, -1)
        self.assertRaises(AssertionError, self.rkls3.get_A, 3)
        self.assertRaises(AssertionError, self.rkls4.get_A, -1)
        self.assertRaises(AssertionError, self.rkls4.get_A, 5)

    def test_get_B(self):
        self.assertEqual(self.rkls1.get_B(0), 1.)
        self.assertEqual(self.rkls2.get_B(0), 1.)
        self.assertEqual(self.rkls2.get_B(1), 0.5)
        self.assertEqual(self.rkls3.get_B(0), 1./3.)
        self.assertEqual(self.rkls3.get_B(1), 15./16.)
        self.assertEqual(self.rkls3.get_B(2), 8./15.)
        self.assertEqual(self.rkls4.get_B(0), 1432997174477./9575080441755.)
        self.assertEqual(self.rkls4.get_B(1), 5161836677717./13612068292357.)
        self.assertEqual(self.rkls4.get_B(2), 1720146321549./2090206949498.)
        self.assertEqual(self.rkls4.get_B(3), 3134564353537./4481467310338.)
        self.assertEqual(self.rkls4.get_B(4), 2277821191437./14882151754819.)
        self.assertRaises(AssertionError, self.rkls1.get_B, -1)
        self.assertRaises(AssertionError, self.rkls1.get_B, 1)
        self.assertRaises(AssertionError, self.rkls2.get_B, -1)
        self.assertRaises(AssertionError, self.rkls2.get_B, 2)
        self.assertRaises(AssertionError, self.rkls3.get_B, -1)
        self.assertRaises(AssertionError, self.rkls3.get_B, 3)
        self.assertRaises(AssertionError, self.rkls4.get_B, -1)
        self.assertRaises(AssertionError, self.rkls4.get_B, 5)

    def test_get_C(self):
        self.assertEqual(self.rkls1.get_C(0), 0.)
        self.assertEqual(self.rkls1.get_C(1), 1.)
        self.assertEqual(self.rkls2.get_C(0), 0.)
        self.assertEqual(self.rkls2.get_C(1), 1.)
        self.assertEqual(self.rkls2.get_C(2), 1.)
        self.assertEqual(self.rkls3.get_C(0), 0.)
        self.assertEqual(self.rkls3.get_C(1), 1./3.)
        self.assertEqual(self.rkls3.get_C(2), 3./4.)
        self.assertEqual(self.rkls3.get_C(3), 1.)
        self.assertEqual(self.rkls4.get_C(0), 0.)
        self.assertEqual(self.rkls4.get_C(1), 1432997174477./9575080441755.)
        self.assertEqual(self.rkls4.get_C(2), 2526269341429./6820363962896.)
        self.assertEqual(self.rkls4.get_C(3), 2006345519317./3224310063776.)
        self.assertEqual(self.rkls4.get_C(4), 2802321613138./2924317926251.)
        self.assertEqual(self.rkls4.get_C(5), 1.)
        self.assertRaises(AssertionError, self.rkls1.get_C, -1)
        self.assertRaises(AssertionError, self.rkls1.get_C, 2)
        self.assertRaises(AssertionError, self.rkls2.get_C, -1)
        self.assertRaises(AssertionError, self.rkls2.get_C, 3)
        self.assertRaises(AssertionError, self.rkls3.get_C, -1)
        self.assertRaises(AssertionError, self.rkls3.get_C, 4)
        self.assertRaises(AssertionError, self.rkls4.get_C, -1)
        self.assertRaises(AssertionError, self.rkls4.get_C, 6)

class test_rk54(TestCase):

    def test_rk54_init(self):
        rk = seistools.integration.rk54coeff()
        self.assertEqual(rk.nstages, 6)
        np.testing.assert_array_equal(rk.a, np.array([0.2, 0.3, 0.6, 1., 0.875]))
        np.testing.assert_array_equal(rk.b, np.array([[0., 0., 0., 0., 0.], [0.2, 0., 0., 0., 0.], [3./40., 9./40, 0., 0., 0.],
                           [0.3, -0.9, 1.2, 0., 0.], [-11./54., 2.5, -70./27., 35./27., 0.],
                           [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]]))
        np.testing.assert_array_equal(rk.c, np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.]))
        np.testing.assert_array_equal(rk.cerr, np.array([37./378.-2825./27648., 0., 250./621.-18575./48384.,
                              125./594.-13525./55296., -277./14336, 512./1771.-0.25]))
