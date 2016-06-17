import numpy as np
from physconst import AMU, CLIGHT, MEV, ME

class Nuc():
    def __init__(self, size = 20):
        if size == 20:
            self.nuc = np.array([
                 [0.,	0.,   1.,   8.375,  2.0,  0.000E+00, -0.100E+03, 0.000E+00], #n1
                 [1.,	1.,   1.,   7.082,  2.0,  0.000E+00, -0.100E+03, 0.000E+00], #p1
                 [2.,	2.,   4.,   2.618,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #He4
                 [3.,	6.,  12.,   0.580,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #C12
                 [4.,	8.,  16.,  -3.964,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #O16
                 [5., 10.,  20.,  -6.077,  1.0, -0.182E+02,  0.129E+01, 0.347E-01], #Ne20
                 [6., 12.,  24., -12.771,  1.0, -0.151E+02,  0.125E+01, 0.407E-01], #Mg24
                 [7., 14.,  28., -20.139,  1.0, -0.200E+02,  0.132E+01, 0.310E-01], #Si28
                 [8., 16.,  32., -24.470,  1.0, -0.232E+02,  0.433E+00, 0.132E+00], #S32
                 [9., 18.,  36., -28.493,  1.0, -0.189E+02, -0.581E-02, 0.170E+00], #Ar36
                 [10., 20.,  40., -32.915,  1.0, -0.415E+02,  0.164E+01, 0.148E+00], #Ca40
                 [11., 22.,  44., -35.420,  1.0, -0.111E+02,  0.629E+00, 0.173E+00], #Ti44
                 [12., 24.,  48., -40.499,  1.0, -0.610E+01, -0.374E-01, 0.212E+00], #Cr48
                 [13., 25.,  54., -51.924, 10.0, -0.531E+00, -0.157E+01, 0.351E+00], #Mn54
                 [14., 26.,  56., -56.877,  1.0, -0.740E+01,  0.498E-01, 0.246E+00], #Fe56
                 [15., 26.,  60., -56.495,  1.0, -0.549E+01, -0.102E+00, 0.281E+00], #Fe60
                 [16., 28.,  56., -51.197,  1.0, -0.254E+02, -0.107E+01, 0.347E+00], #Ni56
                 [17., 28.,  70., -60.250,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #ni70
                 [18., 28., 120., 0.     ,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #niXX
                 [19., 40., 200., 0.     ,  1.0,  0.000E+00, -0.100E+03, 0.000E+00]  #xxXX
                 ])
            self.names = [
                "n",
                "p",
                "He4",
                "C12",
                "O16",
                "Ne20",
                "Mg24",
                "Si28",
                "S32",
                "Ar36",
                "Ca40",
                "Ti44",
                "Cr48",
                "Mn54",
                "Fe56",
                "Fe60",
                "Ni56",
                "Ni70",
                "Ni120",
                "Zr200",
                "Ye",
            ]
        elif size == 41:
            self.nuc = np.array([
                [ 1.,  0.,   1.,   8.375,  2.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 2.,  1.,   1.,   7.082,  2.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 3.,  2.,   4.,   2.619,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 4.,  6.,  12.,   0.582,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 5.,  8.,  16.,  -3.961,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 6.,  8.,  20.,   5.788,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 7.,  9.,  20.,   1.463,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [ 8., 10.,  20.,  -6.077,  1.0, -0.182E+02,  0.129E+01, 0.347E-01], #
                [ 9., 10.,  24.,  -3.765,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [10., 11.,  24.,  -6.745,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [11., 12.,  24., -12.770,  1.0, -0.151E+02,  0.125E+01, 0.407E-01], #
                [12., 14.,  28., -20.135,  1.0, -0.200E+02,  0.132E+01, 0.310E-01], #
                [13., 16.,  32., -24.465,  1.0, -0.232E+02,  0.433E+00, 0.132E+00], #
                [14., 18.,  36., -28.486,  1.0, -0.189E+02, -0.581E-02, 0.170E+00], #
                [15., 20.,  40., -32.906,  1.0, -0.415E+02,  0.164E+01, 0.148E+00], #
                [16., 22.,  44., -35.416,  1.0, -0.111E+02,  0.629E+00, 0.173E+00], #
                [17., 22.,  50., -47.469,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [18., 24.,  48., -40.491,  1.0, -0.610E+01, -0.374E-01, 0.212E+00], #
                [19., 24.,  54., -52.780,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [20., 24.,  55., -50.649,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [21., 25.,  54., -51.914, 10.0, -0.531E+00, -0.157E+01, 0.351E+00], #
                [22., 25.,  55., -53.765,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [23., 25.,  56., -52.660,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [24., 26.,  52., -45.810,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [25., 26.,  54., -53.120,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [26., 26.,  55., -54.044,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [27., 26.,  56., -56.867,  1.0, -0.740E+01,  0.498E-01, 0.246E+00], #
                [28., 26.,  57., -56.136,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [29., 26.,  58., -57.807,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [30., 26.,  60., -56.454,  1.0, -0.549E+01, -0.102E+00, 0.281E+00], #
                [31., 27.,  55., -51.105,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [32., 27.,  56., -52.813,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [33., 28.,  56., -51.188,  1.0, -0.254E+02, -0.107E+01, 0.347E+00], #
                [34., 28.,  57., -53.059,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [35., 28.,  58., -56.903,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [36., 28.,  60., -60.539,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [37., 28.,  70., -53.166,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [38., 28.,  80., -16.215,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [39., 30.,  60., -51.277,  1.0,  0.000E+00, -0.100E+03, 0.000E+00], #
                [40., 40., 121.,  29.796,  1.0,  0.000E+00, -0.100E+03, 0.000E+00]
                ])

    def i(self):
        return self.nuc[:,0]

    def z(self):
        return self.nuc[:,1]

    def a(self):
        return self.nuc[:,2]

    def excess(self):
        return self.nuc[:,3] * MEV

    def erest(self):
        baryon_rest_energy = AMU * CLIGHT**2
        return self.excess() + self.a() * baryon_rest_energy

    def erest_nucleon(self):
        return self.erest() / self.a()

    def erest_electron(self):
        return ME * CLIGHT**2
