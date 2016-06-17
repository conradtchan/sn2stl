"""
David's Reader for VERTEX HDF5 hydro output files

Usage:
Run in directory
y = h5reader_david.hydrof()
y.den(index = 3) gives 3rd time step of density
"""

from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
import time
import physconst as pc
from scipy import interpolate


# For Prometheus, we need to throw out alpha, beta1, beta2, beta3, the Prometheus files don't have them


class hydrof():
    def __init__(self, directory = None, model = 'z40_mix2d', index=0):
        """
        alpha                lapse function
        beta1                shift vector (radial)
        beta2                shift vector (meridional)
        beta3                shift vector (zonal)
        bx                    radius of zone boundaries
        by                    inclination (theta) of zone boundaries
        bz                    azimuth (phi) of zone boundaries
        cpo                    chemical potentials
        den                    density
        dnu                    neutrino number density (lab)
        dt                    hydro timestep interval
        ene                    specific energy
        enu                    neutrino energy density
        eph                    Lapse function (grav. potential)
        fnu                    neutrino energy flux (lab)
        gac                    adiabatic index
        gam                    Gamma factor (grav. potential)
        gpo                    grav. pot. in hydro grid
        ish                    shock zones
        nstep                hydro timestep number
        phi                    conformal factor
        pnu                    neutrino pressure
        pre                    pressure
        qen                    Quell-ene
        qmo                    Quell-mom
        qye                    Quell-Ye
        restmass_version    Energy normalization: different version for the
                            subtraction of rest masses from the energy used in
                            PPM:
                            0: uses energy defined as in EoS
                            1: subtracts from EoS energy the baryon rest masses,
                            assuming that heavy elements have the mass of
                            fe56
                            2: subtracts from EoS energy the baryon rest
                            masses
                            3: subtracts from EoS energy the baryon and
                            unpaired electron rest masses. Caution! This
                            version violates energy!
        sto                    entropy per baryon
        tem                    temperature
        tgm                    Enclosed gravitational mass
        time                physical time
        tm                    Enclosed baryonic mass
        vex                    velocity in radial direction
        vey                    velocity in theta direction
        vez                    velocity in phi direction
        xcart                X coordinate (Cartesian)
        xnu                    composition (mass fractions)
        xzl                    radius at left rim in hydro grid
        xzn                    radius of zone center
        xzr                    radius at right rim in hydro grid
        ycart                Y coordinate (Cartesian)
        yzl                    inclination (theta) at left rim in hydro grid
        yzn                    inclination (theta) of zone center
        yzr                    inclination (theta) at right rim in hydro grid
        zcart                Z coordinate (Cartesian)
        zzl                    azimuth (phi) at left rim in hydro grid
        zzn                    azimuth (phi) of zone center
        zzr                    azimuth (phi) at right rim in hydro grid
        """

# A list of all the variable names.
        self.variables = ['alpha','beta1','beta2','beta3','bx','by','bz','cpo','den','dnu','dt',
        'ene','enu','eph','fnu','gac','gam','gpo','ish','nstep','phi','pnu',
        'pre','qen','qmo','qye','restmass_version','sto','tem','tgm','time',
        'tm','vex','vey','vez','xcart','xnu','xzl','xzn','xzr','ycart','yzl',
        'yzn','yzr','zcart','zzl','zzn','zzr']

        if directory is None:
            directory = '.'

# Read all the hdf5 files into the object.
        # hdf5 = []
        # for root,dirs,files in os.walk(directory):
        #     for name in files:
        #         if name.startswith('z') and name[-8:].isnumeric():
        #             hdf5 += [name]

        hdf5 = glob(os.path.join(directory, '{}.o*'.format(model)))

        hdf5.sort()
        steps = []
        for name in hdf5:
            steps += [h5py.File(name)]
            self.steps = steps

# Creats a mapping key between indices of sub-timesteps and major-steps.
        key = {}
        M=-1
        N=-1
        for i in range(len(self.steps)):
            length = len(self.steps[i])
            N=N+1
            for j in range(length):
                M = M+1
                key[M] = N
        self.key = key

# Create a list of the group (sub-timestep) names and print them.
        self.groups = []
        for i in range(len(self.steps)):
            for j in self.steps[i]:
                # print(j)
                self.groups.append(j)
        self.index = index
        self.ngroups = len(self.groups)
        print('Number of timesteps: ',self.ngroups)

# Method for changing the current index value
    def set_index (self, index, silent = False):
        self.index = index
        if not silent:
            print('Current index is now: ', self.index)

# ------------------------------------------------------------------------------
# Here we define methods to load a single variable at a given timestep.
# The timestep loaded by default is the current index. But any timestep can be
# loaded by providing the keyword argument: index=....

    def alpha (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/alpha']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def beta1 (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/beta1']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def beta2 (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/beta2']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def beta3 (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/beta3']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def bx (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/bx']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def by (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/by']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def bz (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/bz']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def cpo (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/cpo']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def den (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/den']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def dnu (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/dnu']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def dt (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/dt']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def ene (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/ene']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def enu (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/enu']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def eph (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/eph']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def fnu (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/fnu']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def gac (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/gac']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def gam (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/gam']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def gpo (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/gpo']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def ish (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/ish']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def nstep (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/nstep']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def phi (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/phi']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def pnu (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/pnu']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def pre (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/pre']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def qen (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/qen']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def qmo (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/qmo']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def qye (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/qye']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def restmass_version (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/restmass_version']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def sto (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/sto']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def tem (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/tem']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def tgm (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/tgm']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def time (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/time']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def tm (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/tm']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def vex (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/vex']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def vey (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/vey']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def vez (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/vez']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def xcart (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/xcart']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def xnu (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/xnu']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def xzl (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/xzl']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def xzn (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/xzn']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def xzr (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/xzr']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def ycart (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/ycart']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def yzl (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/yzl']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def yzn (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/yzn']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def yzr (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/yzr']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def zcart (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/zcart']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def zzl (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/zzl']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def zzn (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/zzn']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value
    def zzr (self, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs['index']
            if isinstance(index, int) and 0<=index<=self.ngroups-1:
                self.index = index
            else:
                print('Error: Please choose an integer index between 0 and ',self.ngroups-1)
                ok = False
        if ok:
            tmp = self.steps[self.key[self.index]][self.groups[self.index]+'/zzr']
            value = np.empty (tmp.shape)
            tmp.read_direct(value)
            value = np.transpose(value)
            return value

# ------------------------------------------------------------------------------
# This method loads all the variables in all the timesteps ---------------------

    def everything (self):
        print('Loading all timesteps...')
        #timestep = {}
        timestep = []
        T = -1
        for ii in range(len(self.steps)):            # for all coarse steps (by index)
            for i in self.steps[ii]:                # for all substeps in the course step
                T=T+1
                D = {}
                for j in self.steps[ii][i]:            # for all variables names in the substep
                    tmp = self.steps[ii][i+'/'+j]   # set the hdf5 path as a variable
                    value = np.empty(tmp.shape)        # create an empty array with the size/shape required
                    tmp.read_direct(value)            # read the values from the hdf5 path into the empty array
                    value = np.transpose(value)        # reverse index ordering
                    D[j] = value                    # create sub-dictionary of the variables + values
                #timestep[T]=D                        # add into a parent dictionary, containting each sub-step
                timestep += [D]

        self.timestep = timestep
        print('Done.')

# ------------------------------------------------------------------------------
# This method can be used to do an animated plot in time of any of the variables
# as a function of radius. It also has options for log axes and the delay between
# plotting each successive timestep.

    def dav_plot(self,variable = 'vex',xlog=True, ylog=False, delay=0.0):

        # A dictionary of the variable names and their descriptions.
        var = {'alpha':                      'lapse function',
                'beta1':                  'shift vector (radial)',
                'beta2':                  'shift vector (meridional)',
                'beta3'    :                  'shift vector (zonal)',
                'bx'    :                  'radius of zone boundaries',
                'by'    :                  'inclination (theta) of zone boundaries',
                'bz'    :                    'azimuth (phi) of zone boundaries',
                'cpo'    :                  'chemical potentials',
                'den'    :                  'density',
                'dnu'    :                  'neutrino number density (lab)',
                'dt'    :                    'hydro timestep interval',
                'ene'    :                  'specific energy',
                'enu'    :                  'neutrino energy density',
                'eph'    :                  'Lapse function (grav. potential)',
                'fnu'    :                  'neutrino energy flux (lab)',
                'gac'    :                  'adiabatic index',
                'gam'    :                  'Gamma factor (grav. potential)',
                'gpo'    :                  'grav. pot. in hydro grid',
                'ish'    :                  'shock zones',
                'nstep'    :                    'hydro timestep number',
                'phi'    :                  'conformal factor',
                'pnu'    :                  'neutrino pressure',
                'pre'    :                  'pressure',
                'qen'    :                  'Quell-ene',
                'qmo'    :                  'Quell-mom',
                'qye'    :                  'Quell-Ye',
                'restmass_version'    :  '''Energy normalization: different version for the
                      subtraction of rest masses\\nfrom the energy used in
                      PPM:\\n  0: uses energy defined as in EoS\\n
                      1: subtracts from EoS energy the baryon rest masses,
                      assuming\\n     that heavy elements have the mass of
                      fe56\\n  2: subtracts from EoS energy the baryon rest
                      masses\\n  3: subtracts from EoS energy the baryon and
                      unpaired electron\\n     rest masses. Caution! This\
                      version violates energy!''',
                'sto'    :                  'entropy per baryon',
                'tem'    :                  'temperature',
                'tgm'    :                  'Enclosed gravitational mass',
                'time'    :                  'physical time',
                'tm'    :                  'Enclosed baryonic mass',
                'vex'    :                  'velocity in radial direction',
                'vey'    :                  'velocity in theta direction',
                'vez'    :                  'velocity in phi direction',
                'xcart'    :                  'X coordinate (Cartesian)',
                'xnu'    :                  'composition (mass fractions)',
                'xzl'    :                  'radius at left rim in hydro grid',
                'xzn'    :                  'radius of zone center',
                'xzr'    :                  'radius at right rim in hydro grid',
                'ycart'    :                  'Y coordinate (Cartesian)',
                'yzl'    :                  'inclination (theta) at left rim in hydro grid',
                'yzn'    :                  'inclination (theta) of zone center',
                'yzr'    :                  'inclination (theta) at right rim in hydro grid',
                'zcart'    :                  'Z coordinate (Cartesian)',
                'zzl'    :                  'azimuth (phi) at left rim in hydro grid',
                'zzn'    :                  'azimuth (phi) of zone center',
                'zzr'    :                  'azimuth (phi) at right rim in hydro grid'}



        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        for i in range(self.ngroups):
            plt.cla()
            ax = fig.add_subplot(111)
            ax.set_xscale('log')
            x = self.timestep[0]['xzn']
            y = self.timestep[i]['vex'][0,0]
            plt.plot(x,y)
            plt.draw()
        '''
        '''
        import time
        x = self.timestep[0]['xzn']
        y = self.timestep[0]['vex'][0,0]

        fig, axes = plt.subplots(nrows=1)
        styles = ['b-']
        lines = [ax.plot(x, y, style)[0] for ax, style in (axes, styles)]

        fig.show()

        tstart = time.time()
        for i in range(self.ngroups):
            for j, line in enumerate(lines, start=1):
                line.set_ydata(self.timestep[i]['vex'][0,0])
            fig.canvas.draw()

        print('FPS:' , 20/(time.time()-tstart))
        '''
        # Close any existing figure windows
        plt.close()

        # Find the limits for the y axis
        Max = max(self.timestep[0][variable][0,0])
        Min = min(self.timestep[0][variable][0,0])
        for i in range(len(self.timestep)):
            Ma = max(self.timestep[i][variable][0,0])
            Mi = min(self.timestep[i][variable][0,0])
            if Ma>Max: Max = Ma
            if Mi<Min: Min = Mi

        # Set the initial x and y values
        x = self.timestep[0]['xzn']
        y = self.timestep[0][variable][0,0]

        # Set up your figure and axes
        fig, axes = plt.subplots(nrows=1)
        if xlog is True: axes.set_xscale('log')
        axes.set_ylim(Min,Max)
        axes.set_title('{} vs Radius'.format(var[variable]))
        axes.set_xlabel('Radius')
        axes.set_ylabel(var[variable])
        if ylog is True: axes.set_yscale('log')
        axes = np.array([axes],dtype=object)

        styles = ['b-']
        def plot(ax, style):
            return ax.plot(x, y, style, animated=True)[0]
        lines = [plot(ax, style) for ax, style in zip(axes, styles)]

        fig.show()

        # We need to draw the canvas before we start animating...
        fig.canvas.draw()

        # Let's capture the background of the figure
        backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]

        tstart = time.time()
        for i in range(self.ngroups):
            time.sleep(delay)    # Pause between each frame
            items = enumerate(zip(lines, axes, backgrounds), start=1)
            T = self.timestep[i]['time']
            T = float(T)
            print('Time = ', round(T,4), '  sec', end='\r'),
            for j, (line, ax, background) in items:
                fig.canvas.restore_region(background)
                line.set_ydata(self.timestep[i][variable][0,0])
                ax.draw_artist(line)
                fig.canvas.blit(ax.bbox)

        # How many frames it plotted each second
        print('FPS:' , self.ngroups/(time.time()-tstart))

#-------------------------------------------------------------------------------
# Do the analysis (i.e. find shock radius etc.)
# If your computer did not have enough memory to load all the variables at all
# the timesteps using the 'everything' method, then provide the keyword argument
# 'mem=False' to use each information at each timestep individually. This of
# course takes much longer to do.

    def analysis (self, mem=True):
        import time as T
        t_start = T.time()

        import phycon_and_nuc_table as nt
        pc_nuc = nt.pc_nuc
        wc_mb = nt.wc_mb
        wc_me = nt.wc_me
        pc_meverg = nt.pc_meverg
        pc_mb = nt.pc_mb
        pc_msol = nt.pc_msol
        pc_cl = nt.pc_cl

        for i in range(self.ngroups):

            if mem:
                bx = self.timestep[i]['bx']
                by = self.timestep[i]['by']
                bz = self.timestep[i]['bz']
                cpo = self.timestep[i]['cpo']
                den = self.timestep[i]['den']
                dnu = self.timestep[i]['dnu']
                dt = self.timestep[i]['dt']
                ene = self.timestep[i]['ene']
                if np.amax(ene)>1.e40:
                    ene = ene*pc.Kepler.gee/pc.Kepler.c**2
                enu = self.timestep[i]['enu']
                eph = self.timestep[i]['eph']
                fnu = self.timestep[i]['fnu']
                gac = self.timestep[i]['gac']
                gam = self.timestep[i]['gam']
                gpo = self.timestep[i]['gpo']
                ish = self.timestep[i]['ish']
                nstep = self.timestep[i]['nstep']
                pnu = self.timestep[i]['pnu']
                pre = self.timestep[i]['pre']
                qen = self.timestep[i]['qen']
                qmo = self.timestep[i]['qmo']
                qye = self.timestep[i]['qye']
                restmass_version = self.timestep[i]['restmass_version']
                sto = self.timestep[i]['sto']
                tem = self.timestep[i]['tem']
                tgm = self.timestep[i]['tgm']
                time = self.timestep[i]['time']
                tm = self.timestep[i]['tm']
                vex = self.timestep[i]['vex']
                vey = self.timestep[i]['vey']
                vez = self.timestep[i]['vez']
                xcart = self.timestep[i]['xcart']
                xnu = self.timestep[i]['xnu']
                xzl = self.timestep[i]['xzl']
                xzn = self.timestep[i]['xzn']
                xzr = self.timestep[i]['xzr']
                ycart = self.timestep[i]['ycart']
                yzl = self.timestep[i]['yzl']
                yzn = self.timestep[i]['yzn']
                yzr = self.timestep[i]['yzr']
                zcart = self.timestep[i]['zcart']
                zzl = self.timestep[i]['zzl']
                zzn = self.timestep[i]['zzn']
                zzr = self.timestep[i]['zzr']

            else:
                bx = self.bx(index=i)
                by = self.by(index=i)
                bz = self.bz(index=i)
                cpo = self.cpo(index=i)
                den = self.den(index=i)
                dnu = self.dnu(index=i)
                dt = self.dt(index=i)
                ene = self.ene(index=i)
                enu = self.enu(index=i)
                eph = self.eph(index=i)
                fnu = self.fnu(index=i)
                gac = self.gac(index=i)
                gam = self.gam(index=i)
                gpo = self.gpo(index=i)
                ish = self.ish(index=i)
                nstep = self.nstep(index=i)
                pnu = self.pnu(index=i)
                pre = self.pre(index=i)
                qen = self.qen(index=i)
                qmo = self.qmo(index=i)
                qye = self.qye(index=i)
                restmass_version = self.restmass_version(index=i)
                sto = self.sto(index=i)
                tem = self.tem(index=i)
                tgm = self.tgm(index=i)
                time = self.time(index=i)
                tm = self.tm(index=i)
                vex = self.vex(index=i)
                vey = self.vey(index=i)
                vez = self.vez(index=i)
                xcart = self.xcart(index=i)
                xnu = self.xnu(index=i)
                xzl = self.xzl(index=i)
                xzn = self.xzn(index=i)
                xzr = self.xzr(index=i)
                ycart = self.ycart(index=i)
                yzl = self.yzl(index=i)
                yzn = self.yzn(index=i)
                yzr = self.yzr(index=i)
                zcart = self.zcart(index=i)
                zzl = self.zzl(index=i)
                zzn = self.zzn(index=i)
                zzr = self.zzr(index=i)

            qx = np.size(xzn)
            qy = np.size(yzn)
            qz = np.size(zzn)
            qc = np.size(cpo[0,0,0,:])
            qn = np.size(xnu[0,0,0,:])
            qye = qye[:,:,:,0]
            qen = qen[:,:,:]
            qs = np.size(dnu[0,0,0,0])

            if qy == 1:
                yzl[0] = 0.
                yzr[0] = np.pi
            if qz == 1:
                zzl[0] = 0.
                zzr[0] = 2.0*np.pi

            if gpo[1,1] > -1.e15:  # Relativistic model
                if mem:
                    alpha = self.timestep[i]['alpha'][0:,0:,0:]
                    beta1 = self.timestep[i]['beta1'][0:,0:,0:]
                    beta2 = self.timestep[i]['beta2'][0:,0:,0:]
                    beta3 = self.timestep[i]['beta3'][0:,0:,0:]
                    phi   = self.timestep[i]['phi'][0:,0:,0:]
                else:
                    alpha = self.alpha(index=i)
                    beta1 = self.beta1(index=i)
                    beta2 = self.beta2(index=i)
                    beta3 = self.beta3(index=i)
                    phi = self.phi(index=i)
                gpo = np.zeros([qx+1,qy+1,1])
                gpo[1:,1:,:] = phi


            wl =1.0/np.sqrt(1.0-(vex**2+vey**2+vez**2)/pc.Kepler.c**2)
            if np.amin(gpo)<= 0.0:
                wl[:,:,:]=1.0
            else:
                tau = ene*(den*wl)
                ene=(ene+pc.Kepler.c**2*(1.0-wl))/wl+pre*(1.0-wl**2)/(den*wl**2)

            eneoff = -939.5731 + 8.8

            mnuc = np.zeros(21)
            mnuc[0:20] = (pc_nuc[:,3]+pc_nuc[:,2]*wc_mb)/(pc_nuc[:,2])
            mnuc[20] = wc_me
            for n in range(qn):
                ene[:,:,:]=ene[:,:,:]-xnu[:,:,:,n]*mnuc[n]*pc_meverg/pc_mb

            #Now ene is the thermal energy density per unit mass
            ene[:,:,:]=ene[:,:,:]-eneoff*pc_meverg/pc_mb
            # etot = binding energy per unit mass
            etot=alpha*(wl*(pc_cl**2+ene+pre/den)-pre/den)-pc_cl**2


            # Volume  dv of each grid cell
            dv_r = 1./3. * (xzr**3-xzl**3)
            dv_theta = abs(np.cos(yzl)-np.cos(yzr))
            dv_phi = zzr - zzl

            dv = dv_r[:,np.newaxis,np.newaxis]*dv_theta[np.newaxis,:,np.newaxis]*dv_phi[np.newaxis,np.newaxis,:]
            dv=dv*phi**6


            # Compute enclosed mass/mass coordinate
            tma = np.cumsum(np.sum(den*dv, axis = (1,2))) / pc_msol
            # pc_msol: Solar mass in g

    #------------------------------- Shock position (1D/2D) --------------------
            epsiln=1.0

            ishock=np.zeros([qx,qy])
            krit=np.zeros([qx,qy])

            pmin0=pre[0:qx-3,:,0]#??? 3D ? could not broadcast input array from shape (547,1,1) into shape (547,1)

            pmin1=pre[2:qx-1,:,0]#???

            j=np.where(pmin0 > pmin1)
            cnt = np.size(j)
            if (cnt is not 0): pmin0[j]=pmin1[j]
            krit[1:qx-2,:]=np.abs(pre[0:qx-3,:,0]-pre[2:qx-1,:,0])-(epsiln*pmin0)
            j=np.where(krit > 0.0)
            cnt = np.size(j)
            if (cnt > 0): ishock[j]=1.0

            if (qy > 1):
                pmin0=pretot[:,0:qy-3]
                pmin1=pretot[:,2:qy-1]
                j=np.where(pmin0 > pmin1)
                if (cnt is not 0): pmin0[j]=pmin1[j]
                krit[:,1:qy-2]=np.abs(pre[:,0:qy-3,0]-pre[:,2:qy-1,0])-(epsiln*pmin0)
                j=np.where(krit > 0.0)
                cnt = np.size(j)
                if (cnt > 0): ishock[j]=1.0

            jj=np.where(ishock > 0)
            if (cnt < 1):
               vmin=np.min(vex)
               i_shock = np.unravel_index(np.argmin(vex),np.shape(vex))
            else:
               i_shock=np.max(jj)

            rshock=xzn[i_shock[0]]
            print(rshock)


#------------------------------------------------------------------------------#
        #Mass shells
        if np.size(tm0) < 5:
           cx0=[10.0e0**(i/60*4+5) for i in range(1,61)]
           f=interpolate.interp1d(xzn,tma) #??? tm0???
           tm0=f(cx0)            # ???

        f=interpolate.interp1d(xzn,tma) # ???
        r_m=f(tm0)                        # ???

        f=interpol(xzn,den,1e11)
        r_m[0] = f(range(xzn[0],xzn[-1],1e11))
        r_m[0] = np.max([r_m[0],1e5])

        print(r_m)
