import numpy as np
from scipy import interpolate
import h5reader
from matplotlib import pyplot as plt
import seaborn as sns
from physconst import CLIGHT, GRAV, XMSUN, AMU, MEV
CLIGHT2 = CLIGHT**2
from nuc import Nuc
from glob import glob
from utils import cachedmethod

try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

class Evol():
    def __init__(self, model = 'z40_sn', directory = '/media/conrad/DATA1/output/', evol_dir = 'evol/'):
        self.model = model
        self.evol_dir = evol_dir
        self.data = h5reader.hydrof(directory, self.model, index = 0)
        self.m_traj = None


    def plot(self, analysis = 'rshock'):
        t    = np.loadtxt(self.evol_dir + self.model + '.time')
        if analysis == 'rshock':
            rmean = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.mean')
            rmin  = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.min')
            rmax  = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.max')

            plt.plot(t, rmean)
            plt.ylim(1.e6, 1.e9)
            plt.fill_between(t, rmin, rmax, alpha = 0.5)
            plt.xlabel('time (s)')
            plt.ylabel('shock radius (cm)')

            plt.yscale('log')

        elif analysis == 'energy':
            energy = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')

            maxjump = 2
            for i in range(1, len(energy) - 1):
                side_ratio = energy[i - 1] / energy [i + 1]
                if max(side_ratio, 1 / side_ratio) < maxjump:
                    average = 0.5 * (energy[i - 1] + energy[i + 1])
                    ratio = energy[i] / average
                    if max(ratio, 1 / ratio) > maxjump:
                        energy[i] = average

            plt.plot(t, energy)
            plt.xlabel('time (s)')
            plt.ylabel('diagnostic explosion energy (erg)')

        elif analysis == 'dedt':
            energy = np.loadtxt(self.evol_dir + self.model + '.' + 'energy')

            maxjump = 1.1
            for i in range(1, len(energy) - 1):
                side_ratio = energy[i - 1] / energy [i + 1]
                if max(side_ratio, 1 / side_ratio) < maxjump:
                    average = 0.5 * (energy[i - 1] + energy[i + 1])
                    ratio = energy[i] / average
                    if max(ratio, 1 / ratio) > maxjump:
                        energy[i] = average

            denergy = energy[2:] - energy[:-2]
            dt = t[2:] - t[:-2]
            dedt = denergy / dt

            plt.plot(t[1:-1], dedt)
            plt.xlabel('time (s)')
            plt.ylabel(r'$dE_\mathrm{expl}dt$ (erg/s)')

        elif analysis == 'nsmass':
            nsmass = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')
            nsmass = nsmass / XMSUN
            plt.plot(t, nsmass)
            plt.xlabel('time (s)')
            plt.ylabel(r'baryonic neutron star mass (solar mass)')

        elif analysis == 'trajectory':
            r_traj = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.100')
            n_traj = r_traj.shape[1]

            rmean = np.loadtxt(self.evol_dir + self.model + '.' + 'rshock' + '.mean')
            nsradius = np.loadtxt(self.evol_dir + self.model + '.' + 'nsradius' + '.scalar')
            rgain = np.loadtxt(self.evol_dir + self.model + '.' + 'rgain' + '.mean')

            rgrid = np.loadtxt(self.evol_dir + self.model + '.' + 'r')
            comp = np.loadtxt(self.evol_dir + self.model + '.' + 'comp' + '.550')
            ion_names = Nuc(20).names

            ylim = (1e6, 1e9)
            xlim = (0.27, 0.9)

            for i in range(n_traj):
                # iacc = r_traj[:,i] < nsradius
                # itns = np.argmax(iacc)
                # itplot = itns
                # do_label = np.any(iacc) and t[itns] > 0.36

                do_label = ylim[0] < r_traj[0,i] < ylim[1]
                itplot = 5
                if do_label:
                    itraj = np.searchsorted(rgrid, r_traj[0,i])
                    icomp = int(comp[0,itraj])
                    element = ion_names[icomp]
                    if element is not 'Ye':
                        plt.text(t[itplot], r_traj[itplot,i], element)

                plt.plot(t, r_traj[:,i], color = 'silver')
                plt.xlabel('time (s)')
                plt.ylabel('mass shell trajectory (cm)')
                plt.yscale('log')


            plt.plot(t, rmean, label = 'shock radius')
            plt.plot(t, nsradius, label = 'NS radius')
            plt.plot(t, rgain, label = 'gain radius')

            plt.legend()

            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])

        elif analysis == 'nulum':
            nulum = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.3')
            plt.plot(t, nulum[:,0], label = r'$\nu_e$')
            plt.plot(t, nulum[:,1], label = r'$\bar{\nu}_e$')
            plt.plot(t, nulum[:,2], label = r'$\mu+\tau$')
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel(r'neutrino luminosity at 400km (erg/s)')

        elif analysis == 'meanenu':
            meanenu = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.3')
            plt.plot(t, meanenu[:,0] / MEV, label = r'$\nu_e$')
            plt.plot(t, meanenu[:,1] / MEV, label = r'$\bar{\nu}_e$')
            plt.plot(t, meanenu[:,2] / MEV, label = r'$\mu+\tau$')
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel(r'mean neutrino energy at 400km (MeV)')


        elif analysis == 'heatrate':
            heatrate = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')
            plt.plot(t, heatrate)
            plt.ylim(0, 1e53)
            plt.xlabel('time (s)')
            plt.ylabel(r'neutrino heating rate in gain region $r_\mathrm{gain}<r<r_\mathrm{shock}$ (erg/s)')

        elif analysis == 'heateff':
            eff = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')
            plt.plot(t, eff)
            plt.ylim(0, 0.06)
            plt.xlabel('time (s)')
            plt.ylabel(r'neutrino heating efficiency in gain region $r_\mathrm{gain}<r<r_\mathrm{shock}$')

        elif analysis == 'theat':
            theat = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')

            plt.plot(t, theat)

            plt.xlim(0.35, 0.7)
            plt.ylim(0, 0.15)

            plt.xlabel('time (s)')
            plt.ylabel(r'heating timescale $\tau_\mathrm{heat}$ (s)')

        elif analysis == 'tadv':
            theat = np.loadtxt(self.evol_dir + self.model + '.' + analysis + '.scalar')

            plt.plot(t, theat)

            plt.xlim(0.35, 0.7)
            plt.ylim(0, 0.0005)

            plt.xlabel('time (s)')
            plt.ylabel(r'advection timescale $\tau_\mathrm{adv}$ (s)')

        elif analysis == 'timescales':
            theat = np.loadtxt(self.evol_dir + self.model + '.' + 'theat' + '.scalar')
            tadv = np.loadtxt(self.evol_dir + self.model + '.' + 'tadv' + '.scalar')

            plt.plot(t, theat, label = r'$\tau_\mathrm{heat}$')
            plt.plot(t, tadv, label = r'$\tau_\mathrm{adv}$')
            # plt.plot(t, tadv/theat)

            plt.xlim(0.35, 0.75)
            plt.ylim(0, 0.1)

            plt.legend()

            plt.xlabel('time (s)')
            plt.ylabel(r'timescale (s)')


        plt.show()

    def save(self, directory = '/media/conrad/DATA1/output/'):
        t, r, q = self.get()

        np.savetxt(self.evol_dir + self.model + '.time', t)
        np.savetxt(self.evol_dir + self.model + '.r', r)

        for qname in q:
            for op in q[qname]:
                np.savetxt(self.evol_dir + self.model + '.' + qname + '.' + str(op), q[qname][op])

    def get(self,
        analyses = [
                    # ('rshock', 'min'),
                    # ('rshock', 'mean'),
                    # ('rshock', 'max'),
                    # ('energy', 'scalar'),
                    # ('nsmass', 'scalar'),
                    # ('nsradius', 'scalar'),
                    # ('trajectory', 100),
                    # ('nulum', 3),
                    # ('meanenu', 3),
                    # ('rgain', 'mean'),
                    # ('heatrate', 'scalar'),
                    # ('heateff', 'scalar'),
                    # ('theat', 'scalar'),
                    ('tadv', 'scalar'),
                    # ('comp', 550),
                    ]
        ):
        print('Getting evolution...')
        nsteps = self.data.ngroups
        t = np.zeros((nsteps))
        r = self.data.xzn()

        q = {}
        for an, op in analyses:
            if an not in q:
                q[an] = {}
            if type(op) == int:
                q[an][op] = np.ndarray((nsteps, op))
            else:
                q[an][op] = np.ndarray((nsteps))

        for i in tqdm(range(nsteps)):
            self.initialize_step(i)
            t[i] = self.data.time()

            for an, op in analyses:
                x = self.analyse(an)
                if type(op) == int:
                    q[an][op][i,:] = self.array_reduce(x, op)
                else:
                    q[an][op][i] = self.array_reduce(x, op)

        return t, r, q

    def initialize_step(self, i):
        self.i = i
        self.data.set_index(i, silent = True)
        self.shape  = self.data.den().shape

    def analyse(self, op):
        if op == 'rshock':
            rshock = self.shock_radius(self.i)

            return rshock

        elif op == 'energy':
            ebind   = self.binding_energy(self.i)
            ebind[ebind < 0] = 0
            dvgr    = self.rel_volume(self.i)
            eexpl   = np.sum(ebind * dvgr)

            return eexpl

        elif op == 'nsmass':
            den     = self.data.den()

            ins     = self.ns_index(self.i)
            dvgr    = self.rel_volume(self.i)
            w       = self.lorentz(self.i)

            dmgr    = den * w * dvgr

            nsmass  = np.sum(dmgr[:ins,:,:])

            return nsmass

        elif op == 'nsradius':

            r       = self.data.xzn()
            ins     = self.ns_index(self.i)

            nsradius = r[ins]

            return nsradius

        elif op == 'trajectory':
            n       = 100

            r       = self.data.xzn()
            vex     = self.data.vex()
            vey     = self.data.vey()
            vez     = self.data.vez()
            den     = self.data.den()

            dvgr    = self.rel_volume(self.i)
            w       = self.lorentz(self.i)

            dmgr    = den * w * dvgr
            dmgr_s  = np.sum(np.sum(dmgr, axis = 2), axis = 1)
            mcum    = np.cumsum(dmgr_s)
            m       = mcum[-1]

            if self.m_traj is None:
                r_traj = np.logspace(np.log10(1.1*r[0]), np.log10(0.9*r[-1]), n)
                f      = interpolate.interp1d(r, mcum)
                self.m_traj = f(r_traj)

            mcum    = np.concatenate((np.array([0]), mcum))
            r       = np.concatenate((np.array([0]), r))
            f       = interpolate.interp1d(mcum, r)
            r_traj  = f(self.m_traj)

            return r_traj

        elif op == 'nulum':
            nulum   = self.neutrino_luminosity(self.i)

            return nulum

        elif op == 'meanenu':
            enu     = self.data.enu()
            dnu     = self.data.dnu()

            ir      = self.flux_rad_index(self.i)

            eshtot  = np.sum(np.sum(enu[ir,:,:,:], axis = 0), axis = 0)
            dshtot  = np.sum(np.sum(dnu[ir,:,:,:], axis = 0), axis = 0)

            emean   = eshtot / dshtot

            return emean

        elif op == 'rgain':
            r       = self.data.xzn()

            igain   = self.gain_index(self.i)
            rgain   = r[igain]

            return rgain

        elif op == 'heatrate':
            qdot    = self.heating_rate(self.i)

            return qdot

        elif op == 'heateff':
            qdot    = self.heating_rate(self.i)
            nulum   = self.neutrino_luminosity(self.i)

            eff     = qdot / (nulum[0] + nulum[1])

            return eff

        elif op == 'theat':
            theat   = self.heating_time(self.i)

            return theat

        elif op == 'tadv':
            tadv    = self.advection_time(self.i)

            return tadv

        elif op == 'comp':
            comp    = self.shell_composition(self.i)

            return comp

    def array_reduce(self, x, op):
        if op == 'mean':
            if np.all(np.isnan(x)):
                return np.nan
            else:
                return np.nanmean(x)
        elif op == 'min':
            if np.all(np.isnan(x)):
                return np.nan
            else:
                return np.nanmin(x)
        elif op == 'max':
            if np.all(np.isnan(x)):
                return np.nan
            else:
                return np.nanmax(x)
        elif op == 'sum':
            return np.sum(x)
        elif op == 'scalar':
            return x
        elif type(op) == int:
            return x

    @cachedmethod
    def ns_index(self, i):
        nsden = 1.e11

        den     = self.data.den()
        densh   = np.mean(np.mean(den, axis = 2), axis = 1)
        isns    = densh > nsden
        ins     = np.sum(isns)

        return ins

    @cachedmethod
    def flux_rad_index(self, i):
        rs = 4e7

        r       = self.data.xzn()
        ir      = np.searchsorted(r, rs)

        return ir

    @cachedmethod
    def pos_gain_region(self, i):
        qen     = self.data.qen()

        ir      = self.ns_index(self.i)

        posgain = qen > 0
        posgain[:ir,:,:] = False

        return posgain

    @cachedmethod
    def gain_region(self, i):
        ishock  = self.shock_index(self.i)
        isgain  = self.pos_gain_region(self.i)

        nphi    = self.shape[1]
        ntheta  = self.shape[2]

        for i in range(nphi):
            for j in range(ntheta):
                isgain[ishock[i,j]:,i,j] = False

        return isgain

    @cachedmethod
    def gain_index(self, i):
        posgain = self.gain_region(self.i)

        nphi    = self.shape[1]
        ntheta  = self.shape[2]

        igain   = np.ndarray((nphi, ntheta), dtype = 'int')

        for i in range(nphi):
            for j in range(ntheta):
                igain[i,j] = np.argmax(posgain[:,i,j])

        return igain

    @cachedmethod
    def lorentz(self, i):
        vex     = self.data.vex()
        vey     = self.data.vey()
        vez     = self.data.vez()

        vtot2   = vex**2 + vey**2 + vez**2
        w       = 1 / np.sqrt(1 - vtot2 / CLIGHT2)

        return w

    @cachedmethod
    def rel_volume(self, i):
        r1      = self.data.xzl()
        r2      = self.data.xzr()
        phi1    = self.data.yzl()
        phi2    = self.data.yzr()
        theta1  = self.data.zzl()
        theta2  = self.data.zzr()
        phiconf = self.data.phi()

        dr      = (1/3) * (r2**3 - r1**3)
        dphi    = abs(np.cos(phi2) - np.cos(phi1))
        dtheta  = theta2 - theta1

        dv      = dr[:,np.newaxis,np.newaxis] * dphi[np.newaxis,:,np.newaxis] * dtheta[np.newaxis,np.newaxis,:]

        dvgr    = phiconf**6 * dv

        return dvgr

    @cachedmethod
    def shock_index(self, i):
        epsiln = 1.0
        rshock0 = 1.e9
        pre = self.data.pre()
        r   = self.data.xzn()

        xr = np.roll(pre, -1, 0)
        xr[-1,:,:] = xr[-2,:,:]
        xl =  np.roll(pre, 1, 0)
        xl[0,:,:] = xl[1,:,:]

        yr = np.roll(pre, -1, 1)
        yr[:,-1,:] = yr[:,-2,:]
        yl = np.roll(pre, 1, 1)
        yl[:,0,:] = yl[:,1,:]

        zr = np.roll(pre, -1, 2)
        zl = np.roll(pre, 1, 2)

        shockx = (abs(xr - xl) - epsiln * pre)
        shocky = (abs(yr - yl) - epsiln * pre)
        shockz = (abs(zr - zl) - epsiln * pre)

        shock = np.maximum(shockx, shocky)
        shock = np.maximum(shock , shockz)

        detect = (shock > 0.0) * (r[:,np.newaxis,np.newaxis] < 1.2 * rshock0)
        mask = np.sum(detect, axis = 0) > 0
        ishock = self.shape[0] - np.argmax(detect[::-1], axis = 0) - 1

        return ishock

    @cachedmethod
    def shock_radius(self, i):
        r       = self.data.xzn()
        ishock  = self.shock_index(self.i)
        mask    = ishock == self.shape[0] - 1
        rshock  = r[ishock]
        rshock[mask] = np.nan

        return rshock

    @cachedmethod
    def heating_rate(self, i):
        qen     = self.data.qen()

        isgain  = self.gain_region(self.i)
        dvgr    = self.rel_volume(self.i)

        qdot    = np.sum(qen[isgain] * dvgr[isgain])

        return qdot

    @cachedmethod
    def neutrino_luminosity(self, i):
        nuflux  = self.data.fnu()
        r       = self.data.xzn()
        phi1    = self.data.yzl()
        phi2    = self.data.yzr()
        theta1  = self.data.zzl()
        theta2  = self.data.zzr()

        area    = 2 * np.pi * r[:,np.newaxis,np.newaxis]**2 * np.abs(phi2 - phi1)[np.newaxis,:,np.newaxis] * np.abs(theta2 - theta1)[np.newaxis,np.newaxis,:]

        ir      = self.flux_rad_index(self.i)
        nulum   = np.sum(np.sum(nuflux[ir,:,:,:] * area[ir,:,:,np.newaxis], axis = 0), axis = 0)

        return nulum

    @cachedmethod
    def heating_time(self, i):
        egain   = self.gain_binding_energy(self.i)
        qdot    = self.heating_rate(self.i)

        theat   = np.abs(egain / qdot)

        return theat

    @cachedmethod
    def advection_time(self, i):
        mgain   = self.gain_mass(self.i)
        mdot  = self.accretion_rate(self.i)

        tadv    = np.abs(mgain / mdot)
        print(tadv, mgain, mdot)
        return tadv

    @cachedmethod
    def gain_mass(self, i):
        den     = self.data.den()
        w       = self.lorentz(self.i)
        dvgr    = self.rel_volume(self.i)
        isgain  = self.gain_region(self.i)

        mgain   = np.sum(den[isgain] * w[isgain] * dvgr[isgain])

        return mgain

    @cachedmethod
    def accretion_rate(self, i):
        r       = self.data.xzn()
        phi1    = self.data.yzl()
        phi2    = self.data.yzr()
        theta1  = self.data.zzl()
        theta2  = self.data.zzr()
        den     = self.data.den()
        vex     = self.data.vex()
        alpha   = self.data.alpha()
        phiconf = self.data.phi()
        w       = self.lorentz(self.i)

        iacc    = self.flux_rad_index(self.i)

        dphi    = abs(np.cos(phi2) - np.cos(phi1))
        dtheta  = theta2 - theta1

        mdot    = - np.sum(alpha[iacc,:,:] * den[iacc,:,:] * w[iacc,:,:] * vex[iacc,:,:] * r[iacc]**2 * phiconf[iacc,:,:]**4 * dphi[:,np.newaxis] * dtheta[np.newaxis,:])

        return mdot

    @cachedmethod
    def binding_energy(self, i):
        r       = self.data.xzn()
        den     = self.data.den()
        ene     = self.data.ene()
        pre     = self.data.pre()
        alpha   = self.data.alpha()
        xnu     = self.data.xnu()

        if np.max(ene) > 1e40:
            ene     = ene * GRAV / CLIGHT2

        w       = self.lorentz(self.i)
        dvgr    = self.rel_volume(self.i)

        dmgr    = den * w * dvgr
        dmgr_s  = np.sum(np.sum(dmgr, axis = 2), axis = 1)

        dphi    = dmgr_s / r
        phiout  = - GRAV * np.cumsum(dphi[::-1])[::-1]
        phimod  = - 0.5 * phiout

        w2      = w**2
        preden  = pre / den
        enegr   = (ene + CLIGHT2 * (1 - w)) / w + preden * (1 - w2) / w2

        erest   = np.ndarray((21))
        nuclear = Nuc(20)
        erest[0:20] = nuclear.erest_nucleon()
        erest[20]   = nuclear.erest_electron()

        eneoff  = (- 939.5731 + 8.8) * MEV
        enegr   = enegr - (np.sum(xnu * erest, axis = 3) + eneoff)/ AMU * w

        etot    = alpha * (w * (CLIGHT2 + enegr + preden) - preden / w) - CLIGHT2

        ebind   = den * (etot + phimod[:,np.newaxis,np.newaxis])

        return ebind

    @cachedmethod
    def gain_binding_energy(self, i):
        ebind   = self.binding_energy(self.i)
        dvgr    = self.rel_volume(self.i)
        isgain  = self.gain_region(self.i)

        egain   = np.sum(ebind[isgain] * dvgr[isgain])

        return egain

    @cachedmethod
    def shell_composition(self, i):
        xnu     = self.data.xnu()

        xnush   = np.mean(np.mean(xnu, axis = 1), axis = 1)

        xnush[:,0]  = -1
        xnush[:,1]  = -1
        xnush[:,20] = -1

        imax    = np.argmax(xnush, axis = 1)

        return imax
