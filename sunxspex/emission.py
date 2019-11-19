"""Summary
"""
import numpy as np

from scipy.special import lpmv

from sunxspex.constants import Constants

const = Constants()

def bremsstrahlung(elecctron_dist, e_low, e_high, ):
    """
    Summary

    Parameters
    ----------
    elecctron_dist : `astropy.units.Quantity`
        Description
    new : TYPE
        Description

    Returns
    -------
    photon_flux: `astropy.units.Quantity`
        Photon flux in photons s^-1 keV^-1 cm^-2

    Notes
    _____

    This function solves:


    """
    # mc2 = 510.98d+00
    # clight = 2.9979d+10
    # au = 1.496d+13
    # r0 = 2.8179d-13

    photon_flux = 0

    return photon_flux


def broken_powerlaw(x, p, q, eelow, eebrk, eehigh):
    # Obtain normalization coefficient, norm.
    n0 = (q-1.0) / (p-1.0) * eebrk**(p-1) * eelow**(1-p)
    n1 =  n0 - (q-1.0) / (p-1.0)
    n2 = (1.0 - eebrk**(q-1) * eehigh**(1-q))

    norm = 1.0/(n1+n2)

    F_E = np.zeros_like(x)

    if np.where(x<eelow)[0].size > 0:
        F_E[np.where(x<eelow)[0]] = 1.0

    if np.where((x < eebrk) & (x > eelow))[0].size > 0:
        F_E[np.where((x < eebrk) & (x > eelow))[0]] = norm*(n0*eelow**(p-1)*x[np.where((x < eebrk) & (x > eelow))[0]]**(1.0-p)-(q-1.0) / (p-1.0)+n2)

    if np.where((x < eehigh) & (x>eebrk))[0].size > 0:
        F_E[np.where((x < eehigh) & (x>eebrk))[0]] = norm*(eebrk**(q-1)*x[np.where((x < eehigh) & (x>eebrk))[0]]**(1.0-q)-(1.0-n2))

    return F_E


def powerlaw(x, low_energy_cutoff=10, high_energy_cutoff=100, index=3):
    # normalisation = (high_energy_cutoff**(index-1)/(index-1) +
    #     (low_energy_cutoff**(index-1)/(index-1)))

    normalisation = 1/(((high_energy_cutoff**(2 - 2*index))/((1 - index)**2)) - ((low_energy_cutoff**(2 - 2*index))/((1 - index)**2)))

    return normalisation * x**(1-index)


def collisional_loss(electron_energy):
    """Summary

    Parameters
    ----------
    electron_energy : TYPE
        Description
    """
    electron_rest_mass = const.get_constant('mc2')  #* u.keV #c.m_e * c.c**2

    gamma =  (electron_energy/electron_rest_mass)+1.0

    beta = np.sqrt(1.0 - (1.0/gamma**2))

    # TODO figure out what number is?
    energy_loss_rate = np.log(6.9447e+9*electron_energy)/beta

    return energy_loss_rate


def bremsstrahlung_cross_section(electron_energy, photon_energy, z=1.2):
    """
    Compute the relativistic electron-ion bremsstrahlung cross section
    differential in energy.

    Parameters
    ----------
    electron_energy : TYPE
        Electron energy
    photon_energy : TYPE
        Photon
    Z : TYPE
        Mean atomic number

    Returns
    -------
    TYPE
        Bremsstrahlung cross sections.

    Notes
    _____


    """
    # import pdb; pdb.set_trace()

    mc2 = const.get_constant('mc2')
    alpha = const.get_constant('alpha')
    twoar02 = const.get_constant('twoar02')

    # Numerical coefficients.
    c11 = 4.0/3.0
    c12 = 7.0/15.0
    c13 = 11.0/70.0
    c21 = 7.0/20.0
    c22 = 9.0/28.0
    c23 = 263.0/210.0

    # Calculate normalized photon and total electron energies.
    k = np.expand_dims(photon_energy/mc2, axis=1)
    e1 = (electron_energy/mc2)+1.0

    # Calculate energies of scatter electrons and normalized momenta.
    e2 = e1-k
    p1 = np.sqrt(e1**2-1.0)
    p2 = np.sqrt(e2**2-1.0)

    # Define frequently used quantities.
    e1e2 = e1*e2
    p1p2 = p1*p2
    p2sum = p1**2+p2**2
    k2 = k**2
    e1e23 = e1e2**3
    pe = p2sum/e1e23

    # Define terms in cross section.
    ch1 = (c11*e1e2+k2)-(c12*k2/e1e2)-(c13*k2*pe/e1e2)
    ch2 = 1.0+(1.0/e1e2)+(c21*pe)+(c22*k2+c23*p1p2**2)/e1e23

    # Collect terms.
    crtmp = ch1*(2.0*np.log((e1e2+p1p2-1.0)/k)-(p1p2/e1e2)*ch2)
    crtmp = z**2*crtmp/(k*p1**2)

    # Compute the Elwert factor.
    a1 = alpha*z*e1/p1
    a2 = alpha*z*e2/p2

    fe = (a2/a1)*(1.0-np.exp(-2.0*np.pi*a1))/(1.0-np.exp(-2.0*np.pi*a2))

    # Compute the differential cross section (units cm^2).
    cross_section = twoar02*fe*crtmp

    return cross_section


def brem(electron_energy, photon_energy, eelow, eebrk, eehigh, p, q, z=1.2):

    mc2 = const.get_constant('mc2')

    gamma = (electron_energy/mc2) + 1.0

    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z=1.2)

    collision_loss = collisional_loss(electron_energy)

    pc = np.sqrt(electron_energy*(electron_energy+2.0*mc2))

    electron_flux = broken_powerlaw(electron_energy, p, q, eelow, eebrk, eehigh)

    photon_flux = electron_flux * brem_cross * pc / collision_loss / gamma

    return photon_flux


def brm_guass_legendre(x1, x2, npoints):
    eps = 3e-14
    m = (npoints+1)//2

    x = np.zeros((x1.shape[0], npoints))
    w = np.zeros((x1.shape[0], npoints))

    xm = 0.5 * (x2+x1)
    xl = 0.5 * (x2-x1)

    for i in range(1, m+1):
        z = np.cos(np.pi*(i-0.25)/(npoints+0.5))
        z1 = np.inf
        while np.abs(z-z1) > eps:
            p1 = lpmv(0, npoints, z)
            p2 = lpmv(0, npoints-1, z)

            pp = npoints*(z*p1-p2)/(z**2 - 1.0)

            z1 = z
            z = z1-p1/pp

        x[:, i-1] = xm - xl * z
        x[:, npoints-i] = xm + xl * z
        w[:, i-1] = 2.0 * xl / ((1.0 - z**2) * pp**2)
        w[:, npoints-i] = w[:, i-1]

    return x, w

def brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh, p, q, z, a_lg, b_lg, l, intsum, ier):
    nlim = 12

    for ires in range(2, nlim):
        npoint = 2**(ires)
        if npoint > maxfcn:
            ier[l] = 1
            return intsum, ier

        eph1 = eph[l]

        xi, wi, = brm_guass_legendre(a_lg, b_lg, npoint)
        lastsum = np.copy(intsum)
        intsum[l] = np.sum((10.0**xi*np.log(10.0)*wi
                            *brem(10.0**xi, eph1, eelow, eebrk, eehigh, p, q, z)), axis=1)
        l1 = np.abs(intsum-lastsum)
        l2 = rerr*np.abs(intsum)
        ll = np.where(l1 > l2)

        if ll[0].size == 0:
            return intsum, ier
            break

def Brm2_DmlinO(a, b, maxfcn, rerr, eph, eelow, eebrk, eehigh, p, q, z):
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')

    en_vals = [eelow, eebrk, eehigh]
    en_vals = sorted(en_vals)

    # Part 1, below en_val[0] (usually eelow)
    # Create arrays for integral sum and error flags.
    intsum1 = np.zeros_like(a, dtype=np.float64)
    ier1 = np.zeros_like(a, dtype=np.float64)

    P1 = np.where(a < en_vals[0])

    if P1[0].size > 0:

        a_lg = np.log10(a[P1])
        b_lg = np.log10(np.full_like(a_lg, en_vals[0]))

        l = P1

        intsum1, ier1 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                      p, q, z, a_lg, b_lg, l)

    # ier = 1 indicates no convergence.
    if sum(ier1) > 0:
        raise ValueError('Part 1 integral did not converge for some photon energies.')


    # Part 2, between enval[0 and en_val[1](usually eelow and eebrk)
    intsum2 = np.zeros_like(a, dtype=np.float64)
    ier2 = np.zeros_like(a, dtype=np.float64)
    aa = a
    P2 = np.where(a < en_vals[1])

    if (P2[0].size > 0) and (en_vals[1] > en_vals[0]):
        if P1[0] != -1: aa[P1] = en_vals

        aa[P1] = en_vals[0]
        a_lg = np.log10(aa[P2])
        b_lg = np.log10(np.full_like(a_lg, en_vals[1]))

        l = P2[0]

        intsum2, ier2 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                        p, q, z, a_lg, b_lg, l, intsum2, ier2)

        if sum(ier2) > 0:
            raise ValueError('Part 2 integral did not converge for some photon energies.')

    # Part 3: between en_vals[1] and en_vals[2](usually eebrk and eehigh)
    intsum3 = np.zeros_like(a, dtype=np.float64)
    ier3 = np.zeros_like(a, dtype=np.float64)

    aa = a
    P3 = np.where(a < en_vals[2])

    if (P3[0].size > 0) and (en_vals[2] > en_vals[1]):
        if P2[0].size > 0: aa[P2] = en_vals[1]
        a_lg = np.log10(aa[P3])
        b_lg = np.log10(np.full_like(a_lg, en_vals[2]))

        L = P3

        intsum3, ier3 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                        p, q, z, a_lg, b_lg, l, intsum3, ier3)

        if sum(ier3) > 0:
            raise ValueError('Part 3 integral did not converge for some photon energies.')

    DmlinO = (intsum1 + intsum2 + intsum3) * (mc2 / clight)
    ier = ier1 + ier2 + ier3

    return DmlinO, ier


def bremstralung_thicktarget(eph, p, eebrk, q, eelow, eehigh):

    print(f'sub {const.ref}')

    # Constants
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')
    au = const.get_constant('au')
    r0 = const.get_constant('r0')

    # Max number of points
    maxfcn = 2048

    # Average atomic number
    z = 1.2

    # Relative error
    rerr = 1e-4

    # Numerical coefficient for photo flux
    fcoeff = (1.0/(4*np.pi*au**2))*(clight**2/mc2**4)

    decoeff = 4.0*np.pi*(r0**2)*clight

    # Create arrays for the photon flux and error flags.

    flux = np.zeros_like(eph, dtype=np.float64)
    iergq = np.zeros_like(eph, dtype=np.float64)

    if eelow >= eehigh: return flux

    l = np.where((eph < eehigh) & (eph > 0))

    if l[0].size > 0:
        ier = None
        flux[l[0]], iergq[l[0]] = Brm2_DmlinO(eph[l[0]], np.full_like(l[0], eehigh), maxfcn, rerr, eph[l[0]], eelow,
                                      eebrk, eehigh, p, q, z)

        flux = (fcoeff / decoeff) * flux

        return flux
    else:
        raise Warning(f'The photon energies ')