import pytest
import numpy as np
import astropy.units as u

from sunxspex import emission


def test_brem_collisional_loss():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    res = emission.collisional_loss(photon_energies)
    # IDL code to generate values taken from dedt variable
    # Brm_ELoss, [1.0, 10.0, 100.0, 1000.0], dedt
    assert np.allclose(res, [362.75000635609979, 128.02678240553834,
        49.735483535803510, 31.420209156560023])


def test_brew_cross_section():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    electron_energies = photon_energies+1
    res = emission.bremsstrahlung_cross_section(electron_energies, photon_energies)
    # IDL code to generate values taken from cross variable
    # Brm_BremCross, [1.0, 10.0, 100.0, 1000.0] + 1, [1.0, 10.0, 100.0, 1000.0], 1.2, cross
    assert np.allclose(res, [5.7397848834968013e-22, 4.3229682710041217e-24, 1.6053227084818438e-26,
                             1.0327252953769905e-28]
)


def test_bremstralung_thicktarget():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.bremstralung_thicktarget(photon_energies, 5, 1000, 5, 10, 10000)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThickTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5,1000,5,10,10000])
    assert np.allclose(res, [3.5282883164459862e-34, 4.7704599119535816e-35, 5.8706375288383884e-38,
                             5.6778325682088220e-40, 3.1393033972122943e-41, 3.9809017161643275e-42,
                             8.1224603284076486e-43, 2.6828146475529080e-43])
