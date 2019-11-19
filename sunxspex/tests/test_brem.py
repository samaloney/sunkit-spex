import pytest
import numpy as np
import astropy.units as u

from sunxspex import emission

def test_brem_collisional_loss():
    photon_energies = [1.0, 10.0, 100.0, 1000.0] * u.keV
    res = emission.collisional_loss(photon_energies)
    assert np.allclose(res.value, [362.75000635609979, 128.02678240553834,
        49.735483535803510, 31.420209156560023])

def test_brew_cross_section():
    electron_energies = np.array([10.493034, 11.466207, 12.433249, 13.394704])     
    photon_energies = np.array([10.0, 11.0, 12.0, 13.0])
    res = emission.bremsstrahlung_cross_section(electron_energies, photon_energies)
    print(res)
    assert np.allclose(res, [3.7261376970063164e-24, 2.9328889825155176e-24,
           2.3446170409954541e-24,  1.8974174024238350e-24])

