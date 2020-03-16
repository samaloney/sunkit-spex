import pytest
import numpy as np
import astropy.units as u
import time

from sunxspex import emission


def test_brem_collisional_loss():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    res = emission.collisional_loss(photon_energies)
    # IDL code to generate values taken from dedt variable
    # Brm_ELoss, [1.0, 10.0, 100.0, 1000.0], dedt
    # print, dedt, format='(e0.116)'
    assert np.array_equal(res, [3.627500063560997887179837562143802642822265625000000000000000e+02,
                             1.280267824055383414361131144687533378601074218750000000000000e+02,
                             4.973548353580351033542683580890297889709472656250000000000000e+01,
                             3.142020915656002344462649489287286996841430664062500000000000e+01])


def test_brem_cross_section():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    electron_energies = photon_energies+1
    res = emission.bremsstrahlung_cross_section(electron_energies, photon_energies)
    # IDL code to generate values taken from cross variable
    # Brm_BremCross, [1.0, 10.0, 100.0, 1000.0] + 1, [1.0, 10.0, 100.0, 1000.0], 1.2d, cross
    assert np.array_equal(res, [5.73978463331469571122141276398074503417042867600990581737900771142069089592041564173996448516845703125000000000000000e-22,
                                4.32296805664439775969072039261097905266988966784156545115957380161567091825247644010232761502265930175781250000000000e-24,
                                1.60532262380775996282816075092130871839665319276703294728182226053188113787384416752956894924864172935485839843750000e-26,
                                1.03272524007557480944264675309526017312053462559308495900959517166974022377851083476230087399017065763473510742187500e-28])


def test_brem_thicktarget():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.Brm2_ThickTarget(photon_energies, 5, 1000, 5, 10, 10000)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThickTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5,1000,5,10,10000])
    assert np.allclose(res, [3.5282883164459861975673291444470203760637592388101219288332650651371110170457542922313942321266289070536004146561026573181152343750000000000000000000000000000000e-34,
                             4.7704599119535816333888009376307253794338979001680378018735693161898543431597562004370757142782122350865847693057730793952941894531250000000000000000000000000000e-35,
                             5.8706375288383883794653394642884328310351719908731416661717605265534779452117171195165583689730501998349510017760621849447488784790039062500000000000000000000000e-38,
                             5.6778325682088220234323302312058238489241256777671145510307713777148278831380230603705156397264779285193048163904450120753608644008636474609375000000000000000000e-40,
                             3.1393033972122943399217762361196675784442364473962907197871297317258165597092736943466186069060646603374239299588666085583099629729986190795898437500000000000000e-41,
                             3.9809017161643275299947600495788886758098751244080330739303496379803824665480917313931530693948486455428976450721112456676564761437475681304931640625000000000000e-42,
                             8.1224603284076486034498584449141707184518819318528674666657045166661486787317626052468649633670783004042552366552598641646909527480602264404296875000000000000000e-43,
                             2.6828146475529079795700327856387287763742212421289479098429606686084675195201084428187451054412101727228938787342404914681992522673681378364562988281250000000000e-43], rtol=1e-8)

def test_brem_thintarget():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.Brm2_ThinTarget(photon_energies, 5, 1000, 5, 10, 10000)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThinTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5, 1000, 5, 10, 10000])
    res_idl = [1.3792306669225426e-53, 3.2319324672606256e-54, 1.8906418622815277e-58, 2.7707947605222644e-61,
               5.3706858279023008e-63, 3.4603542191953094e-64, 4.3847578461300751e-65, 1.0648152240531652e-65]
    assert np.allclose(res, res_idl, rtol=1e-8)

def test_brem_thintarget2():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.Brm2_ThinTarget(photon_energies, 3, 200, 6, 7, 10000)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThinTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 3, 200, 6, 7, 10000])
    res_idl = [1.410470406773663e-53, 1.631245131596281e-54, 2.494893311659408e-57, 2.082487752231794e-59,
               2.499983876763298e-61, 9.389452475896879e-63, 7.805504370370804e-64, 1.414135608438244e-64]
    assert np.allclose(res, res_idl, rtol=1e-8)
