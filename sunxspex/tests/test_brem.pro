photon_energies = [1.0, 10.0, 100.0, 1000.0] ;in keV
nph = n_elements(photon_energies)
electron_energies = photon_energies + 1.
res = dblarr(nph)
res2 = res
for i=0, nph-1 do begin
    abr_xsection_avg, photon_energies[i], electron_energies[i], xsec
    ; the following is based on Haug 1997: https://ui.adsabs.harvard.edu/abs/1997A%26A...326..417H/abstract
    ; the unit is cm^2 (m_e c^2)^-1 (per ion)
    brm_bremcross, electron_energies[i], photon_energies[i], 1.2, xsec2
    res[i] = xsec
    res2[i] = xsec2/510.98 ;convert to cm^2 keV^-1
    print, 'Photon energy: ', photon_energies[i]
    print, 'X-section 1: ', res[i]
    print, 'X-section 2: ', res2[i]
endfor

; test thick target
photon_energies = [5., 10., 50., 150., 300., 500., 750., 1000.] ; in keV
flux = Brm2_ThickTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5, 1000, 5, 10, 10000])

; test thin target
photon_energies = [5., 10., 50., 150., 300., 500., 750., 1000.] ; in keV

flux = Brm2_ThinTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5, 1000, 5, 10, 10000])
end