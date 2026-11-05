# J1042 empirical PSF test fixtures

`j1042_empirical_v2_psfs.npz` copies the four existing hstpipe empirical_v2
PSF arrays without reconstructing or re-drizzling them. The companion JSON
records the source FITS filenames, source SHA256 hashes, declared centers,
pixel scales, photometric calibrations and median saved per-band noise.

Source: the user's HST product collection at
`products/uvis/psfs/J104242.43+372147.63/`.
No observed target image pixels are included. Tests inject synthetic objects
on 31×31 grids; the Gaussian noise has the measured median amplitude but
omits spatial noise variation and correlated drizzle noise. These tests
validate numerical behavior; they do not establish physical component recovery
for the real galaxy.
