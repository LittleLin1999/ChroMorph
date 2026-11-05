# J1042 data setup

The package uses existing hstpipe DRC products and drizzled empirical_v2 PSFs.
No target images, fitted parameter products, per-filter noise metadata or
empirical kernel arrays are published with the source distribution.

Run discovery against your local data collection:

```sh
chromorph discover-j1042 /home/linxiaojing/data/lrd-squared/hst \
  --output results/j1042_inventory.json --config-dir examples/generated
```

This validates source hashes and discovers actual filters, WCS, calibration,
noise, masks, PSF provenance and available redshift metadata. It writes the
inventory locally and generates component configurations from the selected
fit. Missing metadata or unresolved QA is explicitly reported. Review those
outputs before fitting; do not treat the generic checked-in YAML starting
values as measurements or validated geometry.

Local integration tests can use existing kernel fixtures via
`CHROMORPH_J1042_FIXTURES=/path/to/fixtures pytest tests/test_real_psfs.py`.
The expected local filenames and provenance layout are documented in
`tests/data/README.md`. Their publication requires the data owner's approval.
