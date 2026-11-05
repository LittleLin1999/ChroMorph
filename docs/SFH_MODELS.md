# Independent stellar-population models

The default `FSPSStellarSED` family is constant star formation. The public FSPS
configuration is `sfh=1, const=1.0, fburst=0.0, sf_start=0.0`, with no truncation.
`age` maps to FSPS `tage` and measures the duration of continuous star formation
in Gyr. It is not the age of a single burst. Each component independently fits
its duration, log formed stellar mass, stellar metallicity and dust attenuation.
The constant SFR is derived as formed mass divided by duration in years.

`sfh: ssp` selects FSPS `sfh=0` as an explicit alternative. `sfh: exponential`
selects `sfh=1, const=0, fburst=0, sf_start=0` with an independently fitted `tau`.
The families can differ between components; defaults never create parameter
ties. Each component owns a separate SED object and an isolated FSPS process.

When `nebular: true`, both stellar-powered lines and continuum are enabled.
`gas_logu=-2` is fixed independently in each component by default. To fit it,
explicitly override that component's parameter with `vary: true`. Its gas
metallicity follows its own stellar metallicity unless explicitly overridden.
No such physical template is ever attached to the nuclear point source: its
flux is independent in every filter.

The generated comparison configurations are `A_free.yaml`,
`B_constant_sfr.yaml`, `C_ssp.yaml` and the optional `D_exponential.yaml`.
Compare the first three on identical images, PSFs, masks, noise, calibration
and centroid conventions. The CLI verifies matching likelihood inputs and
saves per-component morphology, formed-mass and photometric differences.
Neither appearance of a spectrum nor a lower informal BIC-like score alone
establishes a more accurate physical decomposition. Four bands may leave
population parameters weakly constrained, and overlapping components can
exchange flux or labels. Report bounds, null directions and uncertainties.

Filter-integrated flux per unit formed mass is cached independently for
each component. Exact keys include SFH configuration, all shape parameters,
nebular and attenuation settings, redshift/distance/foreground context and
throughput fingerprints. Mass changes only rescale cached band fluxes. There
is no rounded-key interpolation, shared global stellar vector or common
mass-to-light normalization. The cache has an explicit byte and entry budget.

Reference: [python-fsps documented population parameters](https://python-fsps.readthedocs.io/en/stable/stellarpop_api/).
