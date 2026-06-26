# fusio

**fusio** is a Python package for converting input/output files between the many different file formats used by fusion plasma simulation codes.

## Architecture

All format classes inherit from a common base class `io` (`fusio/src/fusio/classes/io.py`), which stores data as two xarray `Dataset`s — `input` and `output` — backed by an `xr.DataTree`.

Format conversion is done via:
- `obj.to('target_format')` — converts from any format to another
- `TargetFormat.from_source_format(obj)` — class method alternative

## Supported Formats

| Class | File | Format |
|---|---|---|
| `plasma_io` | `classes/plasma.py` | Internal NetCDF-based intermediate representation |
| `gacode_io` | `classes/gacode.py` | GACODE `input.profiles` / `input.gacode` files |
| `imas_io` | `classes/imas.py` | IMAS data dictionary (NetCDF or HDF5, via `imas-python`) |
| `omas_io` | `classes/omas.py` | OMAS (Ordered Multidimensional Array Structure) |
| `torax_io` | `classes/torax.py` | TORAX (Google's JAX-based transport solver) JSON format |
| `transp_io` | `classes/transp.py` | TRANSP (Princeton) CDF/netcdf output files | incomplete
| `astra_io` | `classes/astra.py` | ASTRA transport code output | incomplete
| `jintrac_io` | `classes/jintrac.py` | JINTRAC (JET integrated modelling) binary output | incomplete

## Central Hub: `plasma_io`

`plasma_io` is the internal intermediate format. Data is stored as xarray NetCDF files with radial profiles indexed by `(time, radius, ion, source, direction)`.

**Base variables:** `density_e`, `density_i`, `temperature_e`, `temperature_i`, `velocity_i`, `magnetic_flux`, `field_axis`, `current`, `r_minor`, `r_geometric`, `z_geometric`, `contour`, plus heat/particle/momentum sources and current sources broken down by source type (ohmic, NBI, ICRH, ECRH, synchrotron, bremsstrahlung, line radiation, ionization, charge exchange, bootstrap, fusion).

**Key methods on `plasma_io`:**
- `add_geometry_from_eqdsk(path)` — reads a g-EQDSK file, traces flux surfaces using `megpy`, and populates geometry fields
- `add_safety_factor_profile(q, r)` — inserts a q profile and derives the missing magnetic flux component
- `_compute_derived_coordinates()` — computes normalized radii, aspect ratio, epsilon, volume, surface area, etc.
- `_compute_derived_reference_quantities()` — computes safety factor, magnetic shear, sound speed, gyroradius, thermal velocities
- `_compute_derived_geometry()` — computes MXH (Miller eXtended Harmonic) shape coefficients (κ, δ, ζ, sin/cos series), field components, flux-surface-averaged quantities
- `_compute_extended_local_inputs()` — computes pressures, beta, normalized gradients, collisionalities, Debye lengths, ExB shearing rate, gyro-Bohm normalizations
- `_compute_integrated_quantities()` — volume/surface integrals of sources, line/volume averages

## Utilities

| File | Purpose |
|---|---|
| `utils/eqdsk_tools.py` | EQDSK read/write, COCOS convention detection and conversion, flux surface tracing via `megpy`, MXH coefficient computation |
| `utils/math_tools.py` | Vectorized derivatives, integrals, interpolation over radial profiles (numpy-based) |
| `utils/plasma_tools.py` | Ion species definitions |
| `utils/field_definitions.py` | Physical field metadata |

## CLI Scripts

Registered as package entry points:

- `generate_tglf_runs` (`scripts/standalone_tglf.py`) — converts TORAX output → TGLF standalone input files
- `generate_qualikiz_runs` (`scripts/standalone_qualikiz.py`) — converts TORAX output → QuaLiKiz input files

## Typical Workflow

1. Load data from a simulation code (e.g. `torax_io.from_file(output=path)`)
2. Convert to the internal `plasma_io` representation via `.to('plasma')` or `plasma_io.from_torax(obj)`
3. Optionally enrich with equilibrium geometry via `add_geometry_from_eqdsk(eqdsk_path)`
4. Compute derived quantities (MXH coefficients, collisionalities, etc.)
5. Convert to target format (e.g. `.to('gacode')`, `.to('imas')`)
6. Write output via `.write(path)`

## Dependencies

- `numpy`, `scipy`, `xarray` — core numerics and data structures
- `imas-python[netcdf,xarray]` — IMAS data dictionary interface
- `omas` — OMAS interface
- `h5py` — HDF5 file support
- `contourpy`, `shapely` — contour tracing and geometry
- `megpy` — flux surface tracing from EQDSK files

## Project Info

- Author: Aaron Ho (aaronkho@mit.edu)
- License: MIT
- Python: >=3.10
- Source layout: `src/fusio/`
- Tests: `tests/`
