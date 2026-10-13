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
| `fastran_io` | `classes/fastran.py` | FASTRAN (Oak Ridge National Laboratory / IPS) `fastran.nc` output files | output only
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

- `fusio_convert` (`scripts/convert.py`) — generic class-to-class format conversion
- `generate_tglf_runs` (`scripts/standalone_tglf.py`) — TGLF standalone input files
- `generate_qualikiz_runs` (`scripts/standalone_qualikiz.py`) — QuaLiKiz input files
- `generate_cgyro_runs` (`scripts/standalone_cgyro.py`) — CGYRO input files

Shared plumbing for the three run generators lives in `scripts/standalone.py`.

### `fusio_convert`

Mirrors `obj.to(fmt)` on the command line. The conversion matrix is built by introspecting
the `from_<source>` methods on each format class, so it tracks the classes rather than
duplicating them; `--list` prints it.

```bash
fusio_convert input.gacode -t plasma -o state.nc        # source format is auto-detected
fusio_convert out.nc -f torax -t gacode -o input.gacode
fusio_convert imas_dir/ -t torax --via plasma gacode    # multi-hop, see below
fusio_convert state.nc -t gacode -e equil.geqdsk        # fold in an equilibrium
fusio_convert --list
```

Key behaviours:

- **Source detection** — inferred from path and contents when `-f` is omitted: suffix for
  `.gacode`, top-level keys for `.json` (TORAX config vs. OMAS), variable names for `.nc`,
  IDS filenames for IMAS/ASTRA directories. Files produced by `io.dump()` are recognized by
  their root `class` attribute and loaded through `io.load()`.
- **Routing** — direct `from_<source>` edges only, unless `--via <fmt> ...` names
  intermediates or `--via auto` searches for the shortest route. `source == target` with no
  self-conversion is a passthrough, which makes `--eqdsk`/derivation-only runs possible.
- **Derived quantities** — `compute_derived_quantities()` runs at every hop whose class
  defines it (`plasma_io`, `gacode_io`); failures warn rather than abort, since derivation is
  enrichment. `--no-derive` opts out.
- **Sides** — conversion methods always deposit onto the target's `input` side, so reads
  default to `output` (`--read-side`) and writes to `input`.
- Valid targets are the formats with a working `write()`: `plasma`, `gacode`, `imas`, `torax`.
  `omas_io._write_omas_json_file` is a no-op and `astra_io.write` only warns, so those are
  intermediates only; `fastran_io.from_plasma` and `transp_io.from_plasma` are stubs that
  raise, and are excluded from the conversion map via `STUB_CONVERSIONS`.

### Standalone run generators

Each generator reads a source file, routes it to a format implementing
`to_<code>_parameters()`, then expands the result into one run directory per operating
point. The producer map is introspected the same way the conversion map is, so defining a
new `to_<code>_parameters()` makes it available to the CLI; `--list` prints it.

```bash
generate_cgyro_runs out.nc -f torax -r 0.4 0.6 -o ./cgyro_runs
generate_cgyro_runs input.gacode -e equil.geqdsk -r 0.5   # routed gacode → plasma → imas
generate_tglf_runs torax_output.nc -t 2.0 -r 0.5 --cell-grid
generate_qualikiz_runs torax_output.nc -x /path/to/qualikiz
```

Who can feed what:

| Code | Native producers | Reachable by conversion |
|---|---|---|
| `tglf` | `torax` | — |
| `qualikiz` | `torax` | — |
| `cgyro` | `torax`, `imas` | `plasma`, `gacode`, `omas`, `fastran`, `astra` |

- **`NATIVE_ONLY_PRODUCERS`** — `torax_io.to_*_parameters()` read TORAX *output* profiles
  from `self.output`, but `torax_io.from_gacode`/`from_omas` build a TORAX *configuration*
  on the input side. A converted `torax_io` therefore yields nothing, so `torax` is marked
  native-only: it is never used as a conversion target, and asking for it via `--via` is a
  hard error rather than a silently empty run set.
- **Hub-preferred routing** — there is no target format to name, so a route to a producer is
  chosen automatically. `standalone.route_to_producer()` prefers a route through `plasma`
  over a shorter direct edge, because the direct cross-conversions cover less: `gacode →
  imas` fills only `equilibrium`, while `gacode → plasma → imas` also fills `core_profiles`
  and `core_sources`. `--via` overrides.
- **EQDSK placement** — `convert.select_eqdsk_hop()` folds the equilibrium in at `plasma`
  when the route passes through it, otherwise at the first capable format. This matters for
  CGYRO: `imas_io.to_cgyro_parameters()` traces MXH coefficients through `to_eqdsk()`, which
  needs the rectangular `profiles_2d` grid that `imas_io.from_plasma` only writes when
  `plasma_io` carries a `poloidal_flux_map` — i.e. only after `add_geometry_from_eqdsk`.
- **Plan shapes** — `build_pointwise_plan()` for the local codes (TGLF, CGYRO: one run per
  time and radius), `build_profile_plan()` for the radial-scan codes (QuaLiKiz: one run per
  time, arrays over rho). CGYRO uses `overlay_all=True` so the variable-length MXH shape
  series (`SHAPE_SIN3`, `SHAPE_S_COS4`, …) survives; TGLF and QuaLiKiz keep the original
  behaviour of writing only keys named in their templates.
- **Producer kwargs** — `call_parameter_generator()` reads the signature of
  `to_<code>_parameters` and passes only what it accepts, since the producers differ
  (`lump_impurities`/`use_cell_grid` on `torax_io`, `side`/`n_mxh_moments`/`n_fine` on
  `imas_io`).
- Run directories are regenerated artifacts, so an existing input file is replaced, matching
  the original scripts' behaviour.

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
