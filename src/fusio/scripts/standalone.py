import copy
import inspect
import logging
from pathlib import Path
from ..classes.io import Any, Final
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence
from numpy.typing import NDArray
import numpy as np
import xarray as xr

from . import convert

logger = logging.getLogger('fusio')


# Gyrokinetic and transport codes which fusio can generate standalone inputs for
SUPPORTED_CODES: Final[Sequence[str]] = [
    'tglf',
    'qualikiz',
    'cgyro',
]

# Producers which build their parameters out of simulation output profiles. A conversion
# into these formats yields a run configuration rather than output profiles, so they can
# only be used on a natively read file of that format, never as a conversion target
NATIVE_ONLY_PRODUCERS: Final[Sequence[str]] = [
    'torax',
]

# Parameters which have to be written as integers rather than floats
INTEGER_PARAMETERS: Final[Sequence[str]] = [
    'NS',
    'N_SPECIES',
    'nion',
]


def build_producer_map() -> Mapping[str, Sequence[str]]:
    """Map each supported code onto the formats which can compute its input parameters.

    Built by introspecting the to_<code>_parameters functions on each format class, so
    that a new producer becomes available to the run generators as soon as it is defined.
    """
    pmap: MutableMapping[str, Sequence[str]] = {}
    for code in SUPPORTED_CODES:
        producers: MutableSequence[str] = []
        for fmt in convert.ALL_FORMATS:
            try:
                cls = convert.load_format_class(fmt)
            except NotImplementedError:
                continue
            if hasattr(cls, f'to_{code}_parameters'):
                producers.append(fmt)
        pmap[code] = producers
    return pmap


def format_producer_map(
    pmap: Mapping[str, Sequence[str]],
    cmap: Mapping[str, Sequence[str]],
) -> str:
    """Render the formats which can feed each code, directly or through a conversion."""
    lines = ['Formats which can generate standalone code inputs:', '']
    width = max([len(code) for code in pmap] + [8])
    native_only = False
    for code in sorted(pmap):
        producers = [fmt for fmt in pmap[code]]
        reachable = []
        for fmt in sorted(cmap):
            if fmt in producers:
                continue
            if resolve_producer_route(fmt, producers, cmap, None) is not None:
                reachable.append(fmt)
        rendered = ', '.join(producers) if producers else '-'
        lines.append(f'  {code:<{width}s} <- {rendered}')
        if reachable:
            lines.append(f'  {"":<{width}s}    via conversion: {", ".join(reachable)}')
        native_only = native_only or any(fmt in NATIVE_ONLY_PRODUCERS for fmt in producers)
    if native_only:
        listed = ', '.join([fmt for fmt in NATIVE_ONLY_PRODUCERS])
        lines += [
            '',
            f'Parameters from {listed} are computed out of simulation output profiles. Converting',
            f'into {listed} yields a run configuration instead, so those sources have to be read directly.',
        ]
    return '\n'.join(lines)


def resolve_producer_route(
    source: str,
    producers: Sequence[str],
    cmap: Mapping[str, Sequence[str]],
    via: Sequence[str] | None,
) -> Sequence[str] | None:
    """Find the chain of formats leading from the source to one which computes parameters.

    Unlike a format to format conversion there is no target format to name, so the
    shortest route to any producer is taken unless via spells one out.
    """
    if source in producers:
        return [source]
    if via:
        route = [source] + [f'{fmt}' for fmt in via]
        for src, tgt in zip(route[:-1], route[1:]):
            if src not in cmap.get(tgt, []):
                raise ValueError(f'Direct conversion from {src} to {tgt} is not implemented.')
        if route[-1] not in producers:
            raise ValueError(f'Format {route[-1]} cannot compute these parameters, it has to be one of {", ".join(producers)}.')
        if route[-1] in NATIVE_ONLY_PRODUCERS:
            raise ValueError(
                f'Converting into {route[-1]} yields a run configuration rather than output profiles, '
                f'so these parameters can only be computed from a {route[-1]} file read directly.'
            )
        return route
    best: Sequence[str] | None = None
    for producer in producers:
        if producer in NATIVE_ONLY_PRODUCERS:
            continue
        candidate = route_to_producer(source, producer, cmap)
        if candidate is None:
            continue
        if best is None or len(candidate) < len(best):
            best = candidate
    return best


def route_to_producer(
    source: str,
    producer: str,
    cmap: Mapping[str, Sequence[str]],
) -> Sequence[str] | None:
    """Route from a source format to a producer, preferring the internal hub.

    The direct cross-format conversions cover less than the ones going through the hub,
    e.g. imas_io.from_gacode only fills the equilibrium while imas_io.from_plasma also
    fills the profiles and sources, so a longer route through the hub is the better one.
    """
    hub = convert.HUB_FORMAT
    if source != hub and producer != hub:
        head = convert.find_shortest_route(source, hub, cmap)
        tail = convert.find_shortest_route(hub, producer, cmap)
        if head is not None and tail is not None and len(head) >= 2 and len(tail) >= 2:
            return list(head) + list(tail)[1:]
    direct = convert.find_shortest_route(source, producer, cmap)
    return direct if direct is not None and len(direct) >= 2 else None


def call_parameter_generator(
    obj: Any,
    code: str,
    side: str,
    **kwargs: Any,
) -> xr.Dataset:
    """Invoke to_<code>_parameters, passing only the arguments its signature accepts.

    The producers do not share a signature, e.g. only the IMAS one takes a side and the
    contour tracing controls, so the accepted names are read off the function itself.
    """
    generator = getattr(obj, f'to_{code}_parameters', None)
    if generator is None:
        raise NotImplementedError(f'Format {obj.format} cannot compute {code} parameters.')
    accepted = inspect.signature(generator).parameters
    passed = {key: val for key, val in kwargs.items() if key in accepted and val is not None}
    if 'side' in accepted:
        passed['side'] = side
    elif side != 'output':
        # The producers without a side argument always read their output side
        logger.debug(f'{obj.format} computes {code} parameters from its output side, moving the data there...')
        obj.output = obj.input
    return generator(**passed)


def load_code_parameters(
    ipath: str | Path,
    code: str,
    source: str | None = None,
    via: Sequence[str] | None = None,
    read_side: str = 'output',
    eqdsk: str | Path | None = None,
    derive: bool = True,
    cmap: Mapping[str, Sequence[str]] | None = None,
    pmap: Mapping[str, Sequence[str]] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Read a source file and compute the local input parameters for the given code."""
    conversion_map = cmap if cmap is not None else convert.build_conversion_map()
    producer_map = pmap if pmap is not None else build_producer_map()
    producers = producer_map.get(code, [])
    if not producers:
        raise NotImplementedError(f'No format in fusio can compute {code} parameters.')

    path = Path(ipath)
    fmt = source if source is not None else convert.detect_format(path)
    if fmt is None:
        raise ValueError(f'Could not determine the format of {path}, pass it explicitly with --format.')
    if source is None:
        logger.info(f'Detected {fmt} format for {path.name}')

    route = resolve_producer_route(fmt, producers, conversion_map, via)
    if route is None:
        raise ValueError(f'Format {fmt} cannot reach any of the {code} parameter producers, which are {", ".join(producers)}.')
    if len(route) > 1:
        logger.info(f'Routing {" -> ".join(route)} to compute the {code} parameters')

    obj, side = convert.read_source(fmt, path, read_side)
    if obj is None or obj.is_empty:
        raise ValueError(f'Reading {path} as {fmt} produced no data.')

    eqdsk_hop = convert.select_eqdsk_hop(route) if eqdsk is not None else None
    eqdsk_added = False
    if eqdsk is not None and obj.format == eqdsk_hop:
        eqdsk_added = convert.add_eqdsk_geometry(obj, eqdsk, side)
    if derive:
        convert.compute_derived_quantities(obj, side)

    for nxt in route[1:]:
        obj = convert.convert_step(obj, nxt, side)
        side = 'input'
        if eqdsk is not None and not eqdsk_added and obj.format == eqdsk_hop:
            eqdsk_added = convert.add_eqdsk_geometry(obj, eqdsk, side)
        if derive:
            convert.compute_derived_quantities(obj, side)

    if eqdsk is not None and not eqdsk_added:
        logger.warning(f'None of the formats in {" -> ".join(route)} can take geometry from a g-EQDSK file, ignoring {Path(eqdsk).name}...')

    params = call_parameter_generator(obj, code, side, **kwargs)
    if not isinstance(params, xr.Dataset) or not params.data_vars:
        raise ValueError(
            f'Computing {code} parameters from the {route[-1]} data produced nothing, '
            f'the source may not carry the profiles these parameters are built from.'
        )
    return params


def build_pointwise_plan(
    params: xr.Dataset,
    code: str,
    template: Mapping[str, Any] | None = None,
    species_template: Mapping[str, Any] | None = None,
    species_count: str | None = None,
    species_offset: int = 1,
    species_separator: str = '_',
    overlay_all: bool = False,
) -> Sequence[Any]:
    """Expand a parameter set into one run per time and radial position.

    Used by the local codes, TGLF and CGYRO, which solve at a single flux surface.
    """
    plan: MutableSequence[Any] = []
    if 'time' not in params.coords or 'rho' not in params.coords:
        return plan
    for j in range(params.sizes['time']):
        for i in range(params.sizes['rho']):
            runpars: MutableMapping[str, Any] = {key: val for key, val in template.items()} if template else {}
            point = params.isel(time=j, drop=True).isel(rho=i, drop=True)
            if species_template and species_count is not None and species_count in point:
                for s in range(int(point[species_count].to_numpy())):
                    runpars.update({f'{key}{species_separator}{s + species_offset:d}': val for key, val in species_template.items()})
            for key in point.data_vars:
                name = f'{key}'
                if not overlay_all and name not in runpars:
                    continue
                value = point[name].to_numpy()
                runpars[name] = int(value) if name in INTEGER_PARAMETERS else float(value)
            for key in [key for key in runpars if runpars[key] is None]:
                runpars.pop(key)
            plan.append({
                'inputs': copy.deepcopy(runpars),
                'location': f'{code}_run_t{j:05d}_r{i:04d}',
            })
    return plan


def build_profile_plan(
    params: xr.Dataset,
    code: str,
    template: Mapping[str, Any] | None = None,
    species_template: Mapping[str, Any] | None = None,
    species_count: str | None = None,
    species_offset: int = 0,
    species_separator: str = '',
    field_conversion_key: str = '#Bunit_by_Bo',
) -> Sequence[Any]:
    """Expand a parameter set into one run per time, each scanning over the radial grid.

    Used by the codes which take a radial profile per run rather than a single flux
    surface, i.e. QuaLiKiz.
    """
    plan: MutableSequence[Any] = []
    if 'time' not in params.coords:
        return plan
    for j in range(params.sizes['time']):
        runpars: MutableMapping[str, Any] = {key: val for key, val in template.items()} if template else {}
        slab = params.isel(time=j, drop=True)
        if species_template and species_count is not None and species_count in slab:
            for s in range(int(slab[species_count].to_numpy())):
                runpars.update({f'{key}{species_separator}{s + species_offset:d}': val for key, val in species_template.items()})
        for key in slab.data_vars:
            name = f'{key}'
            if name in INTEGER_PARAMETERS:
                runpars[name] = int(slab[name].to_numpy())
            elif name in runpars:
                runpars[name] = slab[name].to_numpy().flatten()
        if 'rho' in params.coords:
            runpars['rho'] = params['rho'].to_numpy().flatten()
        for key in [key for key in runpars if runpars[key] is None]:
            runpars.pop(key)
        num_species = runpars.pop(species_count, 1) if species_count is not None else 1
        conversion = runpars.pop(field_conversion_key, np.ones_like(runpars.get('rho', np.array([1.0]))))
        plan.append({
            'inputs': copy.deepcopy(runpars),
            'location': f'{code}_run_t{j:05d}',
            'nion': int(num_species),
            'field_conversion': copy.deepcopy(conversion),
        })
    return plan


def write_input_files(
    plan: Sequence[Any],
    run_path: str | Path,
    filename: str,
) -> int:
    """Write one run directory per entry of the plan, each holding a key value input file.

    Run directories are regenerated artifacts, so an existing input file is replaced, as
    the original standalone generators did.
    """
    runpath = Path(f'{run_path}')
    runpath.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for run in plan:
        rpath = runpath / run['location']
        ipath = rpath / filename
        if ipath.exists():
            logger.debug(f'Replacing the existing {ipath.resolve()}...')
        rpath.mkdir(parents=True, exist_ok=True)
        fstr = '\n'.join([f'{key}={val}' for key, val in run['inputs'].items()])
        with open(ipath, 'w') as f:
            f.write(fstr)
        n_written += 1

    logger.info(f'Created {n_written} {filename} files under {runpath.resolve()}')
    return n_written


def add_common_arguments(
    parser: Any,
    code: str,
) -> Any:
    """Attach the arguments which every standalone run generator shares."""
    parser.add_argument('ifiles', type=str, nargs='*', help=f'Paths to the input files or directories to generate {code.upper()} runs from')
    parser.add_argument('-f', '--format', type=str, default=None, help='Format of the input files, detected from their name and contents when omitted')
    parser.add_argument('-o', '--output', type=str, default=f'./{code}_runs', help=f'Desired path to contain the created {code.upper()} run directories')
    parser.add_argument('-t', '--time', type=float, nargs='*', default=None, help=f'Time slices from which to generate {code.upper()} runs')
    parser.add_argument('-r', '--rho', type=float, nargs='*', default=None, help=f'Radial slices from which to generate {code.upper()} runs')
    parser.add_argument('-i', '--impurities', action='store_true', default=False, help=f'Toggle full impurity definition in the generated {code.upper()} runs, default is lumped')
    parser.add_argument('-l', '--lump-impurities', action='store_true', default=False, help='Lump the impurities into a single effective species, where the producer supports it')
    parser.add_argument('-e', '--eqdsk', type=str, default=None, help='Path to a g-EQDSK file whose equilibrium is folded into the route, at the plasma hub where the route passes through it')
    parser.add_argument('--via', type=str, nargs='*', default=None, help='Intermediate formats to route the source through before the parameters are computed')
    parser.add_argument('--read-side', type=str, default='output', choices=['input', 'output'], help='Side of the source format to take the data from, default is output')
    parser.add_argument('--cell-grid', action='store_true', default=False, help='Use the cell centred radial grid instead of the face grid, where the producer supports it')
    parser.add_argument('--no-derive', action='store_true', default=False, help='Skip the derived quantity computations which are otherwise run on every format which defines them')
    parser.add_argument('--list', action='store_true', default=False, help='Print the formats which can feed this code and exit')
    parser.add_argument('--quiet', action='store_true', default=False, help='Only report warnings and errors')
    parser.add_argument('--debug', action='store_true', default=False, help='Report the detection and conversion steps in detail')
    return parser


def configure_logging(
    quiet: bool = False,
    debug: bool = False,
) -> None:
    """Send the fusio log to the console at a level chosen on the command line."""
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    if debug:
        level = logging.DEBUG
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(level)


def resolve_input_paths(
    ifiles: Sequence[str],
) -> Sequence[Path]:
    """Keep the paths which exist, reporting the ones which do not."""
    ipaths = []
    for ip in ifiles:
        path = Path(f'{ip}')
        if path.exists():
            ipaths.append(path.resolve())
        else:
            logger.error(f'Input path {ip} does not exist, skipping...')
    return ipaths


def parameter_kwargs(
    args: Any,
) -> Mapping[str, Any]:
    """Collect the producer arguments which the command line exposes."""
    return {
        'time': args.time,
        'rho': args.rho,
        'full_impurities': args.impurities,
        'lump_impurities': args.lump_impurities,
        'use_cell_grid': args.cell_grid,
    }
