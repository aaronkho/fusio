import logging
import argparse
import importlib
import json
from pathlib import Path
from ..classes.io import io, Any, Final
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence
import xarray as xr

logger = logging.getLogger('fusio')


# All format classes which provide a read and / or a write function, jintrac is excluded
# since it only exposes structure creation helpers so far
ALL_FORMATS: Final[Sequence[str]] = [
    'plasma',
    'gacode',
    'imas',
    'omas',
    'torax',
    'fastran',
    'transp',
    'astra',
]

# Formats whose write function is implemented, i.e. the valid conversion targets
WRITABLE_FORMATS: Final[Sequence[str]] = [
    'plasma',
    'gacode',
    'imas',
    'torax',
    'transp',
    'fastran',
]

# Conversion functions which are present on the class but only raise, as (target, source)
STUB_CONVERSIONS: Final[Sequence[tuple[str, str]]] = [
    ('fastran', 'plasma'),
    ('transp', 'plasma'),
]

# Default extension of the written file, an empty string denotes a directory
OUTPUT_SUFFIXES: Final[Mapping[str, str]] = {
    'plasma': '.nc',
    'gacode': '.gacode',
    'imas': '',  # Directory holding one netcdf file per IDS
    'torax': '.json',
    'transp': '.nc',
    'fastran': '.nc',
}

# Side which the write function of a format draws its data from, conversion functions
# always deposit their result on the input side
WRITE_SIDES: Final[Mapping[str, str]] = {
    'fastran': 'output',
}

# Lightweight content hints for source format detection, tabulated here rather than taken
# from the format classes so that detection does not require importing all of them
IMAS_IDS_HINTS: Final[Sequence[str]] = [
    'equilibrium',
    'core_profiles',
    'core_sources',
    'core_transport',
    'summary',
    'dataset_description',
    'edge_profiles',
    'pf_active',
    'wall',
    'nbi',
    'ec_launchers',
]
TORAX_CONFIG_HINTS: Final[Sequence[str]] = [
    'profile_conditions',
    'numerics',
    'plasma_composition',
    'geometry',
    'sources',
    'transport',
    'solver',
    'pedestal',
    'time_step_calculator',
    'neoclassical',
    'mhd',
]
PLASMA_VARIABLE_HINTS: Final[Sequence[str]] = [
    'density_e',
    'temperature_e',
    'magnetic_flux',
    'r_minor',
    'r_geometric',
    'field_axis',
]
TORAX_VARIABLE_HINTS: Final[Sequence[str]] = [
    'T_e',
    'T_i',
    'n_e',
    'n_i',
    'Z_eff',
    'rho_norm',
    'rho_cell_norm',
    'rho_face_norm',
    'j_bootstrap',
]
FASTRAN_VARIABLE_HINTS: Final[Sequence[str]] = [
    'rhob',
    'volp',
    'ipol',
    'nebar',
    'betan',
    'tauth',
    'zimp',
    'aimp',
]
TRANSP_VARIABLE_HINTS: Final[Sequence[str]] = [
    'XB',
    'NE',
    'TE',
    'TI',
    'ZEFFP',
    'PPLAS',
    'VTORMP',
]

NETCDF_SUFFIXES: Final[Sequence[str]] = ['.nc', '.ncdf', '.cdf', '.h5', '.hdf5']

# The internal intermediate format, which carries the most complete mapping of every
# other format and is the only one storing the 2-D poloidal flux map
HUB_FORMAT: Final[str] = 'plasma'


def load_format_class(
    fmt: str,
) -> Any:
    """Import the io subclass which implements the given format."""
    try:
        mod = importlib.import_module(f'fusio.classes.{fmt}')
        return getattr(mod, f'{fmt}_io')
    except (ImportError, AttributeError) as err:
        raise NotImplementedError(f'Format {fmt} is not available in this fusio installation.') from err


def build_conversion_map() -> Mapping[str, Sequence[str]]:
    """Map each format onto the list of formats it can be directly converted from.

    Assembled by introspecting the from_<source> functions present on each format class,
    minus the ones which exist only as stubs, so that the map tracks the class definitions
    instead of duplicating them.
    """
    cmap: MutableMapping[str, Sequence[str]] = {}
    for target in ALL_FORMATS:
        try:
            cls = load_format_class(target)
        except NotImplementedError as err:
            logger.debug(f'{err} Omitting it from the conversion map...')
            continue
        cmap[target] = [
            source
            for source in ALL_FORMATS
            if hasattr(cls, f'from_{source}') and (target, source) not in STUB_CONVERSIONS
        ]
    return cmap


def format_conversion_map(
    cmap: Mapping[str, Sequence[str]],
) -> str:
    """Render the conversion map as a table for the command line."""
    lines = [
        'Supported direct conversions, as target <- sources:',
        '',
    ]
    width = max([len(fmt) for fmt in cmap] + [6])
    for target in sorted(cmap):
        sources = [source for source in cmap[target]]
        notes = []
        if not sources:
            notes.append('no conversions implemented yet')
        if target not in WRITABLE_FORMATS:
            notes.append('no write function, usable only as an intermediate')
        note = f'  ({", ".join(notes)})' if notes else ''
        rendered = ', '.join(sources) if sources else '-'
        lines.append(f'  {target:<{width}s} <- {rendered}{note}')
    lines += [
        '',
        f'Readable source formats: {", ".join(sorted(cmap))}',
        f'Writable target formats: {", ".join(sorted([fmt for fmt in WRITABLE_FORMATS if fmt in cmap]))}',
        '',
        'Pairs without a direct conversion need an explicit route, e.g. --via plasma, or',
        '--via auto to let the shortest available route be found.',
    ]
    return '\n'.join(lines)


def find_shortest_route(
    source: str,
    target: str,
    cmap: Mapping[str, Sequence[str]],
) -> Sequence[str] | None:
    """Find the fewest-hop conversion route between two formats, if one exists."""
    if source == target:
        return [source, target] if source in cmap.get(target, []) else [source]
    edges: MutableMapping[str, MutableSequence[str]] = {}
    for tgt in cmap:
        for src in cmap[tgt]:
            edges.setdefault(src, []).append(tgt)
    queue: MutableSequence[Sequence[str]] = [[source]]
    seen = {source}
    while queue:
        route = queue.pop(0)
        for nxt in edges.get(route[-1], []):
            if nxt == target:
                return list(route) + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(list(route) + [nxt])
    return None


def resolve_route(
    source: str,
    target: str,
    via: Sequence[str] | None,
    cmap: Mapping[str, Sequence[str]],
) -> Sequence[str]:
    """Determine the chain of formats which the data passes through.

    Only direct conversions are used unless intermediate formats are requested through
    via, which also accepts the single entry auto to search for the shortest route.
    """
    if via is not None and len(via) == 1 and f'{via[0]}' == 'auto':
        route = find_shortest_route(source, target, cmap)
        if route is None:
            raise ValueError(f'No conversion route from {source} to {target} exists, use --list to see the supported conversions.')
        return route
    route = [source] + [f'{fmt}' for fmt in via] + [target] if via else [source, target]
    for i, (src, tgt) in enumerate(zip(route[:-1], route[1:])):
        if src not in cmap.get(tgt, []):
            # A format with no conversion function from itself is simply passed through
            if src == tgt and len(route) == 2:
                return [source]
            sources = ', '.join(cmap.get(tgt, [])) if cmap.get(tgt, []) else 'nothing'
            hint = f' Conversion into {tgt} is supported from {sources}.'
            if i == 0 and len(route) == 2:
                hint += ' Add --via plasma or --via auto to route through an intermediate format.'
            raise ValueError(f'Direct conversion from {src} to {tgt} is not implemented.{hint}')
    return route


def detect_format(
    path: str | Path,
) -> str | None:
    """Guess the format of a source file or directory from its name and contents."""
    ipath = Path(path)
    name = ipath.name.lower()
    suffix = ipath.suffix.lower()

    if ipath.is_dir():
        # IMAS stores one netcdf file per IDS, or a single master file when using hdf5
        if (ipath / 'master.h5').is_file():
            return 'imas'
        if any((ipath / f'{ids}.nc').is_file() for ids in IMAS_IDS_HINTS):
            return 'imas'
        # ASTRA run directories carry the ufile and netcdf outputs in subdirectories
        if (ipath / 'ncdf_out').is_dir() or (ipath / 'udb').is_dir():
            return 'astra'
        logger.debug(f'Directory {ipath} does not match any known format layout.')
        return None

    if not ipath.is_file():
        return None

    # GACODE profiles are plain text, either input.gacode or <tag>.gacode
    if suffix in ['.gacode'] or name in ['input.gacode', 'input.profiles']:
        return 'gacode'

    if suffix in ['.json']:
        return detect_json_format(ipath)

    if suffix in NETCDF_SUFFIXES:
        return detect_netcdf_format(ipath)

    logger.debug(f'File extension {suffix} of {ipath.name} does not match any known format.')
    return None


def detect_json_format(
    path: str | Path,
) -> str | None:
    """Separate a TORAX configuration from an OMAS data structure by its top level keys."""
    try:
        with open(Path(path), 'r') as jf:
            data = json.load(jf)
    except (OSError, json.JSONDecodeError) as err:
        logger.debug(f'Could not inspect {path} as JSON: {err}')
        return None
    if not isinstance(data, dict):
        return None
    keys = [f'{key}' for key in data]
    n_torax = len([key for key in keys if key in TORAX_CONFIG_HINTS])
    n_omas = len([key for key in keys if key.split('.')[0] in IMAS_IDS_HINTS])
    if n_torax > n_omas:
        return 'torax'
    if n_omas > 0:
        return 'omas'
    return None


def detect_netcdf_format(
    path: str | Path,
) -> str | None:
    """Separate the netcdf-based formats by the names stored inside the file."""
    ipath = Path(path)
    name = ipath.name.lower()
    fmt = read_dumped_format(ipath)
    if fmt is not None:
        return fmt
    if 'fastran' in name:
        return 'fastran'
    if ipath.stem.lower() in IMAS_IDS_HINTS:
        return 'imas'
    try:
        tree = xr.open_datatree(ipath)
    except Exception as err:
        logger.debug(f'Could not inspect {ipath} as netcdf: {err}')
        return None
    with tree:
        names = []
        for group in tree.groups:
            ds = tree[group].to_dataset()
            names += [f'{key}' for key in ds.coords] + [f'{key}' for key in ds.data_vars]
        scores = {
            'plasma': len([key for key in names if key in PLASMA_VARIABLE_HINTS]),
            'torax': len([key for key in names if key in TORAX_VARIABLE_HINTS]),
            'fastran': len([key for key in names if key in FASTRAN_VARIABLE_HINTS]),
            'transp': len([key for key in names if key in TRANSP_VARIABLE_HINTS]),
            'imas': len([key for key in names if f'{key}'.split('.')[0] in IMAS_IDS_HINTS]),
        }
    best = max(scores, key=lambda key: scores[key])
    if scores[best] <= 0:
        logger.debug(f'Contents of {ipath.name} do not match any known netcdf format.')
        return None
    return best


def read_dumped_format(
    path: str | Path,
) -> str | None:
    """Return the format recorded inside a fusio dump file, if the file is one."""
    ipath = Path(path)
    if not (ipath.is_file() and ipath.suffix.lower() in NETCDF_SUFFIXES):
        return None
    try:
        tree = xr.open_datatree(ipath)
    except Exception:
        return None
    with tree:
        if 'input' not in tree.children or 'output' not in tree.children:
            return None
        fmt = tree.to_dataset().attrs.get('class', None)
    return f'{fmt}' if isinstance(fmt, str) and fmt in ALL_FORMATS else None


def read_source(
    fmt: str,
    path: str | Path,
    side: str = 'output',
) -> tuple[Any, str]:
    """Read a source file or directory into its format class.

    Returns the populated object together with the side its data ended up on, which can
    differ from the requested side when reading a fusio dump file.
    """
    ipath = Path(path)
    if read_dumped_format(ipath) == fmt:
        obj = io.load(ipath)
        newside = side
        if side == 'output' and not obj.has_output and obj.has_input:
            newside = 'input'
        if side == 'input' and not obj.has_input and obj.has_output:
            newside = 'output'
        logger.info(f'Loaded dumped {fmt} data from {ipath.name}, using its {newside} side')
        return obj, newside
    cls = load_format_class(fmt)
    if fmt == 'astra':
        # ASTRA spreads its data over subdirectories of a run directory
        return cls.from_rundir(path=ipath), side
    kwargs: MutableMapping[str, Any] = {'input': ipath} if side == 'input' else {'output': ipath}
    return cls.from_file(**kwargs), side


def select_eqdsk_hop(
    route: Sequence[str],
) -> str | None:
    """Choose which format along the route the equilibrium is folded into.

    The hub is preferred when the route passes through it, since it is the only format
    which stores the traced 2-D flux map that the later conversions carry onwards.
    """
    capable = []
    for fmt in route:
        try:
            cls = load_format_class(fmt)
        except NotImplementedError:
            continue
        if hasattr(cls, 'add_geometry_from_eqdsk') and fmt not in capable:
            capable.append(fmt)
    if not capable:
        return None
    return HUB_FORMAT if HUB_FORMAT in capable else capable[0]


def add_eqdsk_geometry(
    obj: Any,
    path: str | Path,
    side: str,
) -> bool:
    """Fold an equilibrium from a g-EQDSK file into the data, where the format allows it."""
    if not hasattr(obj, 'add_geometry_from_eqdsk'):
        return False
    obj.add_geometry_from_eqdsk(path, side=side, overwrite=True)
    logger.info(f'Added geometry from {Path(path).name} into the {side} side of the {obj.format} data')
    return True


def compute_derived_quantities(
    obj: Any,
    side: str,
) -> None:
    """Fill in the quantities which the format class can derive from what it already holds."""
    if not hasattr(obj, 'compute_derived_quantities'):
        logger.debug(f'{obj.format} does not define derived quantities, skipping derivation...')
        return
    try:
        obj.compute_derived_quantities(side=side)
        logger.info(f'Computed derived quantities on the {side} side of the {obj.format} data')
    except Exception as err:
        logger.warning(f'Could not compute derived quantities for the {obj.format} data: {err}')


def convert_step(
    obj: Any,
    target: str,
    side: str,
    **kwargs: Any,
) -> Any:
    """Apply a single from_<source> conversion, placing the result on the input side."""
    cls = load_format_class(target)
    generator = getattr(cls, f'from_{obj.format}', None)
    if generator is None:
        raise NotImplementedError(f'Direct conversion from {obj.format} to {target} is not implemented.')
    newobj = generator(obj, side=side, **{key: val for key, val in kwargs.items() if val is not None})
    if newobj is None or newobj.is_empty:
        raise ValueError(f'Conversion from {obj.format} to {target} produced no data.')
    logger.info(f'Converted {obj.format} data into {target}')
    return newobj


def align_write_side(
    obj: Any,
    target: str,
    side: str,
) -> str:
    """Move the data onto the side which the target format writes from, if they differ."""
    want = WRITE_SIDES.get(target, side)
    if want == side:
        return side
    if want == 'output':
        obj.output = obj.input
    else:
        obj.input = obj.output
    logger.debug(f'Copied {target} data onto its {want} side for writing')
    return want


def resolve_output_path(
    output: str | Path | None,
    ipath: Path,
    target: str,
    total: int,
) -> Path:
    """Determine where the converted data is written.

    A given output path is used as-is for a single input, and as a containing directory
    when several inputs are converted in one call.
    """
    suffix = OUTPUT_SUFFIXES.get(target, '')
    default_name = f'{ipath.stem}_{target}{suffix}'
    if output is None:
        return ipath.parent / default_name
    opath = Path(output)
    if total > 1:
        return opath / default_name
    return opath


def convert_file(
    ipath: Path,
    target: str,
    source: str | None = None,
    via: Sequence[str] | None = None,
    output: str | Path | None = None,
    total: int = 1,
    read_side: str = 'output',
    eqdsk: str | Path | None = None,
    window: Sequence[float] | None = None,
    time: float | None = None,
    impurities: str | None = None,
    derive: bool = True,
    overwrite: bool = False,
    cmap: Mapping[str, Sequence[str]] | None = None,
) -> Path | None:
    """Convert a single source file or directory into the target format and write it out."""
    conversion_map = cmap if cmap is not None else build_conversion_map()

    fmt = source if source is not None else detect_format(ipath)
    if fmt is None:
        raise ValueError(f'Could not determine the format of {ipath}, pass it explicitly with --format.')
    if source is None:
        logger.info(f'Detected {fmt} format for {ipath.name}')

    route = resolve_route(fmt, target, via, conversion_map)
    if len(route) > 2:
        logger.info(f'Routing {fmt} into {target} through {", ".join(route[1:-1])}')

    # Checked up front since the write functions only warn, and the conversion itself can
    # take a while to reach the point where the warning appears
    opath = resolve_output_path(output, ipath, target, total)
    if opath.exists() and not overwrite:
        raise OSError(f'Requested output path, {opath}, already exists! Pass --overwrite to replace it.')

    obj, side = read_source(fmt, ipath, read_side)
    if obj is None or obj.is_empty:
        raise ValueError(f'Reading {ipath} as {fmt} produced no data.')

    # Geometry is folded in once, at the hop chosen below, so that every subsequent
    # conversion and derivation along the route sees the equilibrium
    eqdsk_hop = select_eqdsk_hop(route) if eqdsk is not None else None
    eqdsk_added = False
    if eqdsk is not None and obj.format == eqdsk_hop:
        eqdsk_added = add_eqdsk_geometry(obj, eqdsk, side)
    if derive:
        compute_derived_quantities(obj, side)

    for nxt in route[1:]:
        obj = convert_step(
            obj,
            nxt,
            side,
            window=window,
            time=time,
            impurities=impurities,
        )
        side = 'input'  # Conversion functions always deposit onto the input side
        if eqdsk is not None and not eqdsk_added and obj.format == eqdsk_hop:
            eqdsk_added = add_eqdsk_geometry(obj, eqdsk, side)
        if derive:
            compute_derived_quantities(obj, side)

    if eqdsk is not None and not eqdsk_added:
        logger.warning(f'None of the formats in {" -> ".join(route)} can take geometry from a g-EQDSK file, ignoring {Path(eqdsk).name}...')

    opath.parent.mkdir(parents=True, exist_ok=True)
    side = align_write_side(obj, target, side)
    obj.write(opath, side=side, overwrite=overwrite)
    if not opath.exists():
        raise OSError(f'Writing the {target} data to {opath} produced no output!')
    return opath


def parse_args():
    parser = argparse.ArgumentParser(
        prog='fusio_convert',
        description='Convert plasma simulation input and output files between the formats supported by fusio.',
        epilog='Use --list to print the conversions which are currently implemented.',
    )
    parser.add_argument('ifiles', type=str, nargs='*', help='Paths to the input files or directories to be converted')
    parser.add_argument('-t', '--target', type=str, default=None, help='Format to convert the input files into')
    parser.add_argument('-f', '--format', type=str, default=None, help='Format of the input files, detected from their name and contents when omitted')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path of the converted output, treated as a containing directory when multiple inputs are given')
    parser.add_argument('--via', type=str, nargs='*', default=None, help='Intermediate formats to route the conversion through, or auto to use the shortest available route')
    parser.add_argument('-e', '--eqdsk', type=str, default=None, help='Path to a g-EQDSK file whose equilibrium is folded into the route, at the plasma hub where the route passes through it')
    parser.add_argument('-w', '--window', type=float, nargs='*', default=None, help='Time window, given as a minimum and a maximum, to select from transient input data')
    parser.add_argument('--time', type=float, default=None, help='Time stamp to assign when converting from a format which carries no plasma time')
    parser.add_argument('-i', '--impurities', type=str, default=None, choices=['n_e_ratios', 'fractions'], help='Impurity representation to use when the target format offers a choice')
    parser.add_argument('-r', '--read-side', type=str, default='output', choices=['input', 'output'], help='Side of the source format to take the data from, default is output')
    parser.add_argument('--no-derive', action='store_true', default=False, help='Skip the derived quantity computations which are otherwise run on every format which defines them')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite the converted output if it already exists')
    parser.add_argument('--list', action='store_true', default=False, help='Print the supported conversions and exit')
    parser.add_argument('--quiet', action='store_true', default=False, help='Only report warnings and errors')
    parser.add_argument('--debug', action='store_true', default=False, help='Report the detection and conversion steps in detail')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    level = logging.INFO
    if args.quiet:
        level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(level)

    cmap = build_conversion_map()

    if args.list:
        print(format_conversion_map(cmap))
        return 0

    if not args.ifiles:
        logger.error('No input files given! Pass at least one path, or use --list to see the supported conversions.')
        return 2

    if args.target is None:
        logger.error('No target format given! Pass one with --target, or use --list to see the supported conversions.')
        return 2

    if args.target not in WRITABLE_FORMATS:
        supported = ', '.join([fmt for fmt in WRITABLE_FORMATS if fmt in cmap])
        logger.error(f'Target format {args.target} has no write function in fusio! Supported targets are {supported}.')
        return 2

    if args.format is not None and args.format not in cmap:
        logger.error(f'Source format {args.format} is not recognized! Supported formats are {", ".join(sorted(cmap))}.')
        return 2

    ipaths = [Path(f'{ip}').resolve() for ip in args.ifiles if Path(f'{ip}').exists()]
    for ip in args.ifiles:
        if not Path(f'{ip}').exists():
            logger.error(f'Input path {ip} does not exist, skipping...')

    epath = None
    if args.eqdsk is not None:
        epath = Path(f'{args.eqdsk}').resolve()
        if not epath.is_file():
            logger.error(f'Requested EQDSK file, {epath}, does not exist!')
            return 2

    n_converted = 0
    for ipath in ipaths:
        try:
            opath = convert_file(
                ipath,
                args.target,
                source=args.format,
                via=args.via,
                output=args.output,
                total=len(ipaths),
                read_side=args.read_side,
                eqdsk=epath,
                window=args.window,
                time=args.time,
                impurities=args.impurities,
                derive=(not args.no_derive),
                overwrite=args.overwrite,
                cmap=cmap,
            )
        except (ValueError, NotImplementedError, KeyError, OSError) as err:
            logger.error(f'Conversion of {ipath.name} failed: {err}')
            continue
        if opath is not None:
            n_converted += 1

    logger.info(f'Converted {n_converted} of {len(ipaths)} inputs into {args.target}')
    return 0 if n_converted == len(ipaths) and n_converted > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
