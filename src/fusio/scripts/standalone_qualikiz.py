import logging
import argparse
from pathlib import Path
from ..classes.io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np

from . import convert
from . import standalone

logger = logging.getLogger('fusio')


QUALIKIZ_ION_TEMPLATE: Final[Mapping[str, Any]] = {
    'Ti': None,
    'ni': None,
    'Ati': None,
    'Ani': None,
    'Zi': None,
    'Ai': None,
}

QUALIKIZ_INPUT_TEMPLATE: Final[Mapping[str, Any]] = {
    'Ro': None,
    'Rmin': None,
    'Bo': None,
    'x': None,
    'rho': None,
    'q': None,
    'smag': None,
    'alpha': None,
    'Te': None,
    'ne': None,
    'Ate': None,
    'Ane': None,
    'nion': None,
    '#Bunit_by_Bo': None,
}


def convert_to_qualikiz_input(
    input_path: str | Path,
    time: float | Sequence[float] | NDArray | None = None,
    rho: float | Sequence[float] | NDArray | None = None,
    full_impurities: bool = False,
    source: str | None = None,
    via: Sequence[str] | None = None,
    eqdsk: str | Path | None = None,
    read_side: str = 'output',
    derive: bool = True,
    **kwargs: Any,
) -> Sequence[Any]:
    """Compute QuaLiKiz input parameters from any source format which can reach a producer."""
    ipath = Path(input_path)
    params = standalone.load_code_parameters(
        ipath,
        'qualikiz',
        source=source,
        via=via,
        read_side=read_side,
        eqdsk=eqdsk,
        derive=derive,
        time=time,
        rho=rho,
        full_impurities=full_impurities,
        **kwargs,
    )
    qualikiz_plan = standalone.build_profile_plan(
        params,
        'qualikiz',
        template=QUALIKIZ_INPUT_TEMPLATE,
        species_template=QUALIKIZ_ION_TEMPLATE,
        species_count='nion',
        species_offset=0,
        species_separator='',
    )
    logger.info(f'Generated {len(qualikiz_plan)} QuaLiKiz input sets from {ipath.name}')
    return qualikiz_plan


def convert_torax_to_qualikiz_input(
    torax_output_path: str | Path,
    time: float | Sequence[float] | NDArray | None = None,
    rho: float | Sequence[float] | NDArray | None = None,
    full_impurities: bool = False,
) -> Sequence[Any]:
    """Compute QuaLiKiz input parameters from a TORAX output file."""
    return convert_to_qualikiz_input(torax_output_path, time, rho, full_impurities, source='torax')


def generate_qualikiz_run_directories(
    qualikiz_plan: Sequence[Any],
    executable_path: str | Path,
    qualikiz_run_path: str | Path,
) -> None:

    try:
        import qualikiz_tools.qualikiz_io.inputfiles as qualikiz_inputtools  # type: ignore[import-untyped, import-not-found]
        import qualikiz_tools.qualikiz_io.qualikizrun as qualikiz_runtools  # type: ignore[import-untyped, import-not-found]
    except ImportError:
        logger.error(f'Critical Python package, qualikiz-pythontools, not found in environment. QuaLiKiz run directory generation aborted!')
        return None

    execpath = Path(f'{executable_path}')
    runpath = Path(f'{qualikiz_run_path}')
    runpath.mkdir(parents=True, exist_ok=True)

    for run_input in qualikiz_plan:
        meta = qualikiz_inputtools.QuaLiKizXpoint.Meta(
            maxpts=5e6,
            numsols=2,
            separateflux=True,
            phys_meth=1,
            rhomin=0.0,
            rhomax=1.0,
            maxruns=10,
        )
        options = qualikiz_inputtools.QuaLiKizXpoint.Options(
            recalc_Nustar=False,
        )
        # wavenumber grid
        kthetarhos = [
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.175,
            0.25,
            0.325,
            0.4,
            0.5,
            0.7,
            1.0,
            1.8,
            3.0,
            9.0,
            15.0,
            21.0,
            27.0,
            36.0,
            45.0,
        ]
        # magnetic geometry and rotation
        geometry = qualikiz_inputtools.QuaLiKizXpoint.Geometry(
            x=0.5,  # will be scan variable
            rho=0.5,  # will be scan variable
            Ro=float(run_input['inputs'].pop('Ro').mean()),
            Rmin=float(run_input['inputs'].pop('Rmin').mean()),
            Bo=float(run_input['inputs'].pop('Bo').mean()),
            q=2,  # will be scan variable
            smag=1,  # will be scan variable
            alpha=0,  # will be scan variable
            Machtor=0,
            Autor=0,
            Machpar=0,
            Aupar=0,
            gammaE=0,
        )
        elec = qualikiz_inputtools.Electron(
            T=5,  # will be scan variable
            n=1,  # will be scan variable
            At=0,  # will be scan variable
            An=0,  # will be scan variable
            type=1,
            anis=1,
            danisdr=0,
        )
        ion_list = []
        for i in range(run_input.get('nion', 0)):
            ion_list.append(qualikiz_inputtools.Ion(
                T=1,  # will be scan variable
                n=1 if i == 0 else 0,  # will be scan variable
                At=0,  # will be scan variable
                An=0,  # will be scan variable
                type=1,
                anis=1,
                danisdr=0,
                A=2 if i == 0 else 20,  # Will be a scan variable
                Z=1 if i == 0 else 10,  # will be a scan variable
            ))
        ions = qualikiz_inputtools.IonList(*ion_list)
        xpoint_base = qualikiz_inputtools.QuaLiKizXpoint(
            kthetarhos=kthetarhos,
            electrons=elec,
            ions=ions,
            **geometry,
            **meta,
            **options,
        )
        plan = qualikiz_inputtools.QuaLiKizPlan(
            scan_dict=run_input['inputs'], scan_type='parallel', xpoint_base=xpoint_base
        )
        rpath = runpath / run_input['location']
        run = qualikiz_runtools.QuaLiKizRun(
            parent_dir=str(rpath.parent.resolve()),
            binaryrelpath=str(execpath.resolve()),
            name=str(rpath.name),
            qualikiz_plan=plan,
            verbose=0,
        )
        run.prepare(
            overwrite=True, overwrite_meta=True, overwrite_imp=True
        )
        run.generate_input()
        with open(rpath / 'field_conversion.txt', 'w+') as g:
            for c in run_input['field_conversion']:
                g.write(f'{c:.8f}\n')
        logger.debug(f'Created {rpath.resolve()}...')

    logger.info(f'Created {len(qualikiz_plan)} QuaLiKiz run directories')


def parse_args():
    parser = argparse.ArgumentParser(
        prog='generate_qualikiz_runs',
        description='Generate standalone QuaLiKiz run directories from plasma simulation files.',
        epilog='Use --list to print the formats which can feed QuaLiKiz.',
    )
    parser = standalone.add_common_arguments(parser, 'qualikiz')
    parser.add_argument('-x', '--executable', type=str, default='./', help='Path to the QuaLiKiz executable, required to generate the input directories')
    return parser.parse_args()


def main():
    args = parse_args()
    standalone.configure_logging(quiet=args.quiet, debug=args.debug)

    cmap = convert.build_conversion_map()
    pmap = standalone.build_producer_map()

    if args.list:
        print(standalone.format_producer_map({'qualikiz': pmap['qualikiz']}, cmap))
        return 0

    if not args.ifiles:
        logger.error('No input files given! Pass at least one path, or use --list to see the supported sources.')
        return 2

    xpath = Path(f'{args.executable}')
    if not xpath.is_file():
        logger.error(f'Requested QuaLiKiz executable, {xpath.resolve()}, does not exist! Pass it with --executable.')
        return 2

    epath = None
    if args.eqdsk is not None:
        epath = Path(f'{args.eqdsk}').resolve()
        if not epath.is_file():
            logger.error(f'Requested EQDSK file, {epath}, does not exist!')
            return 2

    ipaths = standalone.resolve_input_paths(args.ifiles)
    n_generated = 0
    for i, ipath in enumerate(ipaths):
        try:
            qualikiz_plan = convert_to_qualikiz_input(
                ipath,
                source=args.format,
                via=args.via,
                eqdsk=epath,
                read_side=args.read_side,
                derive=(not args.no_derive),
                cmap=cmap,
                pmap=pmap,
                **standalone.parameter_kwargs(args),
            )
        except (ValueError, NotImplementedError, KeyError, OSError) as err:
            logger.error(f'QuaLiKiz generation from {ipath.name} failed: {err}')
            continue
        if len(qualikiz_plan) > 0:
            qualikiz_run_path = f'{args.output}_i{i:03d}' if len(ipaths) > 1 else f'{args.output}'
            generate_qualikiz_run_directories(qualikiz_plan, xpath, qualikiz_run_path)
            n_generated += 1

    logger.info(f'QuaLiKiz run generation completed for {n_generated} of {len(ipaths)} inputs')
    return 0 if n_generated == len(ipaths) and n_generated > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
