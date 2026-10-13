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


TGLF_SPECIES_TEMPLATE: Final[Mapping[str, Any]] = {
    'ZS': None,
    'MASS': None,
    'RLNS': None,
    'RLTS': None,
    'TAUS': None,
    'AS': None,
    'VPAR': 0.0,
    'VPAR_SHEAR': 0.0,
    #'VNS_SHEAR': 0.0,
    #'VTS_SHEAR': 0.0,
}

TGLF_INPUT_TEMPLATE: Final[Mapping[str, Any]] = {
    # control
    'UNITS': 'CGYRO',
    'NS': 2,
    'USE_TRANSPORT_MODEL': '.true.',
    'GEOMETRY_FLAG': 1,
    'USE_BPER': '.true.',
    'USE_BPAR': '.false.',
    'USE_BISECTION': '.true.',
    'USE_MHD_RULE': '.false.',
    'USE_INBOARD_DETRAPPED': '.false.',
    'SAT_RULE': 3,
    'KYGRID_MODEL': 4,
    #'KYGRID_MODEL': 0,
    'XNU_MODEL': 3,
    'VPAR_MODEL': 0,
    #'VPAR_SHEAR_MODEL': 0,
    'SIGN_BT': 1.0,
    'SIGN_IT': 1.0,
    'KY': 0.3,
    #'KY': 1.5/1.8,
    'NEW_EIKONAL': '.true.',
    'VEXB': 0.0,
    'VEXB_SHEAR': 0.0,
    'BETAE': 0.0,
    'XNUE': 0.0,
    'ZEFF': 1.0,
    'DEBYE': 0.0,
    'IFLUX': 'T',
    'IBRANCH': -1,
    'NMODES': 5,
    'NBASIS_MAX': 6,  # Default is 4
    'NBASIS_MIN': 2,
    'NXGRID': 16,
    'NKY': 19,
    #'NKY': 75,
    'ADIABATIC_ELEC': '.false.',
    'ALPHA_P': 1.0,
    'ALPHA_MACH': 0.0,
    'ALPHA_E': 1.0,
    'ALPHA_QUENCH': 0.0,
    'ALPHA_ZF': 1.0,
    'XNU_FACTOR': 1.0,
    'DEBYE_FACTOR': 1.0,
    'ETG_FACTOR': 1.25,
    'B_MODEL_SA': 1,
    'FT_MODEL_SA': 1,
    # gaussian
    'WRITE_WAVEFUNCTION_FLAG': 0,
    'WIDTH': 1.65,
    'WIDTH_MIN': 0.3,
    'NWIDTH': 21,
    'FIND_WIDTH': '.true.',
    # miller
    'RMIN_LOC': 0.5,
    'RMAJ_LOC': 3.0,
    'ZMAJ_LOC': 0.0,
    'Q_LOC': 2.0,
    'Q_PRIME_LOC': 16.0,
    'P_PRIME_LOC': 0.0,
    'DRMINDX_LOC': 1.0,
    'DRMAJDX_LOC': 0.0,
    'DZMAJDX_LOC': 0.0,
    'KAPPA_LOC': 1.0,
    'S_KAPPA_LOC': 0.0,
    'DELTA_LOC': 0.0,
    'S_DELTA_LOC': 0.0,
    'ZETA_LOC': 0.0,
    'S_ZETA_LOC': 0.0,
    'KX0_LOC': 0.0,
    # expert
    'THETA_TRAPPED': 0.7,
    'PARK': 1.0,
    'GHAT': 1.0,
    'GCHAT': 1.0,
    'WD_ZERO': 0.1,
    'LINSKER_FACTOR': 0.0,
    'GRADB_FACTOR': 0.0,
    'FILTER': 2.0,
    'DAMP_PSI': 0.0,
    'DAMP_SIG': 0.0,
    #'NN_MAX_ERROR': -1.0,
    'WDIA_TRAPPED': 1.0,
    # extra
    '#RHO': None,
    'SHAT_SA': None,
    '#BUNIT_BY_BREF': None,
}


def convert_to_tglf_input(
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
    """Compute TGLF input parameters from any source format which can reach a producer."""
    ipath = Path(input_path)
    params = standalone.load_code_parameters(
        ipath,
        'tglf',
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
    tglf_plan = standalone.build_pointwise_plan(
        params,
        'tglf',
        template=TGLF_INPUT_TEMPLATE,
        species_template=TGLF_SPECIES_TEMPLATE,
        species_count='NS',
        species_offset=1,
    )
    logger.info(f'Generated {len(tglf_plan)} TGLF input sets from {ipath.name}')
    return tglf_plan


def convert_torax_to_tglf_input(
    torax_output_path: str | Path,
    time: float | Sequence[float] | NDArray | None = None,
    rho: float | Sequence[float] | NDArray | None = None,
    full_impurities: bool = False,
) -> Sequence[Any]:
    """Compute TGLF input parameters from a TORAX output file."""
    return convert_to_tglf_input(torax_output_path, time, rho, full_impurities, source='torax')


def generate_tglf_run_directories(
    tglf_plan: Sequence[Any],
    tglf_run_path: str | Path,
) -> None:
    standalone.write_input_files(tglf_plan, tglf_run_path, 'input.tglf')


def parse_args():
    parser = argparse.ArgumentParser(
        prog='generate_tglf_runs',
        description='Generate standalone TGLF run directories from plasma simulation files.',
        epilog='Use --list to print the formats which can feed TGLF.',
    )
    return standalone.add_common_arguments(parser, 'tglf').parse_args()


def main():
    args = parse_args()
    standalone.configure_logging(quiet=args.quiet, debug=args.debug)

    cmap = convert.build_conversion_map()
    pmap = standalone.build_producer_map()

    if args.list:
        print(standalone.format_producer_map({'tglf': pmap['tglf']}, cmap))
        return 0

    if not args.ifiles:
        logger.error('No input files given! Pass at least one path, or use --list to see the supported sources.')
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
            tglf_plan = convert_to_tglf_input(
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
            logger.error(f'TGLF generation from {ipath.name} failed: {err}')
            continue
        if len(tglf_plan) > 0:
            tglf_run_path = f'{args.output}_i{i:03d}' if len(ipaths) > 1 else f'{args.output}'
            generate_tglf_run_directories(tglf_plan, tglf_run_path)
            n_generated += 1

    logger.info(f'TGLF run generation completed for {n_generated} of {len(ipaths)} inputs')
    return 0 if n_generated == len(ipaths) and n_generated > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
