import logging
import argparse
from pathlib import Path
from ..classes.io import Any, Final
from collections.abc import Mapping, Sequence
import xarray as xr

from . import convert
from . import standalone

logger = logging.getLogger('fusio')


# Settings which CGYRO needs but which are not determined by the plasma state, the
# parameters computed from the source data are overlaid on top of these
CGYRO_INPUT_TEMPLATE: Final[Mapping[str, Any]] = {
    # control
    'PROFILE_MODEL': 1,
    'EQUILIBRIUM_MODEL': 2,  # Miller extended harmonic, covers both producers
    'NONLINEAR_FLAG': 0,
    'ZF_TEST_MODE': 0,
    'N_FIELD': 2,
    'BETA_STAR_SCALE': 1.0,
    'LAMBDA_STAR_SCALE': 1.0,
    'COLLISION_MODEL': 4,
    'COLLISION_FIELD_MODEL': 1,
    'ROTATION_MODEL': 2,
    # resolution
    'N_ENERGY': 8,
    'N_XI': 16,
    'N_THETA': 24,
    'N_RADIAL': 4,
    'N_TOROIDAL': 1,
    'BOX_SIZE': 1,
    'KY': 0.3,
    'E_MAX': 8.0,
    # time stepping
    'DELTA_T_METHOD': 0,
    'DELTA_T': 0.01,
    'MAX_TIME': 100.0,
    'PRINT_STEP': 100,
    'RESTART_STEP': 10,
    'FREQ_TOL': 0.001,
    # numerics
    'UP_RADIAL': 1.0,
    'UP_THETA': 1.0,
    'UP_ALPHA': 0.0,
    'NUP_RADIAL': 3,
    'NUP_THETA': 3,
    'NUP_ALPHA': 3,
    'VELOCITY_ORDER': 2,
    'AMP': 1.0e-6,
    # output
    'MOMENT_PRINT_FLAG': 1,
    'GFLUX_PRINT_FLAG': 1,
    'H_PRINT_FLAG': 0,
}


def convert_to_cgyro_input(
    input_path: str | Path,
    time: float | Sequence[float] | None = None,
    rho: float | Sequence[float] | None = None,
    full_impurities: bool = False,
    source: str | None = None,
    via: Sequence[str] | None = None,
    eqdsk: str | Path | None = None,
    read_side: str = 'output',
    derive: bool = True,
    **kwargs: Any,
) -> Sequence[Any]:
    """Compute CGYRO input parameters from any source format which can reach a producer."""
    ipath = Path(input_path)
    params = standalone.load_code_parameters(
        ipath,
        'cgyro',
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
    # The CGYRO producers emit purpose-built keys, including a variable length MXH shape
    # series, so everything they return is carried into the input file
    cgyro_plan = standalone.build_pointwise_plan(
        params,
        'cgyro',
        template=CGYRO_INPUT_TEMPLATE,
        overlay_all=True,
    )
    logger.info(f'Generated {len(cgyro_plan)} CGYRO input sets from {ipath.name}')
    return cgyro_plan


def convert_torax_to_cgyro_input(
    torax_output_path: str | Path,
    time: float | Sequence[float] | None = None,
    rho: float | Sequence[float] | None = None,
    full_impurities: bool = False,
) -> Sequence[Any]:
    """Compute CGYRO input parameters from a TORAX output file."""
    return convert_to_cgyro_input(torax_output_path, time, rho, full_impurities, source='torax')


def generate_cgyro_run_directories(
    cgyro_plan: Sequence[Any],
    cgyro_run_path: str | Path,
) -> None:
    standalone.write_input_files(cgyro_plan, cgyro_run_path, 'input.cgyro')


def parse_args():
    parser = argparse.ArgumentParser(
        prog='generate_cgyro_runs',
        description='Generate standalone CGYRO run directories from plasma simulation files.',
        epilog='Use --list to print the formats which can feed CGYRO.',
    )
    return standalone.add_common_arguments(parser, 'cgyro').parse_args()


def main():
    args = parse_args()
    standalone.configure_logging(quiet=args.quiet, debug=args.debug)

    cmap = convert.build_conversion_map()
    pmap = standalone.build_producer_map()

    if args.list:
        print(standalone.format_producer_map({'cgyro': pmap['cgyro']}, cmap))
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
            cgyro_plan = convert_to_cgyro_input(
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
            logger.error(f'CGYRO generation from {ipath.name} failed: {err}')
            continue
        if len(cgyro_plan) > 0:
            cgyro_run_path = f'{args.output}_i{i:03d}' if len(ipaths) > 1 else f'{args.output}'
            generate_cgyro_run_directories(cgyro_plan, cgyro_run_path)
            n_generated += 1

    logger.info(f'CGYRO run generation completed for {n_generated} of {len(ipaths)} inputs')
    return 0 if n_generated == len(ipaths) and n_generated > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
