import copy
import logging
import argparse
from pathlib import Path
from ..classes.io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np

logger = logging.getLogger('fusio')


def convert_torax_to_tglf_input(
    torax_output_path: str | Path,
    time: float | Sequence[float] | NDArray | None = None,
    rho: float | Sequence[float] | NDArray | None = None,
    full_impurities: bool = False,
) -> Sequence[Any]:

    from ..classes.torax import torax_io
    t = torax_io.from_file(output=torax_output_path)
    p = t.to_tglf_parameters(time=time, rho=rho, full_impurities=full_impurities)

    species_template = {
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
    tglf_input_template = {
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

    tglf_plan = []
    for j, ptime in enumerate(p['time']):
        for i, prho in enumerate(p['rho']):
            tglf_runpars = copy.deepcopy(tglf_input_template)
            for s in range(int(p['NS'].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())):
                tglf_runpars.update({k+f'_{s+1:d}': v for k, v in species_template.items()})
            for key in p.data_vars:
                if key in ['NS']:
                    tglf_runpars[key] = int(p[key].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())
                elif key in tglf_runpars:
                    tglf_runpars[key] = float(p[key].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())
            for key in list(tglf_runpars.keys()):
                if tglf_runpars[key] is None:
                    tglf_runpars.pop(key)
            tglf_plan.append({
                'inputs': copy.deepcopy(tglf_runpars),
                'location': f'tglf_run_t{j:05d}_r{i:04d}',
            })

    logger.info(f'Generated {len(tglf_plan)} TGLF input sets from {torax_output_path.name}')

    return tglf_plan


def generate_tglf_run_directories(
    tglf_plan: Sequence[Any],
    tglf_run_path: str | Path,
) -> None:

    runpath = Path(f'{tglf_run_path}')
    runpath.mkdir(parents=True, exist_ok=True)

    for run in tglf_plan:
        rpath = runpath / run['location']
        rpath.mkdir(parents=True, exist_ok=True)
        fstr = "\n".join(["{}={}".format(k, v) for k, v in run['inputs'].items()])
        with open(rpath / 'input.tglf', 'w+') as f:
            f.write(fstr)
        logger.debug(f'Created {rpath.resolve()}...')

    logger.info(f'Created {len(tglf_plan)} TGLF run directories')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ifiles', type=str, nargs='+', help='Path to input files for conversion to TGLF runs')
    parser.add_argument('-f', '--format', type=str, help='Format of input files')
    parser.add_argument('-o', '--output', type=str, default='./tglf_runs', help='Desired path to contain created TGLF run directories')
    parser.add_argument('-t', '--time', type=float, nargs='*', default=None, help='Time slices from which to generate TGLF runs')
    parser.add_argument('-r', '--rho', type=float, nargs='*', default=None, help='Radial slices from which to generate TGLF runs')
    parser.add_argument('-i', '--impurities', action='store_true', default=False, help='Toggle full impurity definition in generated TGLF runs, default is lumped')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ipaths = [Path(f'{ip}').resolve() for ip in args.ifiles if Path(f'{ip}').is_file()]
    for i, ipath in enumerate(ipaths):
        tglf_plan = []
        match args.format:
            case 'torax':
                tglf_plan = convert_torax_to_tglf_input(
                    ipath,
                    args.time,
                    args.rho,
                    args.impurities,
                )
            case _:
                tglf_plan = []
        if len(tglf_plan) > 0:
            tglf_run_path = f'{args.output}_i{i:03d}'
            generate_tglf_run_directories(
                tglf_plan,
                tglf_run_path,
            )
    logger.info('TGLF run generation script completed!')


if __name__ == '__main__':
    main()
