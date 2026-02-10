import copy
import argparse
import logging
from pathlib import Path
from ..classes.io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np

logger = logging.getLogger('fusio')


def convert_torax_to_qualikiz_input(
    torax_output_path: str | Path,
    time: float | Sequence[float] | NDArray | None = None,
    rho: float | Sequence[float] | NDArray | None = None,
    full_impurities: bool = False,
) -> Sequence[Any]:

    from ..classes.torax import torax_io
    opath = Path(torax_output_path)
    t = torax_io.from_file(output=opath)
    p = t.to_qualikiz_parameters(time=time, rho=rho, full_impurities=full_impurities)

    ion_template: Final[Mapping[str, Any]] = {
        'Ti': None,
        'ni': None,
        'Ati': None,
        'Ani': None,
        'Zi': None,
        'Ai': None,
    }
    qualikiz_input_template: Final[Mapping[str, Any]] = {
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

    qualikiz_plan = []
    for j, ptime in enumerate(p['time']):
        qualikiz_runpars: MutableMapping[str, Any] = {k: v for k, v in qualikiz_input_template.items()}
        for s in range(int(p['nion'].isel(time=j, drop=True).to_numpy())):
            qualikiz_runpars.update({k+f'{s:d}': v for k, v in ion_template.items()})
        for key in p.data_vars:
            if key in ['nion']:
                qualikiz_runpars[key] = int(p[key].isel(time=j, drop=True).to_numpy())
            elif key in qualikiz_runpars:
                qualikiz_runpars[key] = p[key].isel(time=j, drop=True).to_numpy().flatten()
        if 'rho' in p.coords:
            qualikiz_runpars['rho'] = p['rho'].to_numpy().flatten()
        for key in list(qualikiz_runpars.keys()):
            if qualikiz_runpars[key] is None:
                qualikiz_runpars.pop(key)
        num_ions = qualikiz_runpars.pop('nion', 1)
        conversion = qualikiz_runpars.pop('#Bunit_by_Bo', np.ones_like(qualikiz_runpars['rho']))
        qualikiz_plan.append({
            'inputs': copy.deepcopy(qualikiz_runpars),
            'location': f'qualikiz_run_t{j:05d}',
            'nion': int(num_ions),
            'field_conversion': copy.deepcopy(conversion),
        })

    logger.info(f'Generated {len(qualikiz_plan)} QuaLiKiz input sets from {opath.name}')

    return qualikiz_plan


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
    parser = argparse.ArgumentParser()
    parser.add_argument('ifiles', type=str, nargs='+', help='Path to input files for conversion to QuaLiKiz runs')
    parser.add_argument('-f', '--format', type=str, help='Format of input files')
    parser.add_argument('-x', '--executable', type=str, default='./', help='Path to QuaLiKiz executable, required to generate input directories')
    parser.add_argument('-o', '--output', type=str, default='./qualikiz_runs', help='Desired path to contain created QuaLiKiz run directories')
    parser.add_argument('-t', '--time', type=float, nargs='*', default=None, help='Time slices from which to generate QuaLiKiz runs')
    parser.add_argument('-r', '--rho', type=float, nargs='*', default=None, help='Radial slices from which to generate QuaLiKiz runs')
    parser.add_argument('-i', '--impurities', action='store_true', default=False, help='Toggle full impurity definition in generated QuaLiKiz runs, default is lumped')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ipaths = [Path(f'{ip}').resolve() for ip in args.ifiles if Path(f'{ip}').is_file()]
    xpath = Path(f'{args.executable}')
    for i, ipath in enumerate(ipaths):
        qualikiz_plan: Sequence[Any] = []
        match args.format:
            case 'torax':
                qualikiz_plan = convert_torax_to_qualikiz_input(
                    ipath,
                    args.time,
                    args.rho,
                    args.impurities,
                )
            case _:
                qualikiz_plan = []
        if xpath.is_file() and len(qualikiz_plan) > 0:
            qualikiz_run_path = f'{args.output}_i{i:03d}'
            generate_qualikiz_run_directories(
                qualikiz_plan,
                xpath,
                qualikiz_run_path,
            )
    logger.info('QuaLiKiz run generation script completed!')


if __name__ == '__main__':
    main()
