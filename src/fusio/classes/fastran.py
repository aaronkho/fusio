import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, Sequence
from numpy.typing import NDArray
import numpy as np
import xarray as xr

from .io import io
from ..utils.plasma_tools import define_ion_species

logger = logging.getLogger('fusio')


class fastran_io(io):

    # The FASTRAN NetCDF output carries no variable attributes, so units and descriptions are
    # tabulated here, following the FASTRAN output variable documentation. Entries where that
    # documentation disagrees with the contents of the file itself are marked below, along with
    # the numerical check made against the file to settle the disagreement.
    dim_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'time': ('s', 'plasma time'),
        'rho': ('', 'normalized toroidal flux coordinate, sqrt(phi / phi_boundary)'),
        'imp': ('', 'impurity species index'),
    }
    scalar_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'r0': ('m', 'reference geometric major radius'),
        'b0': ('T', 'vacuum toroidal magnetic field at reference major radius'),
        'a0': ('m', 'minor radius of last closed flux surface'),
        'amain': ('', 'main ion mass number'),
        'zmain': ('', 'main ion charge number'),
        'we': ('MJ', 'electron thermal stored energy'),
        'wi': ('MJ', 'ion thermal stored energy'),
        'wb': ('MJ', 'fast ion stored energy'),
        'betan': ('%', 'normalized toroidal beta'),
        'pnbe': ('MW', 'total electron heating power from neutral beam'),
        'pnbi': ('MW', 'total ion heating power from neutral beam'),
        'prfe': ('MW', 'total electron heating power from radio frequency waves'),
        'prfi': ('MW', 'total ion heating power from radio frequency waves'),
        'prad': ('MW', 'total radiated power, negative as an electron energy sink'),
        'pei': ('MW', 'total collisional energy transfer power from ions to electrons'),
        'pfuse': ('MW', 'total electron heating power from fusion reactions'),
        'pfusi': ('MW', 'total ion heating power from fusion reactions'),
        'pfuse_equiv': ('MW', 'total equivalent electron heating power from fusion reactions'),
        'pfusi_equiv': ('MW', 'total equivalent ion heating power from fusion reactions'),
        'poh': ('MW', 'total ohmic heating power'),
        'taue': ('s', 'electron energy confinement time'),
        'taui': ('s', 'ion energy confinement time'),
        'tauth': ('s', 'thermal energy confinement time'),
        'tautot': ('s', 'total energy confinement time'),
        'tau98': ('s', 'energy confinement time from IPB98(y,2) scaling'),
        'tau89': ('s', 'energy confinement time from ITER89-P scaling'),
        'taunc': ('s', 'energy confinement time from neoclassical transport'),
        'tea': ('keV', 'volume-averaged electron temperature'),
        'tia': ('keV', 'volume-averaged ion temperature'),
        'nebar': ('10^19 m^-3', 'line-averaged electron density'),
        'ip': ('MA', 'total plasma current'),
        'ibs': ('MA', 'total bootstrap current'),
        'inb': ('MA', 'total neutral beam driven current'),
        'irf': ('MA', 'total radio frequency driven current'),
        'sn': ('10^19 s^-1', 'total particle source rate, the volume integral of sion'),
    }
    species_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'aimp': ('', 'impurity ion mass number'),
        'zimp': ('', 'impurity ion charge number'),
    }
    geom_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'rhob': ('m', 'toroidal flux coordinate, sqrt(phi / pi / b0)'),
        'volp': ('m^2', 'radial derivative of enclosed plasma volume, dV/drhob'),
        'area': ('m^2', 'flux surface area'),
        'ipol': ('', 'normalized poloidal current function, f / (r0 * b0)'),
        'g11': ('m^2', 'flux surface metric, dV/drhob * <|grad(rhob)|^2>'),
        'g22': ('m', 'flux surface metric, r0 * dV/drhob * <|grad(rhob)|^2 / r^2> / (4 * pi^2)'),
        'g33': ('', 'flux-surface-average r0^2 / r^2'),
        'gradrho': ('', 'flux-surface-average |grad(rhob)|'),
        'aminor': ('m', 'flux surface minor radius'),
        'rmajor': ('m', 'flux surface major radius'),
        'shift': ('m', 'flux surface Shafranov shift, rmajor - r0'),
        'delta': ('', 'flux surface triangularity'),
        'kappa': ('', 'flux surface elongation'),
        'fpol': ('Wb', 'poloidal magnetic flux'),
        'q': ('', 'safety factor'),
        'qmhd': ('', 'safety factor from MHD equilibrium'),
        'shat': ('', 'magnetic shear, (rho/q) * dq/drho'),
        'pmhd': ('Pa', 'total pressure from MHD equilibrium'),
        'ermhd': ('V m^-1', 'radial electric field from MHD equilibrium'),
        'betan_loc': ('%', 'local normalized toroidal beta'),
    }
    prof_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'te': ('keV', 'electron temperature'),
        'ti': ('keV', 'ion temperature'),
        'ne': ('10^19 m^-3', 'electron density'),
        'ni': ('10^19 m^-3', 'main ion density'),
        'nhe': ('10^19 m^-3', 'thermal helium ash density'),
        'nbeam': ('10^19 m^-3', 'fast ion density from neutral beam'),
        'nalpha': ('10^19 m^-3', 'fast alpha density from fusion reactions'),
        'wbeam': ('MJ m^-3', 'fast ion energy density from neutral beam'),
        'walp': ('MJ m^-3', 'fast alpha energy density from fusion reactions'),
        'omega': ('rad s^-1', 'toroidal angular velocity'),
        'zeff': ('', 'effective charge'),
        'sigma': ('MS m^-1', 'parallel electrical conductivity'),
        'vloop': ('V', 'loop voltage'),
        'nue_s': ('', 'normalized electron collisionality'),
        # Documented as derivatives with respect to rho, but the file holds derivatives with
        # respect to rhob, since tep / gradient(te, rho) is 1 / rhob[-1] at every interior point
        'tep': ('keV m^-1', 'electron temperature gradient, dte/drhob'),
        'tip': ('keV m^-1', 'ion temperature gradient, dti/drhob'),
        'nep': ('10^19 m^-4', 'electron density gradient, dne/drhob'),
        'nip': ('10^19 m^-4', 'main ion density gradient, dni/drhob'),
    }
    impurity_prof_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'nz': ('10^19 m^-3', 'impurity ion density'),
    }
    magn_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'j_tot': ('MA m^-2', 'total parallel current density'),
        'j_bs': ('MA m^-2', 'bootstrap current density'),
        'j_bs_0': ('MA m^-2', 'bootstrap current density before smoothing'),
        'j_nb': ('MA m^-2', 'neutral beam driven current density'),
        'j_rf': ('MA m^-2', 'radio frequency driven current density'),
        'j_oh': ('MA m^-2', 'ohmic current density'),
    }
    source_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'pe_nb': ('MW m^-3', 'electron energy source density from neutral beam'),
        'pi_nb': ('MW m^-3', 'ion energy source density from neutral beam'),
        'pe_rf': ('MW m^-3', 'electron energy source density from radio frequency waves'),
        'pi_rf': ('MW m^-3', 'ion energy source density from radio frequency waves'),
        'pe_fus': ('MW m^-3', 'electron energy source density from fusion reactions'),
        'pi_fus': ('MW m^-3', 'ion energy source density from fusion reactions'),
        'pe_ionization': ('MW m^-3', 'electron energy source density from ionization'),
        'pi_ionization': ('MW m^-3', 'ion energy source density from ionization'),
        'pi_cx': ('MW m^-3', 'ion energy source density from charge exchange'),
        'p_rad': ('MW m^-3', 'radiated energy source density, negative as an electron energy sink'),
        'p_ohm': ('MW m^-3', 'ohmic energy source density'),
        'p_ei': ('MW m^-3', 'collisional energy transfer density from ions to electrons'),
        'pe': ('MW m^-3', 'total electron energy source density'),
        'pi': ('MW m^-3', 'total ion energy source density'),
        'sion': ('10^19 m^-3 s^-1', 'electron particle source density from neutral beam, ionization and pellets'),
        'torque_nb': ('N m^-2', 'toroidal torque density from neutral beam'),
    }
    flux_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'chie': ('m^2 s^-1', 'electron heat diffusivity from transport model'),
        'chie_etg': ('m^2 s^-1', 'electron heat diffusivity from high-k electron temperature gradient modes'),
        'chie_paleo': ('m^2 s^-1', 'electron heat diffusivity from paleoclassical transport'),
        'chii': ('m^2 s^-1', 'ion heat diffusivity from transport model'),
        'chiv': ('m^2 s^-1', 'momentum diffusivity from transport model'),
        'chin': ('m^2 s^-1', 'particle diffusivity from transport model'),
        'chieneo': ('m^2 s^-1', 'electron heat diffusivity from neoclassical transport'),
        'chiineo': ('m^2 s^-1', 'ion heat diffusivity from neoclassical transport'),
        'chinneo': ('m^2 s^-1', 'particle diffusivity from neoclassical transport'),
        'chie_exp': ('m^2 s^-1', 'electron heat diffusivity from power balance'),
        'chii_exp': ('m^2 s^-1', 'ion heat diffusivity from power balance'),
        'chiv_exp': ('m^2 s^-1', 'momentum diffusivity from power balance'),
        'chin_exp': ('m^2 s^-1', 'particle diffusivity from power balance'),
        # Documented in MW/m^2, but the file omits the 1.602e-3 conversion FASTRAN applies
        # elsewhere, since fluxe_exp equals -ne * chie_exp * tep and reproduces the enclosed
        # electron power divided by the flux surface area only after scaling by that factor
        'fluxe': ('keV 10^19 m^-2 s^-1', 'electron heat flux from transport model'),
        'fluxi': ('keV 10^19 m^-2 s^-1', 'ion heat flux from transport model'),
        'fluxe_exp': ('keV 10^19 m^-2 s^-1', 'electron heat flux from power balance'),
        'fluxi_exp': ('keV 10^19 m^-2 s^-1', 'ion heat flux from power balance'),
        'fluxn_exp': ('10^19 m^-2 s^-1', 'electron particle flux from power balance'),
    }
    # The FASTRAN solver writes the elongation into the variable named delta and the triangularity
    # into the variable named kappa, a known defect in its NetCDF writer. The documented meaning of
    # each name is restored on read and undone again on write, mapping class variable to file
    # variable below, so that delta always holds the triangularity and kappa always the elongation.
    swapped_vars: Final[Mapping[str, str]] = {
        'delta': 'kappa',
        'kappa': 'delta',
    }
    # FASTRAN leaves the unlimited time dimension unwritten, so it comes back as the NetCDF
    # default double fill value instead of a plasma time
    netcdf_fill_tolerance: Final[float] = 1.0e30


    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        ipath = None
        opath = None
        for arg in args:
            if ipath is None and isinstance(arg, (str, Path)):
                ipath = Path(arg)
            elif opath is None and isinstance(arg, (str, Path)):
                opath = Path(arg)
        for key, kwarg in kwargs.items():
            if ipath is None and key in ['input'] and isinstance(kwarg, (str, Path)):
                ipath = Path(kwarg)
            if opath is None and key in ['path', 'file', 'output'] and isinstance(kwarg, (str, Path)):
                opath = Path(kwarg)
        if ipath is not None:
            self.read(ipath, side='input')
        if opath is not None:
            self.read(opath, side='output')
        self.autoformat()


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            logger.warning(f'{self.format} input reading function not defined yet...')
        else:
            self.output = self._read_fastran_netcdf_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'output',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            logger.warning(f'{self.format} input writing function not defined yet...')
        else:
            self._write_fastran_netcdf_file(path, self.output, overwrite=overwrite)


    def _read_fastran_netcdf_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        data = xr.Dataset()
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                temp_data = xr.open_dataset(load_path, engine='netcdf4', decode_times=False)
                ntime = int(temp_data.sizes.get('time', 0))
                coords: MutableMapping[str, Any] = {}
                time: NDArray = np.arange(ntime).astype(float)
                if 'time' in temp_data:
                    time = temp_data['time'].to_numpy().astype(float)
                    unset = ~np.isfinite(time) | (np.abs(time) >= self.netcdf_fill_tolerance)
                    if np.any(unset):
                        logger.warning(f'{self.format} output file does not define a plasma time, using slice index instead...')
                        time = np.where(unset, np.arange(len(time)).astype(float), time)
                coords['time'] = (['time'], time, {'units': self.dim_vars['time'][0], 'description': self.dim_vars['time'][1]})
                if 'rho' in temp_data:
                    coords['rho'] = (['rho'], temp_data['rho'].to_numpy(), {'units': self.dim_vars['rho'][0], 'description': self.dim_vars['rho'][1]})
                nimp = int(temp_data.sizes.get('imp', 0))
                if nimp > 0:
                    coords['imp'] = (['imp'], np.arange(nimp).astype(int), {'units': self.dim_vars['imp'][0], 'description': self.dim_vars['imp'][1]})
                subsets = [
                    self.scalar_vars,
                    self.species_vars,
                    self.geom_vars,
                    self.prof_vars,
                    self.magn_vars,
                    self.source_vars,
                    self.flux_vars,
                ]
                data_vars: MutableMapping[str, Any] = {}
                if np.any([tag in temp_data for tag in self.swapped_vars.values()]):
                    logger.warning(f'{self.format} output file exchanges the elongation and triangularity variable names, correcting on read...')
                for subset in subsets:
                    for var, meta in subset.items():
                        tag = self.swapped_vars.get(var, var)
                        if tag in temp_data:
                            new_dims = [str(dim) for dim in temp_data[tag].dims]
                            if np.all([dim in coords for dim in new_dims]):
                                data_vars[var] = (new_dims, temp_data[tag].to_numpy(), {'units': meta[0], 'description': meta[1]})
                # FASTRAN writes one profile per impurity species, these are stacked onto the imp dimension
                for var, meta in self.impurity_prof_vars.items():
                    stack = [temp_data[f'{var}{k:d}'] for k in range(nimp) if f'{var}{k:d}' in temp_data]
                    if len(stack) == nimp and nimp > 0:
                        new_dims = [str(dim) for dim in stack[0].dims] + ['imp']
                        if np.all([dim in coords for dim in new_dims]):
                            new_data = np.stack([val.to_numpy() for val in stack], axis=-1)
                            data_vars[var] = (new_dims, new_data, {'units': meta[0], 'description': meta[1]})
                    elif len(stack) > 0:
                        logger.warning(f'Found {len(stack)} {var} profiles but {nimp} impurity species in {self.format} output file! Dropping {var}...')
                data = xr.Dataset(data_vars=data_vars, coords=coords)
        return data


    def _write_fastran_netcdf_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not opath.exists() or overwrite:
                # Restore the variable names which the FASTRAN solver exchanges on output, taken
                # from the original data since the targets overwrite one another
                wdata = data.copy(deep=True)
                for var, tag in self.swapped_vars.items():
                    if var in data and tag in wdata:
                        wdata[tag] = data[var].copy(deep=True)
                # Unstack the impurity profiles to recover the one-variable-per-species layout
                for var in self.impurity_prof_vars:
                    if var in wdata and 'imp' in wdata[var].dims:
                        for k in range(wdata.sizes['imp']):
                            wdata[f'{var}{k:d}'] = wdata[var].isel(imp=k, drop=True)
                        wdata = wdata.drop_vars([var])
                wdata.to_netcdf(opath, mode='w', format='NETCDF4')
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def resolve_ion_species(
        self,
        data: xr.Dataset,
    ) -> tuple[list[str], list[str], list[NDArray], list[NDArray], list[NDArray], list[NDArray]]:
        """Resolve the ion species carried by a FASTRAN output dataset.

        FASTRAN carries a lumped main ion, a thermal helium ash, an arbitrary number of
        impurities indexed on the imp dimension, and fast populations from the neutral beam
        and from fusion reactions. Species resolving to the same element symbol are merged,
        which is exactly charge-conserving since a shared symbol implies a shared charge, so
        quasi-neutrality and the effective charge are both preserved while only the mass of
        the merged species becomes approximate. Fast species take a _fast suffix to keep them
        distinct from the thermal species of the same element, applied through
        define_ion_species' fast keyword and only where a name would otherwise be duplicated.

        Expects a dataset already reduced to the desired time window. Returns the species
        names, types, masses, charges, densities and temperatures as lists, with masses and
        charges shaped per time and densities and temperatures shaped per time and radius.
        """

        ntime = len(data['time']) if 'time' in data.coords else 1
        nrho = len(data['rho']) if 'rho' in data.coords else 0
        ones = np.ones(ntime)
        zeros = np.zeros((ntime, nrho))
        ti = data['ti'].to_numpy() if 'ti' in data else copy.deepcopy(zeros)
        amain = data['amain'].to_numpy().flatten() if 'amain' in data else 2.0 * ones
        zmain = data['zmain'].to_numpy().flatten() if 'zmain' in data else 1.0 * ones
        mname, _, _ = define_ion_species(z=float(zmain[-1]), a=float(amain[-1]))
        species = []
        if 'ni' in data:
            species.append((f'{mname}', '[therm]', amain, zmain, data['ni'].to_numpy(), ti))
        if 'nhe' in data:
            species.append(('He', '[therm]', 4.0 * ones, 2.0 * ones, data['nhe'].to_numpy(), ti))
        if 'nz' in data and 'aimp' in data and 'zimp' in data and 'imp' in data.coords:
            aimp = data['aimp'].to_numpy().flatten()
            zimp = data['zimp'].to_numpy().flatten()
            for k in range(len(data.coords['imp'])):
                zname, _, _ = define_ion_species(z=float(zimp[k]), a=float(aimp[k]))
                species.append((f'{zname}', '[therm]', aimp[k] * ones, zimp[k] * ones, data['nz'].isel(imp=k).to_numpy(), ti))
        # Fast species are only taken when populated, since FASTRAN writes their profiles even
        # when the corresponding heating or fusion source is switched off, and a zero-density
        # species breaks downstream normalizations
        for dkey, wkey, fmass, fcharge in [
            ('nbeam', 'wbeam', amain, zmain),
            ('nalpha', 'walp', 4.0 * ones, 2.0 * ones),
        ]:
            if dkey in data and np.any(data[dkey].to_numpy() > 0.0):
                fname, _, _ = define_ion_species(z=float(fcharge[-1]), a=float(fmass[-1]))
                nfast = data[dkey].to_numpy()
                tfast = copy.deepcopy(ti)
                if wkey in data:
                    # FASTRAN stores the fast ion energy density as n * (eperp + epll) * 1.602e-3,
                    # inverted here into the equivalent temperature (2 / 3) * (eperp + epll)
                    tfast = np.where(
                        nfast > 0.0,
                        (2.0 / 3.0) * data[wkey].to_numpy() / np.where(nfast > 0.0, nfast, 1.0) / 1.602e-3,
                        tfast,
                    )
                species.append((fname, '[fast]', fmass, fcharge, nfast, tfast))
        names: list[str] = []
        types: list[str] = []
        masses: list[np.ndarray] = []
        charges: list[np.ndarray] = []
        densities: list[np.ndarray] = []
        temperatures: list[np.ndarray] = []
        for sname, stype, smass, scharge, sdensity, stemperature in species:
            newname = f'{sname}'
            if newname in names and stype == '[fast]':
                # The type field already distinguishes the fast population from the thermal one, so
                # the tag is only applied where it is needed to keep the species names distinct
                newname, _, _ = define_ion_species(short_name=sname, fast=True)
            if newname in names:
                logger.warning(f'Multiple {newname} species found in {self.format} data, merging their densities...')
                densities[names.index(newname)] = densities[names.index(newname)] + sdensity
                continue
            names.append(newname)
            types.append(stype)
            masses.append(smass)
            charges.append(scharge)
            densities.append(sdensity)
            temperatures.append(stemperature)
        return names, types, masses, charges, densities, temperatures


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    @classmethod
    def from_plasma(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError('Conversion from plasma class not implemented yet!')
