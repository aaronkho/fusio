import re
import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from .io import io

logger = logging.getLogger('fusio')


class transp_io(io):


    # TODO: Variable lists are currently non-exhaustive
    dim_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'time': ('', 'plasma time'),
        'x': ('', 'r/a cell centres'),
        'xb': ('', 'r/a cell boundaries'),
        'iasym': ('', 'shaping coefficient index'),
    }
    coord_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'rzon': ('cm', 'zone radius'),
        'rboun': ('cm', 'boundary radius'),
        'rmajb': ('cm', 'major radius boundary'),
        'rmnmp': ('cm', 'minor radius midplane'),
        'rmjmp': ('cm', 'major radius centroid midplane'),
        'ympa': ('cm', 'vertical position'),
        #('ympbdy', '', 'vertical position boundary'),
        'raxis': ('cm', 'major radius of magnetic axis'),
        'yaxis': ('cm', 'vertical position of magnetic axis'),
        'darea': ('cm**2', 'zone cross-sectional area'),
        'dvol': ('cm**3', 'zone volume'),
        'surf': ('cm**2', 'flux surface area'),
        'lpol': ('cm', 'poloidal path length'),
        #'pvol': ('cm**3', 'integrated volume'),
        #'parea': ('cm**2', 'integrated cross-sectional area'),
    }
    geom_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'gri': ('cm**-1', 'flux-surface-average 1/R'),
        'gr2': ('cm**2', 'flux-surface-average R**2'),
        'gr2i': ('cm**-2', 'flux-surface-average 1/R**2'),
        'gr2x2': ('', 'flux-surface-average R**2*grad(x)**2'),
        'gx2r2i': ('cm**-4', 'flux-surface-average grad(x)**2/R**2'),
        'grixi': ('', 'flux-surface-average 1/(R*grad(x))'),
        'gxi': ('cm**-1', 'flux-surface-average grad(x)'),
        'gxi2': ('cm**-2', 'flux-surface-average grad(x)**2'),
        'drav': ('cm', 'flux-surface-average dR'),
        'dravfac': ('', '<dR> * <1/dR>'),
        'gbr2': ('T cm**2', 'flux-surface-average B*R**2'),
        'gb1': ('T', 'flux-surface-average B'),
        'gb2': ('T**2', 'flux-surface-average B**2'),
        'gb2i': ('T**-2', 'flux-surface-average 1/B**2'),
        'gx2b2i': ('T**-2 cm**-2', 'flux-surface-average grad(x)**2/B**2'),
        'jgphr2i': ('A cm**-1', '<J.grad(phi)> / <1/R**2>'),
        'arat': ('', 'aspect ratio'),
        'elong': ('', 'flux surface elongation'),
        'triang': ('', 'flux surface triangularity'),
        'triangu': ('', 'flux surface upper triangularity'),
        'triangl': ('', 'flux surface lower triangularity'),
        'square_uo': ('', 'flux surface upper outer squareness'),
        'square_lo': ('', 'flux surface lower outer squareness'),
        'sshaf': ('', 'Shafranov shift'),
        #'ashaf': ('', 'Shafranov shift of magnetic axis'),
    }
    geom_moment_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'rmc': ('', 'R cosine moment'),
        'ymc': ('', 'Z cosine moment'),
        'rms': ('', 'R sine moment'),
        'yms': ('', 'Z sine moment'),
        'rmcb': ('', 'R cosine moment boundary'),
        'ymcb': ('', 'Z cosine moment boundary'),
        'rmsb': ('', 'R sine moment boundary'),
        'ymsb': ('', 'Z sine moment boundary'),
    }
    magn_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'bpol': ('T', 'poloidal magnetic field'),
        'bz': ('T', 'vertical magnetic field at outer midplane'),
        'cur': ('A cm**-2', 'total plasma current density'),
        'curoh': ('A cm**-2', 'ohmic plasma current density'),
        'curxt': ('A cm**-2', 'externally-driven plasma current density'),
        'pljb': ('A T cm**-2', 'flux-surface-average j.B'),
        'pljbxoh': ('A T cm**-2', 'flux-surface-average ohmic j.B'),
        'pljbxt': ('A T cm**-2', 'flux-surface-average externally-driven j.B'),
        #'pljbxtr': ('A T cm**-2', 'flux-surface-average resistive j.B'),
        #'pljbsnc': ('A T cm**-2', 'flux-surface-average neoclassical bootstrap j.B'),
        #'pljbsneo': ('A T cm**-2', 'flux-surface-average NEO-GK bootstrap j.B'),
        #'pljbbgpi': ('', ''),
        'v': ('V', 'voltage'),
        'vpoh': ('V', 'voltage for ohmic calculations'),
        'poh': ('W cm**-3', 'ohmic heating power'),
        #'poht': ('W', 'total ohmic input power'),
        #'ipxvs': ('W', 'flux surface power'),
        'plflx': ('Wb rad**-1', 'poloidal flux'),
        'trflx': ('Wb', 'toroidal flux'),
        #'plflxa': ('Wb rad**-1', 'enclosed poloidal flux'),
        #'tflux': ('Wb', 'enclosed toroidal flux'),
        'curbs': ('A cm**-2', 'bootstrap current density'),
        #'curbsneo': ('A cm**-2', 'NEO-GK bootstrap current density'),
        #'curbshag': ('A cm**-2', 'Hager bootstrap current density'),
        #'curbswnc': ('A cm**-2', 'NCLASS bootstrap current density'),
        #'curbseps': ('A cm**-2', 'aspect ratio bootstrap current density'),
        'eta_use': ('Ohm cm', 'resistivity used in TRANSP calculations'),
        #'eta_nc': ('Ohm cm', 'neoclassical resistivity'),
        #'eta_sp': ('Ohm cm', 'Spitzer resistivity'),
        #'eta_sps': ('Ohm cm', 'Sauter Spitzer resistivity'),
        #'eta_wnc': ('Ohm cm', 'NCLASS resistivity'),
        #'eta_tsc': ('Ohm cm', 'TSC neoclassical resistivity'),
        #'eta_snc': ('Ohm cm', 'Sauter neoclassical resistivity'),
        #'dflux': ('Wb', 'diamagnetic flux'),
        #'dflxm': ('Wb', 'measured diamagnetic flux'),
        'lio2c': ('', 'internal inductance li/2 from current profile'),
        #'lio2': ('', 'internal inductance li/2'),
        #'lio2m': ('', 'internal inductance li/2 from magnetics measurements'),
        'li_3': ('', 'internal inductance, ITER definition'),
        #'li_1': ('', 'internal inductance'),
        #'li_vdiff': ('', 'internal inductance from voltage difference'),
        'li2pb': ('', 'internal inductance li/2 + beta poloidal'),
        'betae': ('', 'electron beta poloidal'),
        'betai': ('', 'thermal ion beta poloidal'),
        'betat': ('', 'total beta poloidal'),
        'betar': ('', 'rotation beta poloidal'),
        'bbeta': ('', 'beam beta poloidal'),
        'bte': ('', 'electron beta toroidal'),
        'bti': ('', 'thermal ion beta toroidal'),
        'bttot': ('', 'total beta toroidal'),
        'btrot': ('', 'rotation beta toroidal'),
        'btbe': ('', 'beam beta toroidal'),
        #'btpl': ('', 'plasma beta toroidal'),
        'btal': ('', 'alpha beta toroidal'),
    }
    species_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'xzimpj': ('', 'zone average impurity charge'),
        'aimpj': ('', 'zone average impurity mass number'),
        'zimps_tok': ('', 'average impurity charge from TOK'),
    }
    prof_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'q': ('', 'safety factor'),
        'shat': ('', 'magnetic shear (r/q)*(dq/dr)'),
        'qp': ('', 'safety factor profile'), # What?
        'ne': ('cm**-3', 'electron density'),
        'te': ('eV', 'electron temperature'),
        'ni': ('cm**-3', 'total ion density'),
        'ti': ('eV', 'ion temperature'),
        'tipro': ('eV', 'measured ion temperature'),
        'pplas': ('Pa', 'plasma pressure'),
        'nimp': ('cm**-3', 'impurity ion density'),
        'nimps_tok': ('cm**-3', 'impurity ion density from TOK'),
        'tx': ('eV', 'impurity ion temperature'),
        'zeffp': ('', 'effective charge'),
        #'tmj': ('eV', 'minority ion temperature'),
        #'tmjsm': ('eV', 'minority ion temperature smoothed'),
        #'tiav': ('eV', 'averaged ion temperature including minority'),
        #'nd': ('', ''),
        'nalpha': ('cm**-3', 'alpha ion density'),
        'vtormp': ('cm s**-1', 'toroidal velocity on outer midplane'),
        'vpolmp': ('cm s**-1', 'poloidal velocity on outer midplane'),
        'omgnc': ('rad s**-1', 'neoclassical toroidal angular velocity'),
        'omega': ('rad s**-1', 'toroidal angular velocity'),
        'omega_nc': ('rad s**-1', 'neoclassical toroidal angular velocity'), # Duplicated?
        'nusti': ('', 'ion normalized collisionality'),
        'nuste': ('', 'electron normalized collisionality'),
        'cloge': ('', 'electron coulomb logarithm'),
        'clogi': ('', 'ion coulomb logarithm'),
        'etae': ('', 'logarithmic electron temperature gradient by logarithmic electron density gradient'),
        'etai': ('', 'logarithmic ion temperature gradient by logarithmic ion density gradient'),
        'etaie': ('', 'logarithmic ion temperature gradient by logarithmic electron density gradient'),
        'srexba': ('rad s**-1', 'ExB shearing rate'),
        'srexbphi': ('rad s**-1', 'ExB shearing rate from toroidal velocity'),
        'srexbtht': ('rad s**-1', 'ExB shearing rate from poloidal velocity'),
        'srexbgrp': ('rad s**-1', 'ExB shearing rate from pressure gradient'),
        #'srexbmod': ('rad s**-1', 'ExB shearing rate in transport model'),
        'vrpot': ('V', 'radial electric potential'),
        'epotro': ('V', 'radial electric potential from rotation velocity'),
    }
    global_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'voltsec0': ('V s', 'axial flux consumption'),
        #'voltsec': ('V s', 'volt-second flux consumption'),
        'tauea': ('s', 'thermal energy confinement time'),
        'taua1': ('s', 'total energy confinement time'),
        #'tauee': ('s', 'electron energy confinement time'),   # old?
        'tee': ('s', 'electron energy confinement time'),
        'teest': ('', 'normalized electron energy confinement time'),
        'tei': ('s', 'ion energy confinement time'),
        'teist': ('', 'normalized ion energy confinement time'),
        'taupe': ('s', 'electron particle confinement time'),
        'tapwe': ('s', 'electron particle confinement time Ware correction'),
        'taupi': ('s', 'ion energy confinement time'),
        #'taupd': ('s', 'deuterium particle confinement time'),
        'tauphi': ('s', 'momentum confinement time'),
        'taue': ('s', 'plasma energy confinement time'),
        'taues': ('s', 'normalized plasma energy confinement time'),
        'taue98y1': ('s', 'energy confinement time from 98y,1 scaling'),
        'taue98y2': ('s', 'energy confinement time from 98y,2 scaling'),
        #'taue89p': ('s', 'energy confinement time from 89P scaling'),
        #'taue97lg': ('s', 'energy confinement time from 97L,g scaling'),
        #'taue97lth': ('s', 'energy confinement time from 97L,th scaling'),
        #'tauest06': ('s', 'energy confinement time from ST06 scaling'),
        'h98y1': ('', 'confinement factor against 98y,1 scaling'),
        'h98y2': ('', 'confinement factor against 98y,2 scaling'),
        #'h89p': ('', 'confinement factor against 89P scaling'),
        #'h97lg': ('', 'confinement factor against 97L,g scaling'),
        #'h97lth': ('', 'confinement factor against 97L,th scaling'),
        #'hst06': ('', 'confinement factor against ST06 scaling'),
        'pl2hreq': ('W', 'L-H transition power'),
        'pl2htot': ('W', 'total heating power'),
        #'lhmode': ('', 'H-mode indicator'),
        'bplim': ('W', 'fast ion orbit energy loss rate'),
        'bpth': ('W', 'thermalized fast ion power'),
        'bpst': ('W', 'stored fast ion power'),
        'bsorb': ('s**-1', 'fast ion orbit particle loss rate'),
        'bsth': ('s**-1', 'fast ion thermalization rate'),
        'bdndt': ('s**-1', 'fast ion population change rate'),
        'neutt': ('s**-1', 'total neutron generation rate'),
        'neutx': ('s**-1', 'neutron generation rate from thermal reactions'),
    }
    transp_vars: Final[Mapping[str, tuple[str, ...]]] = {
        #'gaine': ('', ''),
        #'eheat': ('', ''),
        #'pcnve': ('', ''),
        #'pcnde': ('', ''),
        #'dnedt': ('', ''),
        #'divfe': ('', ''),
        #'scew': ('', ''),
        #'scev': ('', ''),
        #'scez': ('', ''),
        #'dnidt': ('', ''),
        #'divfi': ('', ''),
        #'sbtot': ('', ''),
        #'swtot': ('', ''),
        #'sbal_ion': ('', ''),
        #'dnimp': ('', ''),
        #'dfimp': ('', ''),
        #'scimp': ('', ''),
        #'dzimp': ('', ''),
        #'dnddt': ('', ''),
        #'divfd': ('', ''),
        #'svd': ('', ''),
        #'swd': ('', ''),
        #'sbal_d': ('', ''),
        #'mdot': ('', ''),
        #'tqin': ('', ''),
        #'mconv': ('', ''),
        #'mvisc': ('', ''),
        #'m0net': ('', ''),
        #'tibal': ('', ''),
        #'tebal': ('', ''),
        #'phbal': ('', ''),
        #'bpbal': ('', ''),
        #'bsbal': ('', ''),
        #'bphck': ('', ''),
        #'bpbal_d': ('', ''),
        #'bsbal_d': ('', ''),
        #'bphck_d': ('', ''),
        #'p0bal': ('', ''),
    }
    source_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'qie': ('W cm**-3', 'ion-electron equipartition power density'),
        'qrot': ('W cm**-3', 'ion heat density from rotational stress'),
        'gasd': ('s**-1', 'deuterium neutral gas particle flow rate'),
        'rcyd': ('s**-1', 'deuterium recycling particle flow rate'),
        'tqin': ('N m cm**-3', 'total injection torque'),
        'tqntv': ('N m cm**-3', 'torque from neoclassical toroidal viscosity'),
        'prad': ('W cm**-3', 'energy source density from radiation'),
        'prad_br': ('W cm**-3', 'energy source density from Bremsstrahlung radiation'),
        'prad_li': ('W cm**-3', 'energy source density from line radiation'),
        'prad_cy': ('W cm**-3', 'energy source density from cyclotron radiation'),
        'pion': ('W cm**-3', 'energy source density from ionization'),
        'prads_tok': ('W cm**-3', 'energy source density from TOK radiation'),
        'prls_tok': ('W cm**-3', 'energy source density from TOK line radiation'),
        'prbs_tok': ('W cm**-3', 'energy source density from TOK Bremsstrahlung radiation'),
        'pni': ('W cm**-3', 'neutral ionization energy source density'),
        'sbtot': ('cm**-3 s**-1', 'total ion particle source density (beam + halo)'),
        'sbe': ('cm**-3 s**-1', 'electron particle source density from neutral beam'),
        'sbth': ('cm**-3 s**-1', 'ion particle source density from neutral beam fast ion thermalization'),
        'pbe': ('W cm**-3', 'electron energy source density from neutral beam'),
        'pbi': ('W cm**-3', 'ion energy source density from neutral beam'),
        'pbth': ('W cm**-3', 'ion energy source density from neutral beam fast ion thermalization'),
        #'sbe_d': ('cm**-3 s**-1', 'electron particle source density from deuterium neutral beam'),
        #'sbth_d': ('cm**-3 s**-1', 'ion particle source density from deuterium neutral beam fast ion thermalization'),
        #'pbe_d': ('W cm**-3', 'electron energy source density from deuterium neutral beam'),
        #'pbi_d': ('W cm**-3', 'ion energy source density from deuterium neutral beam'),
        #'pbth_d': ('W cm**-3', 'ion energy source density from deuterium neutral beam fast ion thermalization'),
        #'sdbbi': ('cm**-3 s**-1', ''),
        #'sdbbx': ('cm**-3 s**-1', ''),
        #'sdb_ii': ('cm**-3 s**-1', ''),
        #'sdb_ie': ('cm**-3 s**-1', ''),
        #'sdb_iz': ('cm**-3 s**-1', ''),
        'omegb': ('rad s**-1', 'average ion angular velocity of neutral beam'),
        'tqbe': ('N m cm**-3', 'electron torque from neutral beam'),
        'tqbi': ('N m cm**-3', 'ion torque from neutral beam'),
        'tqbco': ('N m cm**-3', 'collisional torque from neutral beam'),
        'tqbth': ('N m cm**-3', 'thermalization torque from neutral beam'),
        'tqjxb': ('N m cm**-3', 'JxB torque from neutral beam'),
        'tqbcx': ('N m cm**-3', 'charge exchange anti-torque from neutral beam'),
        #'tqohb': ('', ''),
        #'tqxfr': ('', ''),
        'pbtotnb': ('W cm**-3', 'total energy source density from neutral beam'),
        'pbenb': ('W cm**-3', 'total electron energy source density from neutral beam'),
        'pbinb': ('W cm**-3', 'total ion energy source density from neutral beam'),
        'pbthnb': ('W cm**-3', 'total thermalization energy source density from neutral beam'),
        'bdensnb': ('cm**-3', 'total density from neutral beam'),
        'bdepnb': ('cm**-3 s**-1', 'total particle source density from neutral beam'),
        'tqtotnb': ('N m cm**-3', 'total torque from neutral beam'),
        'tqcolnb': ('N m cm**-3', 'total collisional torque from neutral beam'),
        'tqthnb': ('N m cm**-3', 'total thermalization torque from neutral beam'),
        'tqjbnb': ('N m cm**-3', 'total JxB torque from neutral beam'),
        'tqjbnbd': ('N m cm**-3', 'total JxB torque deposited from neutral beam'),
        #'peech': ('', ''),
        #'eccur': ('', ''),
        #'piich': ('', ''),
        #'peich': ('', ''),
        #'iccur': ('', ''),
        #'nmini': ('', ''),
        'uthrm': ('J cm**-3', 'thermal energy density'),
        'ufastpp': ('J cm**-3', 'perpendicular fast ion energy density'),
        'ufastpa': ('J cm**-3', 'parallel fast ion energy density'),
        'ualphpp': ('J cm**-3', 'perpendicular fast alpha energy density'),
        'ualphpa': ('J cm**-3', 'parallel fast alpha energy density'),
        'tausal': ('s', 'fast alpha slowing down time'),
        'sceal': ('cm**-3 s**-1', 'fast alpha source density from fusion reactions'),
        'pale': ('W cm**-3', 'electron energy source density from fast alpha thermalization'),
        'pali': ('W cm**-3', 'ion energy source density from fast alpha thermalization'),
        'thntx': ('cm**-3 s**-1', 'neutron source density from thermal reactions'),
        'ttntx': ('cm**-3 s**-1', 'total neutron source density'),
    }
    flux_vars: Final[Mapping[str, tuple[str, ...]]] = {
        #'iptr': ('cm**-3 s**-1', 'ion particle transport'),
        #'eptr': ('cm**-3 s**-1', 'electron particle transport'),
        #'xptr': ('cm**-3 s**-1', 'impurity particle transport'),
        #'ietr': ('W cm**-3', 'ion energy transport'),
        #'eetr': ('W cm**-3', 'electron energy transport'),
        #'amtr': ('N m cm**-3', 'angular momentum transport'),
        #'iptr_mod': ('cm**-3 s**-1', 'ion particle transport from model'),
        #'eptr_mod': ('cm**-3 s**-1', 'electron particle transport from model'),
        #'xptr_mod': ('cm**-3 s**-1', 'impurity particle transport from model'),
        #'ptrd_mod': ('cm**-3 s**-1', 'deuterium particle transport from model'),
        #'ietr_mod': ('W cm**-3', 'ion energy transport from model'),
        #'eetr_mod': ('W cm**-3', 'electron energy transport from model'),
        #'amtr_mod': ('N m cm**-3', 'angular momentum transport from model'),
        'iptr_obs': ('cm**-3 s**-1', 'ion particle transport from observations'),
        'eptr_obs': ('cm**-3 s**-1', 'electron particle transport from observations'),
        'xptr_obs': ('cm**-3 s**-1', 'impurity particle transport from observations'),
        #'ptrd_obs': ('cm**-3 s**-1', 'deuterium particle transport from observations'),
        'ietr_obs': ('W cm**-3', 'ion energy transport from observations'),
        'eetr_obs': ('W cm**-3', 'electron energy transport from observations'),
        'amtr_obs': ('N m cm**-3', 'angular momentum transport from observations'),
        'gflnc_e': ('cm**-3 s**-1', 'electron particle transport from neoclassical model'),
        'qflnc_e': ('W cm**-3', 'electron energy transport from neoclassical model'),
        'qflncc_e': ('W cm**-3', 'electron energy transport from NCLASS model'),
        'gflnc_i': ('cm**-3 s**-1', 'ion particle transport from neoclassical model'),
        'qflnc_i': ('W cm**-3', 'ion energy transport from neoclassical model'),
        'qflncc_i': ('W cm**-3', 'ion energy transport from NCLASS model'),
        'gflnc_x': ('cm**-3 s**-1', 'impurity particle transport from neoclassical model'),
        'qflnc_x': ('W cm**-3', 'impurity energy transport from neoclassical model'),
        'qflncc_x': ('W cm**-3', 'impurity energy transport from NCLASS model'),
        #'gflnc_d': ('cm**-3 s**-1', 'deuterium particle transport from neoclassical model'),
        #'qflnc_d': ('W cm**-3', 'deuterium energy transport from neoclassical model'),
        #'qflncc_d': ('W cm**-3', 'deuterium energy transport from NCLASS model'),
    }
    eq_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'rgrid': ('cm', 'radial grid points for equilibrium poloidal flux map'),
        'zgrid': ('cm', 'vertical grid points for equilibrium poloidal flux map'),
        'psirz': ('Wb rad**-1', 'equilibrium poloidal flux map'),
    }
    meta_vars: Final[Mapping[str, tuple[str, ...]]] = {
        'teped': ('eV', 'electron pedestal temperature'),
        'tiped': ('eV', 'ion pedestal temperature'),
        'neped': ('cm**-3', 'electron pedestal density'),
        'tepedw': ('', 'electron temperature pedestal width'),
        'tipedw': ('', 'ion temperature pedestal width'),
        'nepedw': ('', 'electron density pedestal width'),
        #'sc_teped': ('', 'electron pedestal temperature scaling factor'),
        #'sc_tiped': ('', 'ion pedestal temperature scaling factor'),
        #'sc_neped': ('', 'electron pedestal density scaling factor'),
        #'upwind_te': ('', 'electron energy balance upwind adjustment factor'),
        #'upwind_ti': ('', 'ion energy balance upwind adjustment factor'),
        #'upwind_mo': ('', 'momentum balance upwind adjustment factor'),
        #'upwind_d': ('', 'deuterium energy balance upwind adjustment factor'),
        #'sx_te': ('', 'electron energy balance solver range'),
        #'sx_ti': ('', 'ion energy balance solver range'),
        #'sx_ne': ('', 'electron particle balance solver range'),
        #'sx_omega': ('', 'angular momentum balance solver range'),
        #'nptriter': ('', 'number of Newton iterations in solution'),
        'tr_year': ('', 'year label of TRANSP release'),
        'tr_majv': ('', 'major version number of TRANSP release'),
        'trminv': ('', 'minor version number of TRANSP release'),
        'run': ('', 'run number'),
    }



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
            self.input = self._read_transp_ufile_file(path)
        else:
            self.output = self._read_transp_netcdf_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_transp_ufile_file(path, self.input, overwrite=overwrite)
        else:
            self._write_transp_netcdf_file(path, self.output, overwrite=overwrite)


    def _read_transp_netcdf_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        data = xr.Dataset()
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                temp_data = xr.open_dataset(path, engine='netcdf4')
                coords = {}
                for var in self.dim_vars.keys():
                    if var.upper() in temp_data:
                        temp_coord = temp_data[var.upper()].to_numpy()
                        while temp_coord.ndim > 1:
                            temp_coord = temp_coord[0, ...]
                        coords[var] = ([var], temp_coord, {'units': self.dim_vars[var][0], 'description': self.dim_vars[var][1]})
                subsets = [
                    self.coord_vars,
                    self.geom_vars,
                    self.magn_vars,
                    self.species_vars,
                    self.prof_vars,
                    self.global_vars,
                    self.transp_vars,
                    self.source_vars,
                    self.flux_vars,
                    self.eq_vars,
                    self.meta_vars,
                ]
                data_vars = {}
                for subset in subsets:
                    subset_temp_data = {var: temp_data[var.upper()] for var in list(subset.keys()) if var.upper() in temp_data}
                    subset_data = {}
                    for k, v in subset_temp_data.items():
                        new_dims = [dim.lower().replace('time3', 'time') for dim in v.dims]
                        if np.all([d in coords for d in new_dims]):
                            subset_data[k] = (new_dims, v.to_numpy(), {'units': subset[k][0], 'description': subset[k][1]})
                    data_vars.update(subset_data)
                max_asym_length = 0
                geom_asym_data = {}
                for k, v in self.geom_moment_vars.items():
                    geom_moments = []
                    for var in temp_data.keys():
                        if re.match(f'^{k.upper()}[0-9]+$', var):
                            geom_moments.append(var)
                    new_data = [temp_data[var].to_numpy() for var in sorted(geom_moments)]
                    if len(new_data) > 0:
                        new_dims = [dim.lower().replace('time3', 'time') for dim in temp_data[geom_moments[0]].dims]
                        if k.endswith('s') or k.endswith('sb'):
                            new_data = [np.zeros(new_data[0].shape)] + new_data
                        new_data = np.stack(new_data, axis=0)
                        max_asym_length = max(max_asym_length, new_data.shape[0])
                        geom_asym_data[k] = (new_dims, new_data)
                dtag = 'iasym'
                for k in geom_asym_data:
                    val = geom_asym_data[k][1]
                    while val.shape[0] < max_asym_length:
                        val = np.concatenate((val, np.zeros((1, *val.shape[1:]))), axis=0)
                    new_dims = [dtag] + geom_asym_data[k][0]
                    data_vars[k] = (new_dims, val, {'units': self.geom_moment_vars[k][0], 'description': self.geom_moment_vars[k][1]})
                    if dtag not in coords:
                        coords[dtag] = ([dtag], np.arange(max_asym_length), {'units': self.dim_vars[dtag][0], 'description': self.dim_vars[dtag][1]})
                # data = xr.Dataset(data_vars=data_vars, coords=coords)
                data = temp_data
        return data


    def _write_transp_netcdf_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not opath.exists() or overwrite:
                data.to_netcdf(opath, mode='w', format='NETCDF4')
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def _read_transp_ufile_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        raise NotImplementedError('TRANSP U-FILE read not yet implemented!')
    

    def _write_transp_ufile_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        raise NotImplementedError('TRANSP U-FILE write not yet implemented!')


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