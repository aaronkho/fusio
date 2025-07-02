# type: ignore
import json
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io

logger = logging.getLogger('fusio')


class astra_io(io):

    scalar_fields = [
        'AB',
        'ABC',
        'AIM1',
        'AIM2',
        'AIM3',
        'AMJ',
        'AWALL',
        'BTOR',
        'ELONG',
        'ELONM',
        'ENCL',
        'ENWM',
        'FECR',
        'FFW',
        'FICR',
        'FLH',
        'GN2E',
        'GN2I',
        'IPL',
        'LEXT',
        'NNCL',
        'NNWM',
        'QECR',
        'QFW',
        'QICR',
        'QLH',
        'QNBI',
        'RTOR',
        'SHIFT',
        'TRIAN',
        'TRICH',
        'UEXT',
        'UPDWN',
        'WNE',
        'WTE',
        'WTI',
        'ZMJ',
        'CF1',
        'CF2',
        'CF3',
        'CF4',
        'CF5',
        'CF6',
        'CF7',
        'CF8',
        'CF9',
        'CF10',
        'CF11',
        'CF12',
        'CF13',
        'CF14',
        'CF15',
        'CF16',
        'CV1',
        'CV2',
        'CV3',
        'CV4',
        'CV5',
        'CV6',
        'CV7',
        'CV8',
        'CV9',
        'CV10',
        'CV11',
        'CV12',
        'CV13',
        'CV14',
        'CV15',
        'CV16',
        'CHE1',
        'CHE2',
        'CHE3',
        'CHE4',
        'CHI1',
        'CHI2',
        'CHI3',
        'CHI4',
        'CNB1',
        'CNB2',
        'CNB3',
        'CNB4',
        'CNBI1',
        'CNBI2',
        'CNBI3',
        'CNBI4',
        'CCD1',
        'CCD2',
        'CCD3',
        'CCD4',
        'CRF1',
        'CRF2',
        'CRF3',
        'CRF4',
        'CNEUT1',
        'CNEUT2',
        'CNEUT3',
        'CNEUT4',
        'CPEL1',
        'CPEL2',
        'CPEL3',
        'CPEL4',
        'CBND1',
        'CBND2',
        'CBND3',
        'CBND4',
        'CFUS1',
        'CFUS2',
        'CFUS3',
        'CFUS4',
        'CIMP1',
        'CIMP2',
        'CIMP3',
        'CIMP4',
        'CMHD1',
        'CMHD2',
        'CMHD3',
        'CMHD4',
        'CRAD1',
        'CRAD2',
        'CRAD3',
        'CRAD4',
        'CSOL1',
        'CSOL2',
        'CSOL3',
        'CSOL4',
        'CSCL1',
        'CSCL2',
        'CSCL3',
        'CSCL4',
        'CSCL5',
        'CSCL6',
        'CSCL7',
        'CSCL8',
        'CDWM0',
        'CDWM1',
        'CDWM2',
        'CDWM3',
        'CDWM4',
        'CDWM5',
        'CDWM6',
        'CDWM7',
        'CDWM8',
        'CDWM9',
        'CDYM0',
        'CDYM1',
        'CDYM2',
        'CDYM3',
        'CDYM4',
        'CDYM5',
        'CDYM6',
        'CDYM7',
        'CDYM8',
        'CDYM9',
        'CDVM0',
        'CDVM1',
        'CDVM2',
        'CDVM3',
        'CDVM4',
        'CDVM5',
        'CDVM6',
        'CDVM7',
        'CDVM8',
        'CDVM9',
        'CDBC1',
        'CDBC2',
        'CDBC3',
        'CDBC4',
        'CDBC5',
        'CDBC6',
        'CDBC7',
        'CDBC8',
        'CDBC9',
        'CDJM1',
        'CDJM2',
        'CDJM3',
        'CDJM4',
        'CDJM5',
        'CDJM6',
        'CDJM7',
        'CDJM8',
        'CDJM9',
        'CDMJ1',
        'CDMJ2',
        'CDMJ3',
        'CDMJ4',
        'CDMJ5',
        'CDMJ6',
        'CDMJ7',
        'CDMJ8',
        'CDMJ9',
        'CDHJ1',
        'CDHJ2',
        'CDHJ3',
        'CDHJ4',
        'CDHJ5',
        'CDHJ6',
        'CDHJ7',
        'CDHJ8',
        'CDHJ9',
        'DROUT',
        'DTOUT',
        'DPOUT',
        'TIME',
        'TAUMIN',
        'TAUMAX',
        'TAUINC',
        'DELVAR',
        'ITEREX',
        'NITREQ',
        'TINIT',
        'TSCALE',
        'NB1',
        'NUF',
        'XOUT',
        'XINPUT',
        'NB2EQL',
        'NEQUIL',
        'NBND',
        'XFLAG',
        'DTEQL',
        'MEQUIL',
        'TPAUSE',
        'TEND',
        'INUME1',
        'INUME2',
        'INUME3',
        'INUME4',
        'IPROT',
        'ITFBE',
        'ITFBP',
        'ICIRCQ',
        'IPCTRL',
        'ADCMPF',
        'FLXDR',
        'SGNIP',
        'SGNBT',
        'IFBEG',
        'IPEQL',
        'TSTART',
        'TAU',
        'TAUPRP',
        'HRO',
        'HROX',
        'ALBPL',
        'NNCX',
        'QETB',
        'QFF0B',
        'QFF1B',
        'QFF2B',
        'QFF3B',
        'QFF4B',
        'QFF5B',
        'QFF6B',
        'QFF7B',
        'QFF8B',
        'QFF9B',
        'QITB',
        'QNNB',
        'FTO',
        'FTN',
        'BTN',
        'IPLN',
        'IFBEY',
        'ROB',
        'ROWALL',
        'ROC',
        'ROCO',
        'RON',
        'ROE',
        'ROI',
        'RO0',
        'RO1',
        'RO2',
        'RO3',
        'RO4',
        'RO5',
        'RO6',
        'RO7',
        'RO8',
        'RO9',
        'ROU',
        'VOLUME',
        'PSIAX',
        'PSIBO',
        'GP',
        'GP2',
        'PSIFB',
        'PSIFBO',
        'PSIEXT',
        'PSPLEX',
        'PSIEXO',
        'PSPLXO',
        'IPLFBE',
        'ATREQ',
        'PTREQ',
        'RBDOT',
        'BBDOT',
        'NA',
        'NA1',
        'NA1N',
        'NA1E',
        'NA1I',
        'NA1U',
        'NA10',
        'NA11',
        'NA12',
        'NA13',
        'NA14',
        'NA15',
        'NA16',
        'NA17',
        'NA18',
        'NA19',
        'NAB',
        'ITREQ',
        'NITOT',
        'NSTEPS',
        'QBEAM',
    ]
    array_fields = [
        'MUX',
        'MVX',
        'GNX',
        'SNX',
        'PEX',
        'PIX',
        'PRADX',
        'TEX',
        'TIX',
        'NEX',
        'CUX',
        'ZEFX',
        'VRX',
        'SHX',
        'ELX',
        'TRX',
        'G11X',
        'G22X',
        'G33X',
        'DRODAX',
        'IPOLX',
        'NIX',
        'VPOLX',
        'VTORX',
        'SLATX',
        'SHIVX',
        'SQUAX',
        'AIMPT',
        'AMAIN',
        'AMETR',
        'AREAT',
        'B0DB2',
        'BDB0',
        'BDB02',
        'BMAXT',
        'BMINT',
        'CC',
        'CD',
        'CE',
        'CI',
        'CN',
        'CNPAD',
        'CNPAP',
        'CNPAR',
        'CU',
        'CUBM',
        'CUBS',
        'CUECR',
        'CUFI',
        'CUFW',
        'CUICR',
        'CULH',
        'CUTOR',
        'CV',
        'DC',
        'DN',
        'DDNEO',
        'DDNEOD',
        'DIMP1',
        'DIMP2',
        'DIMP3',
        'DLNEO',
        'DLNEOD',
        'DRODA',
        'ELON',
        'EQFF',
        'EQPF',
        'ER',
        'F0',
        'F0O',
        'F1',
        'F1O',
        'F2',
        'F2O',
        'F3',
        'F3O',
        'F4',
        'F4O',
        'F5',
        'F5O',
        'F6',
        'F6O',
        'F7',
        'F7O',
        'F8',
        'F8O',
        'F9',
        'F9O',
        'FOFB',
        'FP',
        'FPO',
        'FP_NORM',
        'FV',
        'G11',
        'G22',
        'G22E',
        'G33',
        'G33E',
        'G41',
        'G42',
        'G43',
        'G44',
        'G45',
        'GN',
        'GRADRO',
        'HC',
        'HE',
        'IPOL',
        'MRHO',
        'MU',
        'MV',
        'NALF',
        'NDEUT',
        'NE',
        'NEO',
        'NHE3',
        'NHYDR',
        'NI',
        'NIBM',
        'NIMPT',
        'NIO',
        'NIZ1',
        'NIZ2',
        'NIZ3',
        'NMAIN',
        'NN',
        'NNBM1',
        'NNBM2',
        'NNBM3',
        'NRATE',
        'NTRIT',
        'PBEAM',
        'PBLON',
        'PBOL1',
        'PBOL2',
        'PBOL3',
        'PBPER',
        'PDE',
        'PDI',
        'PE',
        'PEBM',
        'PEECR',
        'PEFW',
        'PEICR',
        'PEIQI',
        'PELH',
        'PELON',
        'PEPER',
        'PERIM',
        'PETOT',
        'PFAST',
        'PI',
        'PIBM',
        'PIFW',
        'PIICR',
        'PITOT',
        'PRAD',
        'PRES',
        'PSXR1',
        'PSXR2',
        'PSXR3',
        'QE',
        'QF0',
        'QF1',
        'QF2',
        'QF3',
        'QF4',
        'QF5',
        'QF6',
        'QF7',
        'QF8',
        'QF9',
        'QI',
        'QN',
        'QU',
        'RHO',
        'RHO_POL',
        'RUPAR',
        'RUPFR',
        'RUPYR',
        'SCUBM',
        'SD0',
        'SD1',
        'SD2',
        'SD3',
        'SD4',
        'SD5',
        'SD6',
        'SD7',
        'SD8',
        'SD9',
        'SDN',
        'SF0TOT',
        'SF1TOT',
        'SF2TOT',
        'SF3TOT',
        'SF4TOT',
        'SF5TOT',
        'SF6TOT',
        'SF7TOT',
        'SF8TOT',
        'SF9TOT',
        'SGNEO',
        'SGNEOD',
        'SHEAR',
        'SHIF',
        'SHIV',
        'SLAT',
        'SN',
        'SNEBM',
        'SNIBM1',
        'SNIBM2',
        'SNIBM3',
        'SNNBM',
        'SNTOT',
        'SQEPS',
        'SQUARN',
        'SRHO',
        'SXHO',
        'TE',
        'TEO',
        'TI',
        'TIO',
        'TN',
        'TRIA',
        'TTRQ',
        'TTRQI',
        'ULON',
        'UPAR',
        'UPARO',
        'UPL',
        'UPS0',
        'UPS0O',
        'UPS1',
        'UPS1O',
        'UPS2',
        'UPS2O',
        'VIMP1',
        'VIMP2',
        'VIMP3',
        'VOLUM',
        'VP',
        'VPFP',
        'VPOL',
        'VR',
        'VRO',
        'VRS',
        'VTOR',
        'XC',
        'XI',
        'XRHO',
        'XUPAD',
        'XUPAP',
        'XUPAR',
        'ZEF',
        'ZEF1',
        'ZEF2',
        'ZEF3',
        'ZIM1',
        'ZIM2',
        'ZIM3',
        'ZIMPT',
        'ZMAIN',
    ]
    ufile_preamble = [
        '-SHOT #- F(X) DATA',
        '-SHOT DATE-  UFILES ASCII FILE SYSTEM',
        '-NUMBER OF ASSOCIATED SCALAR QUANTITIES-',
        '-INDEPENDENT VARIABLE LABEL: X1-',
        '-INDEPENDENT VARIABLE LABEL: X0-',
        '-DEPENDENT VARIABLE LABEL-',
        '-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM',
        '-# OF  X1 PTS-',
        '-# OF  X0 PTS-',
    ]
    ufile_postamble = [
        '----END-OF-DATA-----------------COMMENTS:-----------',
    ]
    ufile_profs = {
        'te': ('Electron Temp', 'eV'),
        'ti': (),
        'ne': ('Electron Density', 'cm**-3'),
        'q': ('EFIT q profile', ''),
    }
    ufile_bdrys = {
        'rbndry': ('R BOUNDARY', 'm'),
        'zbndry': ('Z BOUNDARY', 'm'),
    }
    

    def __init__(self, *args, **kwargs):
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


    def read(self, path, side='output'):
        if side == 'input':
            #self.input = self._read_astra_file(path)
            logger.warning(f'{self.format} input reading function not defined yet...')
        else:
            self.output = self._read_astra_json_dir(path)


    def write(self, path, side='input', overwrite=False):
        #if side == 'input':
        #    self._write_astra_ufile(path, self.input, overwrite=overwrite)
        #else:
        #    self._write_astra_file(path, self.output, overwrite=overwrite)
        logger.warning(f'{self.format} writing function not defined yet...')


    def _read_astra_ufile_dir(self, path):

        coords = {}
        data_vars = {}
        attrs = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():
                for ufile_path in ipath.glob('*.ufile'):
                    lines = []
                    with open(ufile_path, 'r') as ufile:
                        lines = ufile.readlines()
                    preamble = [lines.pop(0) for istr in self.ufile_preamble]
                    postamble = [lines.pop(-1) for istr in self.ufile_postamble][::-1]
                    profile = lines
                    vtag = preamble[5][:22].strip()
                    xvar = None
                    yvar = None
                    for key, val in ufile_profs.items():
                        if vtag == val[0]:
                            xvar = 'rho_tor'
                            yvar = key
                    for key, val in ufile_bdrys.items():
                        if vtag == val[0]:
                            xvar = 'pos_bndry'
                            yvar = key
                    if xvar is not None and yvar is not None:
                        fsplit = False
                        profile_x = []
                        profile_y = []
                        for line in profile[1:]:
                            if fsplit:
                                profile_y.append(line.strip())
                            else:
                                profile_x.append(line.strip())
                        xval = np.fromstring(' '.join(profile_x))
                        yval = np.fromstring(' '.join(profile_y))
                        if f'{xvar}' not in coords:
                            coords[f'{xvar}'] = copy.deepcopy(xval)
                        data_vars[f'{yvar}'] = ([f'{xvar}'], copy.deepcopy(yval))

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _read_astra_json_dir(self, path):

        ds = xr.Dataset()
        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():
                dsv = []
                for slice_path in ipath.glob('*.json'):
                    coords = {}
                    data_vars = {}
                    attrs = {}
                    with open(slice_path, 'r') as jsonfile:
                        jsondata = json.load(jsonfile)
                    astra_data = jsondata.get('astra', {})
                    time = astra_data.pop('TIME', {})
                    val = np.array([time.get('data', 0.0)]).squeeze()
                    attr = {'units': time.get('units', ''), 'description': time.get('long_name', '')}
                    coords['time'] = (['time'], np.expand_dims(val, axis=0), attr)
                    xrho = astra_data.pop('XRHO', {})
                    val = np.array(xrho.get('data', []))
                    attr = {'units': xrho.get('units', ''), 'description': xrho.get('long_name', '')}
                    coords['xrho'] = (['xrho'], val, attr)
                    for key in list(astra_data.keys()):
                        if key in self.scalar_fields:
                            field = astra_data.pop(key, {})
                            attr = {'units': field.get('units', ''), 'description': field.get('long_name', '')}
                            val = np.array([field.get('data', 0.0)]).squeeze()
                            data_vars[key.lower()] = (['time'], np.expand_dims(val, axis=0), attr)
                        elif key in self.array_fields:
                            field = astra_data.pop(key, {})
                            attr = {'units': field.get('units', ''), 'description': field.get('long_name', '')}
                            val = np.array(field.get('data', []))
                            if len(val) > 0:
                                data_vars[key.lower()] = (['time', 'xrho'], np.expand_dims(val, axis=0), attr)
                    dsv.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs))
                ds = xr.concat(dsv, dim='time').sortby('time')

        return ds


    def _write_astra_ufile_dir(self, path, data, overwrite=False):
        nscalar = 0
        time_label = 'Time'
        time_units = 'Seconds'
        pos_label = 'POSITION'
        pos_units = ''
        rho_label = 'rho_tor'
        rho_units = ''
        for var in self.ufile_profs:
            if var in data and 'rho_tor' in data:
                var_label, var_units = self.ufile_profs[var]
                rho = data['rho_tor'].to_numpy()
                val = data[var].to_numpy()
                ntime = 1
                nrho = len(rho)
                dstr = [
                    f'{tag:<10} 2 0 6',  # What do these 3 integers mean?
                    f'',
                    f'{nscalar:3d}',
                    f'{time_label:<20}{time_units:<10}',
                    f'{rho_label:<20}{rho_units:<10}',
                    f'{var_label:<20}{var_units:10}',
                    f'{ncode}',
                    f'{ntime:10d}',
                    f'{nrho:10d}',
                ]
                preamble = [f' {d:<30};{u}' for d, u in zip(dstr, self.ufile_preamble)]
                profile_x = []
                profile_y = []
                ii = 0
                while ii < nrho:
                    jj = ii + 6 if (ii + 6) < nrho else None
                    profile_x.append(' ' + ' '.join([f'{num:.6e}' for num in rho[ii:jj]]))
                    profile_y.append(' ' + ' '.join([f'{num:.6e}' for num in val[ii:jj]]))
                    ii = jj if jj is not None else nrho
                dummy = 0.0
                profile = [' {dummy:.6e}'] + profile_x + [' '] + profile_y
                postamble = [f' ;{u};' for u in self.ufile_postamble]
                ufile_data = preamble + profile + postamble
                with open(path / f'{var}_astra.ufile', 'w') as uf:
                    uf.write('\n'.join(ufile_data))
        for var in self.ufile_bdrys:
            if var in data and 'pos_bndry' in data:
                var_label, var_units = self.ufile_bdrys[var]
                pos = data['pos_bndry'].to_numpy()
                val = data[var].to_numpy()
                ntime = 1
                npos = len(pos)
                dstr = [
                    f'{tag:<10} 2 0 6',  # What do these 3 integers mean?
                    f'',
                    f'{nscalar:3d}',
                    f'{time_label:<20}{time_units:<10}',
                    f'{pos_label:<20}{pos_units:<10}',
                    f'{var_label:<20}{var_units:10}',
                    f'{ncode}',
                    f'{ntime:10d}',
                    f'{npos:10d}',
                ]
                preamble = [f' {d:<30};{u}' for d, u in zip(dstr, self.ufile_preamble)]
                profile_x = []
                profile_y = []
                ii = 0
                while ii < npos:
                    jj = ii + 6 if (ii + 6) < npos else None
                    profile_x.append(' ' + ' '.join([f'{num:.6e}' for num in rho[ii:jj]]))
                    profile_y.append(' ' + ' '.join([f'{num:.6e}' for num in val[ii:jj]]))
                    ii = jj if jj is not None else npos
                dummy = 0.0
                profile = [' {dummy:.6e}'] + profile_x + [' '] + profile_y
                postamble = [f' ;{u};' for u in self.ufile_postamble]
                ufile_data = preamble + profile + postamble
                with open(path / f'{var}_astra.ufile', 'w') as uf:
                    uf.write('\n'.join(ufile_data))
    

    def _write_astra_json_dir(self, path, data, overwrite=False):
        pass


    @classmethod
    def from_rundir(cls, path=None, input=None, output=None):
        if isinstance(input, (str, Path)):
            input = Path(input)
            if input.name != 'udb' and (input / 'udb').is_dir():
                input = input / 'udb'
        elif isinstance(path, (str, Path)):
            input = Path(path) / 'udb'
        if isinstance(output, (str, Path)):
            output = Path(output)
            if output.name != 'ncdf_out' and (output / 'ncdf_out').is_dir():
                output = output / 'ncdf_out'
        elif isinstance(path, (str, Path)):
            output = Path(path) / 'ncdf_out'
        return cls(input=input, output=output)  # Places data into output side unless specified


