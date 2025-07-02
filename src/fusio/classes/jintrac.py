# type: ignore
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
import re
import struct
import copy
from itertools import tee
from .io import io
from ..utils.plasma_tools import define_ion_species

logger = logging.getLogger('fusio')


class jintrac_io(io):

    dtypes: Final[Mapping[str, Any]] = {
        'int': 'i',
        'float': 'f',
        'double': 'd',
        'char': None,
    }
    dsizes: Final[Mapping[str, int]] = {
        'int': 4,
        'float': 4,
        'double': 8,
    }
    byte_orderingtag: Final[Mapping[str, str]] = {
        '>': 'Big-endian',
        '<': 'Little-endian',
    }
    byte_ordering: Final[Mapping[str, str]] = {
        'b': '>',
        'l': '<',
    }

    header_finder: Final[re.Pattern] = re.compile(r'\*\n\*[\w, \d]+\n'.encode('utf-8'))
    spec_finder: Final[re.Pattern] = re.compile(r'#\w+;\d+;\d+;[\w, \d\-]+;\d+\n'.encode('utf-8'))
    tracking_conversion: Final[str, int] = {
        'File Header': 0,
        'General Info': 0,
        'PPF Attributes': 1,
        'PPF Base Vectors': 2,
        'Profiles': 3,
        'Traces': -1,
        '': -1,
    }
    tracking_tags: Final[Sequence[str]] = [
        ' in file header',
        ' in provenance section',
        ' in base vector section',
        ' in time slice',
        '',
    ]
    supported_file_extensions: Final[Sequence[str]] = [
        '.ex',
        '.ext',
        '.jsp',
        '.jst',
        '.jss',
        '.jse',
        '.jhp',
        '.jht',
        '.ssp',
        '.sst',
        '.jasp',
        '.jast',
        '.jsd',
    ]
    metadata_tags: Final[Sequence[str]] = [
        'units',
        'desc',
        'scstr',
        'xbase',
        'uid',
        'dda',
        'dtype',
        'seq',
    ]


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


    def _pairwise(
        self,
        iterable: Iterable,
    ) -> Iterable:
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


    def _decode_spec(
        self,
        spec: str,
    ) -> Sequence[str, str, int, int]:
        if not spec.startswith('#'):
            logger.error(f'{spec} is not a spec line')
        splitted = spec[1:-1].split(';')
        if len(splitted) != 5:
            logger.error(f'{spec} is not a spec line')
        dformat = splitted[0]
        ignored = splitted[1]
        npoints = int(splitted[2])
        label = splitted[3]
        nlines = int(splitted[4])
        return label, dformat, npoints, nlines


    def _decode_metadata(
        self,
        raw: Sequence[bytes],
    ) -> MutableMapping[str, Any]:
        itag = None
        decoded = list(map(lambda x: x.decode('latin-1').strip(), raw))
        meta = dict(zip(self.metadata_tags, decoded))
        if 'scstr' in meta:
            meta['scale'] = float(meta['scstr'].strip())
        if 'units' in meta:
            if meta['units'] == '':
                meta['units'] = None
        return meta


    def _decode_block(
        self,
        block: Sequence[bytes],
        endianness: str = '>',
        track_num: int = -1,
    ) -> Sequence[str, Any, Any, bool]:
        # A block is the entire portion between spec lines, and typically contain the data for a single variable
        loc = self.tracking_tags[track_num] if track_num < 3 else self.tracking_tags[3]
        if track_num >= 3:
            loc += f' number {track_num - 3:d}'
        corrupted = False
        # Read specification
        m = self.spec_finder.match(block)
        spec_string = block[:m.end()]
        block_string = block[m.end():]
        spec = spec_string.decode('latin-1')
        var_label, dformat, npoints, nlines = self._decode_spec(spec)
        var_format = self.dtypes[dformat]
        spec = spec.strip()
        # Read metadata
        try:
            splitted = re.split(b'(\n)', block_string)
            raw_metadata = splitted[:2*nlines:2]
            raw_data = b''.join(splitted[2*nlines:-2])
        # Protection against invalid specification line
        except:
            corrupted = True
            raw_metadata = None
            raw_data = None
            logger.warning(f'Could not split {var_label} block with spec {spec}{loc}')
        try:
            var_metadata = self._decode_metadata(raw_metadata)
            var_metadata['label'] = var_label
            var_metadata['form'] = dformat
        # Protection against incorrect number of metadata lines
        except:
            corrupted = True
            var_metadata = {}
            logger.warning(f'Could not decode {var_label} metadata with spec {spec}{loc}')
        # Read data
        if var_format is None:
            # No decoding needed, just read
            try:
                decoded_var = raw_data.decode('latin-1').strip()
            # Protection against plain read failure (highly unlikely)
            except:
                corrupted = True
                decoded_var = None
        else:
            if len(raw_data) == 0:
                # Empty variable
                if var_format in ['i', 'f', 'd']:
                    # Empty float array with a shape, fill with nans
                    decoded_var = np.full(npoints, np.nan)
            else:
                try:
                    decoded_var = struct.unpack(endianness + npoints * var_format, raw_data)
                    if var_format is None:
                        decoded_var = decoded_var[0]
                    elif var_format in ['i', 'f', 'd']:
                        decoded_var = np.array(decoded_var)
                # Protection against incorrect specification of binary line length / format
                except:
                    corrupted = True
                    decoded_var = np.full(npoints, np.nan)
                    logger.warning(f'Could not decode {var_label} data with spec {spec}{loc}')
        # Scale numerical data based on metadata value
        if 'scale' in var_metadata:
            try:
                decoded_var *= var_metadata['scale']
            # Protection against invalid value in scaling factor metadata
            except:
                corrupted = True
                decoded_var = np.full(npoints, np.nan)
                logger.warning(f'Could not rescale {var_label}{loc}')
        var_label = re.sub(' ', '_', var_label)
        return var_label.lower(), decoded_var, var_metadata, corrupted


    def _decode_section(self, section, sec_num, endianness='>', metadata=None, data_start_num=-1):
        # A section is the entire portion between lines beginning with **
        known_info = metadata if isinstance(metadata, dict) else {}
        m = self.header_finder.match(section)
        header_string = section[:m.end()]
        section_string = section[m.end():]
        section_header = header_string[3:-1].decode('latin-1')
        # Look for all data specs in the section (starts with #dtype;....)
        spec_starts = [m.start() for m in self.spec_finder.finditer(section_string)]
        spec_starts.append(len(section_string)) # To also read the last block
        blocks = [section_string[start:next_start] for start, next_start in self._pairwise(spec_starts)]
        if b''.join(blocks) != section_string:
            logger.error('Something weird happened, did not split all blocks correctly')
        section_data = {}
        section_info = {}
        track_num = self.tracking_conversion[section_header] if section_header in self.tracking_conversion else -1
        if track_num == 3 and data_start_num > 0:
            track_num += (sec_num - data_start_num)
        for block_num, block in enumerate(blocks):
            var_label, block_data, var_info, corrupted = self._decode_block(block, endianness=endianness, track_num=track_num)
            record = True
            # Check if variable metadata is already known (i.e. corrupted entry is not in the first time slice)
            if corrupted and var_label in known_info:
                var_info = metadata[var_label]
            # Check if data already exists (i.e. duplicate entry in section)
            if var_label in section_info and var_label in section_data:
                record = False
                # Check if existing data is already good, if yes, keep it
                good_entry = ('xbase' in section_info[var_label] and section_data[var_label] is not None and np.all(np.isfinite(section_data[var_label])))
                if not good_entry and not corrupted:
                    record = True
            if record:
                section_data[var_label] = block_data
                section_info[var_label] = var_info
                section_info[var_label]['section_number'] = sec_num + 1
        return section_header, section_data, section_info


    def _read_jintrac_file(self, path):

        coords = {}
        data_vars = {}
        attrs = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if not ipath.is_file():
                logging.error(f'File {ipath.absolute()} not found. Abort.')
            if ((ipath.suffix not in supported_file_extensions) and
                (not ipath.suffix.startswith('.ssp')) and
                (not ipath.suffix.startswith('.sst')) and
                (not ipath.suffix.startswith('.jsp'))
            ):
                logger.error(f'Extension of file {ipath.absolute()} not in allowed list. Abort.')

            #if isinstance(output_file,str):
            #    convert_binary_file(str(ipath.absolute()), output_file)

            data = {}
            data['info'] = {}

            with open(ipath, 'rb') as bf:
                indata = bf.read()

            # First split the huge string into sections. A section starts with *\n*HEADER
            section_header_starts = [m.start() for m in self.header_finder.finditer(indata)]
            section_header_starts.append(len(indata)) # Also read the last section
            sections = [indata[start:next_start] for start, next_start in self._pairwise(section_header_starts)]
            if b''.join(sections) != indata:
                logger.error(f'Something weird happened, did not split all sections correctly')

            # First section has info we need to decode the rest
            section_header, section_data, header_info = self._decode_section(sections[0], 0)
            data[section_header] = section_data
            endianness = self.byte_ordering[section_data['file_format']]
            data_start_num = -1

            # Now decode all sections
            for sec_num, section in enumerate(sections[1:-1], start=1):
                metadata = data['info'] if data['info'] else None
                if 'Profiles' in data and data_start_num < 0:
                   data_start_num = sec_num - 1
                section_header, section_data, section_info = self._decode_section(
                    section,
                    sec_num,
                    endianness=endianness,
                    metadata=metadata,
                    data_start_num=data_start_num
                )
                if section_header == 'Profiles':
                    if section_header not in data:
                        data[section_header] = []
                    data[section_header].append(section_data)
                else:
                    data[section_header] = section_data
                for k, v in section_info.items():
                    if k not in data['info']:
                         data['info'][k] = v

            # Check length of 1D profile arrays
            if 'Profiles' in data:
                profiles = {}
                for slice_num, prof in enumerate(data['Profiles']):
                    for key in prof.keys():
                        if key not in profiles:
                            profiles[key] = []
                        prof_data = copy.deepcopy(prof[key])
                        if 'PPF Base Vectors' in data and 'info' in data and key in data['info'] and 'xbase' in data['info'][key]:
                            if not isinstance(prof_data, np.ndarray):
                                prof_data = np.full(data['PPF Base Vectors'][data['info'][key]['xbase']].size, np.nan)
                            if prof_data.size != data['PPF Base Vectors'][data['info'][key]['xbase']].size:
                                prof_data = np.full(data['PPF Base Vectors'][data['info'][key]['xbase']].size, np.nan)
                        if len(profiles[key]) != slice_num:
                            fill_data = np.full(prof_data.shape, np.nan)
                            profiles[key].append(fill_data)
                            logger.warning(f'Missing {key} data in time slice number {slice_num - 1:d}')
                        profiles[key].append(prof_data)
                # Merge all 1D profiles of like quantities together into 2D arrays
                for key in list(profiles.keys()):
                    profiles[key] = np.stack(profiles[key])
                data['Profiles'] = profiles

            # Check length of 1D time arrays
            if 'Traces' in data:
                traces = {}
                for key in data['Traces'].keys():
                    if key not in traces:
                        traces[key] = []
                    trac_data = copy.deepcopy(data['Traces'][key])
                    if 'PPF Base Vectors' in data and 'info' in data and key in data['info'] and 'xbase' in data['info'][key]:
                        if not isinstance(trac_data, np.ndarray):
                            trac_data = np.full(data['PPF Base Vectors'][data['info'][key]['xbase']].size, np.nan)
                        if trac_data.size != data['PPF Base Vectors'][data['info'][key]['xbase']].size:
                            trac_data = np.full(data['PPF Base Vectors'][data['info'][key]['xbase']].size, np.nan)
                    traces[key] = trac_data
                data['Traces'] = traces

            data = self._remove_duplicate_times(data, keep='last')
            data = self._standardize_data_representation(data, header_info, endianness)

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_jintrac_binary_file(self, data, output_file='custom'):

        if not isinstance(data, dict):
            raise TypeError('Invalid input data structure, must be a dictionary. Abort.')
        elif 'info' not in data or 'sections' not in data:
            raise TypeError('Input data dictionary not correctly formatted. Abort.')

        di = data['info']
        ds = data['sections']
        baseidx = ds.index('PPF Base Vectors') if 'PPF Base Vectors' in ds else 3
        ofname = 'custom.ex'
        if isinstance(output_file, str):
            ofname = output_file
        if 'Profiles' in ds and not ofname.endswith('.ex'):
            ofname = ofname + '.ex'
        if 'Traces' in ds and not ofname.endswith('.ext'):
            ofname = ofname + '.ext' if not ofname.endswith('.ex') else ofname + 't'
        opath = Path(ofname)
        if not opath.parent.exists():
            opath.parent.mkdir(parents=True)

        status = 1
        bord = self.byte_ordering[data.get('file_format', 'b')]
        with open(str(opath.absolute()), 'wb') as ofile:
            NL = '\n'.encode()
            for ii in range(1, len(ds) + 1):

                if not re.match('^Profiles$', ds[ii-1], flags=re.IGNORECASE):
                    hstr1 = '*'
                    ofile.write(hstr1.encode() + NL)
                    hstr2 = '*' + ds[ii-1]
                    ofile.write(hstr2.encode() + NL)
                    for key in data:
                        if 'section_number' in di[key] and di[key]['section_number'] == ii:
                            nstr = '1'
                            rnum = 0
                            if not re.match(r'^char$', di[key]['form'], flags=re.IGNORECASE):
                                nstr = f'{data[key].shape[1]:d}'
                                rnum = -1
                            for ii, key2 in enumerate(self.metadata_tags):
                                if key2 in di[key]:
                                    rnum = ii + 1
                            rstr = f'{rnum:d}' if rnum >= 0 and not re.match(r'^Shot$', key, flags=re.IGNORECASE) else '0'
                            hdelim = ';'
                            fstr = '#' + hdelim.join((di[key]['form'], '1', nstr, di[key]['label'], rstr))
                            ofile.write(fstr.encode() + NL)
                            for ii, key2 in enumerate(self.metadata_tags):
                                if rnum > ii:
                                    estr = di[key].get(key2, None)
                                    if not isinstance(estr, str):
                                        estr = ''
                                    ofile.write(estr.encode() + NL)
                            if rnum == 0:
                                ofile.write(data[key].encode() + NL)
                            else:
                                scale = di[key].get('scale', 1.0)
                                dtype = self.dtypes[di[key]['form']]
                                dsize = self.dsizes[di[key]['form']]
                                for jj in range(data[key].shape[1]):
                                    dstr = b''
                                    if re.match(r'^[fd]$', dtype, flags=re.IGNORECASE):
                                        dstr = struct.pack(bord+dtype, float(data[key][0, jj] / scale))
                                    else:
                                        dstr = struct.pack(bord+dtype, int(data[key][0, jj] / scale))
                                    ofile.write(dstr)
                                ofile.write(NL)

                elif 'tvec1' in data and isinstance(data['tvec1'], np.ndarray):
                    for tt in range(data['tvec1'].shape[0]):
                        hstr1 = '*'
                        ofile.write(hstr1.encode() + NL)
                        hstr2 = '*' + ds[ii-1]
                        ofile.write(hstr2.encode() + NL)
                        for key in data:
                            if 'section_number' in di[key] and di[key]['section_number'] == ii:
                                nstr = '1'
                                rnum = 0
                                if not re.match(r'^char$', di[key]['form'], flags=re.IGNORECASE):
                                    nstr = f'{data[key].shape[1]:d}'
                                    rnum = -1
                                for ii, key2 in enumerate(self.metadata_tags):
                                    if key2 in di[key]:
                                        rnum = ii + 1
                                rstr = f'{rnum:d}' if rnum >= 0 else '0'
                                hdelim = ';'
                                fstr = '#' + hdelim.join((di[key]['form'], '1', nstr, di[key]['label'], rstr))
                                ofile.write(fstr.encode() + NL)
                                for ii, key2 in enumerate(self.metadata_tags):
                                    if rnum > ii:
                                        estr = di[key].get(key2, None)
                                        if not isinstance(estr, str):
                                            estr = ''
                                        ofile.write(estr.encode() + NL)
                                if rnum == 0:
                                    ofile.write(data[key].encode() + NL)
                                else:
                                    scale = di[key].get('scale', 1.0)
                                    dtype = dtypes[di[key]['form']]
                                    dsize = dsizes[di[key]['form']]
                                    for jj in range(data[key].shape[1]):
                                        dstr = b''
                                        if re.match(r'^[fd]$', dtype, flags=re.IGNORECASE):
                                            if data[key].shape[0] > tt:
                                                dstr = struct.pack(bord+dtype, float(data[key][tt, jj] / scale))
                                            else:
                                                dstr = struct.pack(bord+dtype, float(data[key][-1, jj] / scale))
                                                logger.warning(f'   Less time slices than expected in field: {di[key]["label"]:10}')
                                        else:
                                            if data[key].shape[0] > tt:
                                                dstr = struct.pack(bord+dtype, int(data[key][tt, jj] / scale))
                                            else:
                                                dstr = struct.pack(bord+dtype, int(data[key][-1, jj] / scale))
                                                logger.warning(f'   Less time slices than expected in field: {di[key]["label"]:10}')
                                        ofile.write(dstr)
                                    ofile.write(NL)

            hstr1 = '*'
            ofile.write(hstr1.encode() + NL)
            hstr2 = '*EOF'
            ofile.write(hstr2.encode() + NL)
            if ii == len(ds):
                status = 0

        return status


    def _remove_duplicate_times(self, data, keep='first'):
        time_vector = None
        if 'Traces' in data and 'PPF Base Vectors' in data and 'tvec1' in data['PPF Base Vectors']:
            time_vector = data['PPF Base Vectors']['tvec1'].flatten()
        if 'Profiles' in data and 'time' in data['Profiles']:
            time_vector = data['Profiles']['time'].flatten()
        if time_vector is not None:
            full_length = len(time_vector)
            unique_values, unique_indices = np.unique(time_vector, return_index=True)
            if len(unique_indices) != full_length:
                if keep == 'last':
                    index_diff = np.hstack((np.diff(unique_indices), 1)).astype(np.int64)
                    unique_indices = unique_indices + index_diff - 1
                if 'Traces' in data:
                    data['PPF Base Vectors']['tvec1'] = np.take(data['PPF Base Vectors']['tvec1'], unique_indices, axis=0)
                    for key in data['Traces']:
                        data['Traces'][key] = np.take(data['Traces'][key], unique_indices, axis=0)
                if 'Profiles' in data:
                    for key in data['Profiles']:
                        data['Profiles'][key] = np.take(data['Profiles'][key], unique_indices, axis=0)
        return data


    def _standardize_data_representation(self, data, header_info, endianness):
        data['sections'] = [key for key in data.keys() if key != 'info']
        for key in data['sections']:
            dateval = None
            timeval = None
            if key == 'File Header':
                dateval = data['File Header'].pop('date')
                timeval = data['File Header'].pop('time')
            data.update(data.pop(key))
            if dateval is not None:
                data['creation_date'] = dateval
            if timeval is not None:
                data['creation_time'] = timeval
        for timename in ['time', 'tvec1']:
            if timename in data:
                data[timename] = np.expand_dims(data[timename], axis=-1)
        dateinfo = header_info.pop('date')
        timeinfo = header_info.pop('time')
        data['info'].update(header_info)
        data['info']['file_format']['fullname'] = self.byte_orderingtag[endianness]
        data['info']['creation_date'] = dateinfo
        data['info']['creation_time'] = timeinfo
        data['info']['info'] = {'desc': 'Additional information on data fields'}
        data['info']['sections'] = {'desc': 'Labelled section and order within EX-FILE'}
        for key, val in data.items():
            if key not in ['info', 'sections']:
                if isinstance(val, np.ndarray) and val.ndim < 2:
                    data[key] = np.atleast_2d(val)
        return data


    @classmethod
    def create_empty_structure(cls, database, shot, version_tag=None, extfile=False):

        # Required user input - forced crash if improper
        if not isinstance(database, str):
            raise ValueError('Database field for ex-file generation must be a string')
        if not isinstance(shot, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
            raise ValueError('Shot number field for ex-file generation must be numeric')
        datablock_tag = 'Traces' if extfile else 'Profiles'

        # Initialize structure
        data = {}
        data['info'] = {}
        data['sections'] = ['File Header', 'General Info', 'PPF Attributes', 'PPF Base Vectors', datablock_tag]

        # Add standard descriptions of required metadata - covers until end of 'PPF Attributes' section
        data['info']['info'] = {'desc': 'Additional information on data fields'}
        data['info']['sections'] = {'desc': 'Labelled section and order within ex-file'}
        data['info']['file_format'] = {'form': 'char', 'fullname': 'Big-endian', 'label': 'File Format', 'section_number': 1}
        data['info']['file_description'] = {'form': 'char', 'label': 'File Description', 'section_number': 1}
        data['info']['version'] = {'form': 'char', 'label': 'Version', 'section_number': 1}
        data['info']['creation_date'] = {'form': 'char', 'label': 'Date', 'section_number': 1}
        data['info']['creation_time'] = {'form': 'char', 'label': 'Time', 'section_number': 1}
        data['info']['database_name'] = {'form': 'char', 'label': 'Database Name', 'section_number': 2}
        data['info']['user_ex-file'] = {'form': 'char', 'label': 'User EX-file', 'section_number': 2}
        data['info']['user_pre-modex_ex-file'] = {'form': 'char', 'label': 'User Pre-Modex EX-file', 'section_number': 2}
        data['info']['shot'] = {'form': 'int', 'label': 'Shot', 'section_number': 3}
        data['info']['dda_name'] = {'form': 'char', 'label': 'DDA Name', 'section_number': 3}
        if not extfile:
            data['info']['xvec1'] = {'units': None, 'desc': 'RHO normalised', 'scstr': '1.0', 'scale': 1.0, 'label': 'XVEC1', 'form': 'float', 'section_number': 4}
            data['info']['tvec1'] = {'units': 'secs', 'desc': 'TIME', 'scstr': '1.0', 'scale': 1.0, 'label': 'TVEC1', 'form': 'float', 'section_number': 5}
        else:
            data['info']['tvec1'] = {'units': 'secs', 'desc': 'TIME', 'scstr': '1.0', 'scale': 1.0, 'label': 'TVEC1', 'form': 'float', 'section_number': 4}

        # Add required metadata - covers until end of 'PPF Attributes' section
        data['file_format'] = 'b'
        data['file_description'] = 'EX-FILE'
        data['version'] = version_tag if isinstance(version_tag, str) else 'fusio-0.1'
        data['creation_date'] = datetime.datetime.now().strftime("%d/%m/%Y")
        data['creation_time'] = datetime.datetime.now().strftime("%H:%M:%S")
        data['database_name'] = database
        data['user_ex-file'] = ''
        data['user_pre-modex_ex-file'] = ''
        data['shot'] = np.atleast_2d([int(shot)])
        data['dda_name'] = 'FUSIO'
        if not extfile:
            data['xvec1'] = np.atleast_2d([])
            data['tvec1'] = np.atleast_3d([])
        else:
            data['tvec1'] = np.atleast_2d([])

        return data


#def add_entry(data, key, moddata, dtype, units, description, scale, xbase, tag=None):
#    ikey = None
#    idata = None
#    ibase = None
#    itag = 'Python Addition Tool'
#    if isinstance(key, str):
#        ikey = key
#    if isinstance(moddata, (list, tuple, np.ndarray)):
#        idata = np.atleast_2d(moddata)
#    if isinstance(xbase, str):
#        ibase = xbase
#    if isinstance(tag, str):
#        itag = itag + ' - ' + tag
#
#    odata = copy.deepcopy(data)
#    if isinstance(data, dict) and ikey is not None:
#        check_flag = True
#        if ikey not in ["TVEC1", "XVEC1"]:
#            check_array = [False] * idata.ndim
#            for ii in range(len(check_array)):
#                if "TVEC1" in data and idata.shape[ii] == data["TVEC1"].size:
#                    check_array[ii] = True
#                if "XVEC1" in data and idata.shape[ii] == data["XVEC1"].size:
#                    check_array[ii] = True
#                if idata.shape[ii] == 1:
#                    check_array[ii] = True
#            check_flag = all(check_array)
#        if check_flag:
#            if ikey not in data:
#                uname = getuser()
#                odata[ikey] = idata.copy()
#                odata["INFO"][ikey] = dict()
#                odata["INFO"][ikey]["UNITS"] = units            # Units string
#                odata["INFO"][ikey]["DESC"] = description       # Description string
#                odata["INFO"][ikey]["SCSTR"] = scale            # String representation of scaling for writing into ex-file
#                odata["INFO"][ikey]["SCALE"] = float(scale)     # Numeric value for actual scaling operation
#                odata["INFO"][ikey]["FORM"] = dtype             # Data type for conversion to binary representation
#                odata["INFO"][ikey]["LABEL"] = ikey             # Name of data field, printed verbatim into ex-file
#                secnum = odata["SECTIONS"].index('Profiles') + 1 if 'Profiles' in odata["SECTIONS"] else odata["SECTIONS"].index('Traces') + 1
#                if ikey == "XVEC1":
#                    secnum = odata["SECTIONS"].index('PPF Base Vectors') + 1
#                elif ikey == "TVEC1" and "XVEC1" not in odata:
#                    secnum = odata["SECTIONS"].index('PPF Base Vectors') + 1
#                elif ikey not in ["TVEC1", "XVEC1"]:
#                    if ibase is None:
#                        ibase = "XVEC1" if "XVEC1" in odata else "TVEC1"
#                    odata["INFO"][ikey]["XBASE"] = ibase        # Data field name of reference vector as given in PPF Base Vectors section
#                    odata["INFO"][ikey]["UID"] = uname          # User ID (auto-detected)
#                    odata["INFO"][ikey]["DDA"] = itag           # DDA name in PPF system (replaced with metadata)
#                    odata["INFO"][ikey]["DTYPE"] = ''           # Data field name in PPF system (replaced with empty string)
#                    odata["INFO"][ikey]["SEQ"] = '0'            # Sequence number in PPF system (replaced with zero string)
#                odata["INFO"][ikey]["SECNUM"] = secnum          # Section number of data field for proper separation in ex-file
#            else:
#                print("   Requested field already present in data structure. Addition aborted.")
#        else:
#            print("   Input %s data is not consistent with base vectors present in data structure. Addition aborted." % (ikey))
#    else:
#        print("   Base data to be modified is not a dictionary. Use read_binary_file() or create_exfile_structure() function to create it.")
#
#    return odata


#def modify_entry(data, key, moddata, dtype=None, units=None, description=None, scale=None, tag=None):
#    ikey = None
#    idata = None
#    itag = 'Python Modification Tool'
#    if isinstance(key, str):
#        ikey = key
#    if isinstance(moddata, (list, tuple)):
#        idata = np.array(moddata)
#    elif isinstance(moddata, np.ndarray):
#        idata = moddata.copy()
#    if isinstance(tag, str):
#        itag = itag + ' - ' + tag
#
#    odata = None
#    if isinstance(data, dict) and ikey in data:
#        if data[ikey].size == idata.size:
#            idata = np.reshape(idata, data[ikey].shape)
#        writeData = False
#        dataShape = None
#        if "XBASE" in data["INFO"][ikey]:
#            if data["INFO"][ikey]["XBASE"] == "XVEC1":
#                dataShape = (data["TVEC1"].shape[0], data["XVEC1"].shape[1])
#            elif data["INFO"][ikey]["XBASE"] == "TVEC1":
#                dataShape = (1, data["TVEC1"].shape[0])
#            else:
#                print(f"   Unrecognised XBASE for {ikey}: {data['INFO'][ikey]['XBASE']}.")
#            if idata.shape == dataShape:
#                writeData = True
#            elif isinstance(dataShape, (list, tuple)):
#                print("   Input data is not the same shape as existing data structure: ", dataShape)
#        else:
#            if ikey in ["XVEC1", "TVEC1"]:
#                writeData = True
#            else:
#                print(f"   XBASE is missing for {ikey} in existing data structure.")
#        if writeData:
#            odata = copy.deepcopy(data)
#            odata[ikey] = idata.copy()
#            if ikey not in ["XVEC1", "TVEC1"]:
#                uname = getuser()
#                odata["INFO"][ikey]["UID"] = uname    # User ID (auto-detected)
#                odata["INFO"][ikey]["DDA"] = itag     # DDA name in PPF system (replaced with metadata)
#                odata["INFO"][ikey]["DTYPE"] = ''     # Data field name in PPF system (replaced with empty string)
#                odata["INFO"][ikey]["SEQ"] = '0'      # Sequence number in PPF system (replaced with zero string)
#            if isinstance(units, str):
#                odata["INFO"][ikey]["UNITS"] = units            # Units string
#            if isinstance(description, str):
#                odata["INFO"][ikey]["DESC"] = description       # Description string
#            if isinstance(scale, str):
#                odata["INFO"][ikey]["SCSTR"] = scale            # String representation of scaling for writing into ex-file
#                odata["INFO"][ikey]["SCALE"] = float(scale)     # Numeric value for actual scaling operation
#            if isinstance(dtype, str):
#                odata["INFO"][ikey]["FORM"] = dtype             # Data type for conversion to binary representation
#            odata["INFO"][ikey]["LABEL"] = ikey                 # Name of data field, printed verbatim into ex-file
#    elif isinstance(data, dict):
#        print("   Base data to be modified does not contain the entry %s. Modification aborted.")
#    else:
#        print("   Base data to be modified is not a dictionary. Use read_binary_file() function to create it.")
#
#    return odata


#def modify_exfile(inputdata=None, qlist=None, exfilename=None, outfile=None, globaltag=None):
#    pdata = None
#    qqlist = None
#    gtag = None
#    expath = None
#    opath = Path('./modded_exfile.txt')
#    if isinstance(inputdata, dict):
#        pdata = copy.deepcopy(inputdata)
#    if isinstance(qlist, (list, tuple)) and len(qlist) > 0:
#        qqlist = []
#        for item in qlist:
#            if isinstance(pdata, dict) and item in pdata:
#                qqlist.append(item)
#    if isinstance(exfilename, str):
#        exname = exfilename
#        if not exname.endswith('.ex'):
#            exname = exname + '.ex'
#        expath = Path(exname)
#    if isinstance(outfile, str):
#        opath = Path(outfile)
#    if not opath.parent.exists():
#        opath.parent.mkdir(parents=True)
#    if isinstance(globaltag, str):
#        gtag = globaltag
#
#    status = 1
#    if pdata is not None and qqlist is not None:
#        if expath is not None and expath.is_file():
#            exdata = read_binary_file(str(expath.absolute()))
#            fmodified = False
#            for qq in qqlist:
#                if qq in exdata:
#                    exdata = modify_entry(exdata, qq, pdata[qq].flatten(), tag=gtag)
#                    fmodified = True
#                else:
#                    print("Quantity %s not found in ex-file, %s. Quantity not changed." % (qq, exname))
#            if fmodified:
#                exfile = opath.parent / (str(opath.stem) + '.ex')
#                status = write_binary_exfile(exdata, str(exfile.absolute()))
#                if status != 0:
#                    print("Error occurred while writing binary file. Check inputs and try again.")
#            else:
#                print("No quantities changed. Binary writing aborted.")
#        else:
#            print("Ex-file %s not found. Binary writing aborted." % (str(expath.absolute())))
#        extpath = Path(exname+'t')
#        if extpath.is_file():
#            extfile = opath.parent / (str(opath.stem) + '.ext')
#            shutil.copy2(str(extpath.absolute()), str(extfile.absolute()))
#
#    return status


#def repackage_data(data, quantities):
#    odata = None
#    qlist = []
#    if isinstance(data, dict):
#        odata = copy.deepcopy(data)
#    if isinstance(quantities, str):
#        qlist = quantities.split(',')
#    elif isinstance(quantities, (list, tuple)):
#        qlist = list(quantities)
#
#    if odata is not None and qlist:
#        qlist.extend(["INFO","SECTIONS","CREATION_DATE","CREATION_TIME"])
#        dlist = []
#        for key in odata:
#            if key not in qlist:
#                dlist.append(key)
#        for dkey in dlist:
#            del odata[dkey]
#            if "INFO" in odata and dkey in odata["INFO"]:
#                del odata["INFO"][dkey]
#    elif odata is not None:
#        print("Invalid quantity list provided for repackaging JETTO data, returning input data!")
#
#    return odata


    @classmethod
    def generate_entry_info(cls, fieldname, target='jsp'):
        lookup_jsp = {
            'ra': (None, 'Minor radius, normalised', 'XVEC1', '1.0', 1.0),
            'xrho': (None, 'Normalised toroidal flux', 'XVEC1', '1.0', 1.0),
            'psi': (None, 'Normalised poloidal flux', 'XVEC1', '1.0', 1.0),
            'spsi': (None, 'Sqrt of normalised poloidal flux', 'XVEC1', '1.0', 1.0),
            'r': ('m', 'Minor radius', 'XVEC1', '0.01', 0.01),
            'rho': ('m', 'JETTO rho coordinate', 'XVEC1', '0.01', 0.01),
            'pr': ('Pa', 'Pressure (from equilibrium)', 'XVEC1', '1.0', 1.0),
            'q': (None, 'q (safety factor)', 'XVEC1', '1.0', 1.0),
            'ne': ('m-3', 'Electron Density', 'XVEC1', '1000000.0', 1000000.0),
            'te': ('eV', 'Electron Temperature', 'XVEC1', '1.0', 1.0),
            'ti': ('eV', 'Ion Temperature', 'XVEC1', '1.0', 1.0),
            'zeff': (None, 'Z-effective', 'XVEC1', '1.0', 1.0),
            'angf': ('s-1', 'Angular Frequency', 'XVEC1', '1.0', 1.0),
            'nimp': ('m-3', 'Impurity 1 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp2': ('m-3', 'Impurity 2 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp3': ('m-3', 'Impurity 3 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp4': ('m-3', 'Impurity 4 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp5': ('m-3', 'Impurity 5 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp6': ('m-3', 'Impurity 6 Density', 'XVEC1', '1000000.0', 1000000.0),
            'nimp7': ('m-3', 'Impurity 7 Density', 'XVEC1', '1000000.0', 1000000.0),
            'trqi': ('N m-2', 'Intrinsic Torque', 'XVEC1', '0.1', 0.1),
            'prad': ('W m-3', 'Radiation', 'XVEC1', '0.1', 0.1),
            'qnbe': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'qnbi': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'sb1': ('m-3 s-1', 'Particle Source 1', 'XVEC1', '1000000.0', 1000000.0),
            'sb2': ('m-3 s-1', 'Particle Source 2', 'XVEC1', '1000000.0', 1000000.0),
            'jznb': ('A m-2', 'NB Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0),
            'nb': ('m-3', 'Fast Ion Density', 'XVEC1', '1000000.0', 1000000.0),
            'wfnb': ('J m-3', 'Fast Ion Energy Density', 'XVEC1', '0.1', 0.1),
            'torq': ('N m-2', 'Torque', 'XVEC1', '0.1', 0.1),
            'qrfe': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'qrfi': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'rf': ('m-3', 'Fast Ion Density', 'XVEC1', '1000000.0', 1000000.0),
            'wfrf': ('J m-3', 'Fast Ion Energy Density', 'XVEC1', '0.1', 0.1),
            'qece': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'jzec': ('A m-2', 'ECRH Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0),
            'qebe': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'qebi': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'jzeb': ('A m-2', 'EBW Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0)
        }
        lookup_jst = {
            'cur': ('A', 'Plasma Current', 'TVEC1', '1.0', 1.0)
        }
        lookup = lookup_jsp
        if target == 'jst':
            lookup = lookup_jst
        entry = None
        if fieldname in lookup:
            entry = {
                'units': None,
                'desc': '',
                'scstr': '',
                'xbase': '',
                'uid': '',
                'dda': '',
                'dtype': '',
                'seq': '0',
                'scale': 1.0,
                'label': '',
                'form': 'float',
                'section_number': 5
            }
            unit, desc, xbase, scstr, sc = lookup[fieldname]
            entry['units'] = unit
            entry['desc'] = desc
            entry['scstr'] = scstr
            entry['xbase'] = xbase
            entry['scale'] = sc
            entry['label'] = fieldname
        return entry


    @classmethod
    def convert_jsp_to_exfile(cls, outname, runfolder='./', filename='jetto.jsp', outdir='./'):
        lookup = {
            'xa': 'ra',
            'xrho': 'xrho',
            'xpsi': 'psi',
            'xpsq': 'spsi',
            'r': 'r',
            'rho': 'rho',
            'pr': 'pr',
            'q': 'q',
            'ne': 'ne',
            'te': 'te',
            'ti': 'ti',
            'zeff': 'zeff',
            'angf': 'angf',
            'nim1': 'nimp',
            'nim2': 'nimp2',
            'nim3': 'nimp3',
            'nim4': 'nimp4',
            'nim5': 'nimp5',
            'nim6': 'nimp6',
            'nim7': 'nimp7',
            'trqi': 'trqi',
            'qrad': 'prad',
            'qnbe': 'qnbe',
            'qnbi': 'qnbi',
            'sbd1': 'sb1',
            'sbd2': 'sb2',
            'jznb': 'jznb',
            'dnbd': 'nb',
            'wnbd': 'wfnb',
            'torq': 'torq',
            'qrfe': 'qrfe',
            'qrfi': 'qrfi',
            'drfd': 'rf',
            'wrfd': 'wfrf',
            'qece': 'qece',
            'jzec': 'jzec',
            'qebe': 'qebe',
            'qebi': 'qebi',
            'jzeb': 'jzeb'
        }
        status = 1
        idir = Path(runfolder)
        if idir.is_dir() and (idir / filename).is_file():
            iname = idir / filename
            idata = read_binary_file(str(iname.resolve()))
            if idata is not None:
                odata = create_exfile_structure(idata['dda_name'], int(idata['shot']))
                odata['xvec1'] = copy.deepcopy(idata['xvec1'])
                odata['tvec1'] = copy.deepcopy(idata['time'])
                for var, exvar in lookup.items():
                    if var in idata and np.abs(np.sum(idata[var])) > 1.0e-10:
                        info = generate_entry_info(exvar)
                        info['dda'] = 'JSP'
                        info['dtype'] = var
                        odata['info'][exvar] = info
                        odata[exvar] = copy.deepcopy(idata[var])
                odir = Path(outdir)
                if not odir.is_dir():
                    odir.mkdir(parents=True)
                oname = odir / outname
                status = write_binary_exfile(odata, output_file=str(oname))
        return status

