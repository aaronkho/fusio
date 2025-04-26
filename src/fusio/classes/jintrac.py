import os.path
from getpass import getuser
import datetime
import re
import struct
import shutil
import copy
import collections
from itertools import tee

from pathlib import Path
import logging
import numpy as np
from .io import io

logger = logging.getLogger('fusio')

dtypes = {'int':'i', 'float':'f', 'double':'d', 'char':None}
byte_orderingtag = {'>':'Big-endian', '<':'Little-endian'}
byte_ordering = {'b': '>', 'l': '<'}

header_finder = re.compile(b'\*\n\*[\w, \d]+\n')
spec_finder = re.compile(b'#\w+;\d+;\d+;[\w, \d\-]+;\d+\n')
tracking_conversion = {
    'File Header': 0,
    'General Info': 0,
    'PPF Attributes': 1,
    'PPF Base Vectors': 2,
    'Profiles': 3,
    'Traces': -1,
    '': -1
}
tracking_tags = [' in file header', ' in provenance section', ' in base vector section', ' in time slice', '']


class jintrac_io(io):

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


    def _pairwise(self, iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


    def _decode_spec(self, spec):
        if not spec.startswith('#'):
            raise Exception(spec, ' is not a spec line')
        splitted = spec[1:-1].split(';')
        if len(splitted) != 5:
            raise Exception(spec, ' is not a spec line')
        jet_format = splitted[0]
        ignored = splitted[1]
        npoints = int(splitted[2])
        label = splitted[3]
        nlines = int(splitted[4])
        return label, jet_format, npoints, nlines


    def _decode_metadata(self, raw_metadata):
        itag = None
        tags = ['UNITS', 'DESC', 'SCSTR', 'XBASE', 'UID', 'DDA', 'DTYPE', 'SEQ']
        meta = list(map(lambda x: x.decode('latin-1').strip(), raw_metadata))
        metadata = dict(zip(tags, meta))
        if 'SCSTR' in metadata:
            metadata['SCALE'] = float(metadata['SCSTR'].strip())
        if 'UNITS' in metadata:
            if metadata['UNITS'] == '':
                metadata['UNITS'] = None
        return metadata


    def _decode_block(self, block, endianness='>', track_num=-1):
        """
        A block is the entire portion between spec lines, and typically contain the data for a single variable
        """
        loc = tracking_tags[track_num] if track_num < 3 else tracking_tags[3]
        if track_num >= 3:
            loc += ' number {:d}'.format(track_num - 3)
        corrupted = False
        # Read specification
        m = spec_finder.match(block)
        spec_string = block[:m.end()]
        block_string = block[m.end():]
        spec = spec_string.decode('latin-1')
        var_label, jet_format, npoints, nlines = decode_spec(spec)
        var_format = dtypes[jet_format]
        spec = spec.strip()
        # Read metadata
        try:
            splitted = re.split(b'(\n)', block_string)
            raw_metadata = splitted[:2*nlines:2]
            raw_data = b''.join(splitted[2*nlines:-2])
        # Protection against invalid specification line
        except Exception:
            corrupted = True
            raw_metadata = None
            raw_data = None
            print('Could not split {!s} block with spec {!s}{!s}'.format(var_label, spec, loc))
        try:
            var_metadata = decode_metadata(raw_metadata)
            var_metadata['LABEL'] = var_label
            var_metadata['FORM'] = jet_format
        # Protection against incorrect number of metadata lines
        except Exception:
            corrupted = True
            var_metadata = {}
            print('Could not decode {!s} metadata with spec {!s}{!s}'.format(var_label, spec, loc))
        # Read data
        if var_format is None:
            # No decoding needed, just read
            try:
                decoded_var = raw_data.decode('latin-1').strip()
            # Protection against plain read failure (highly unlikely)
            except Exception:
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
                    decoded_var = struct.unpack(endianness + npoints*var_format, raw_data)
                    if var_format is None:
                        decoded_var = decoded_var[0]
                    elif var_format in ['i', 'f', 'd']:
                        decoded_var = np.array(decoded_var)
                # Protection against incorrect specification of binary line length / format
                except Exception:
                    corrupted = True
                    decoded_var = np.full(npoints, np.nan)
                    print('Could not decode {!s} data with spec {!s}{!s}'.format(var_label, spec, loc))
        # Scale numerical data based on metadata value
        if 'SCALE' in var_metadata:
            try:
                decoded_var *= var_metadata['SCALE']
            # Protection against invalid value in scaling factor metadata
            except Exception:
                corrupted = True
                decoded_var = np.full(npoints, np.nan)
                print('Could not rescale {!s}{!s}'.format(var_label, loc))
        return var_label.upper(), decoded_var, var_metadata, corrupted


    def _decode_section(self, section, sec_num, endianness='>', metadata=None, data_start_num=-1):
        """
        A section is the entire portion between lines beginning with **
    
        Metadata option improves robustness of read routine by providing NaN vector shape in case of read failure
        """
        known_info = metadata if isinstance(metadata, dict) else {}
        m = header_finder.match(section)
        header_string = section[:m.end()]
        section_string = section[m.end():]
        section_header = header_string[3:-1].decode('latin-1')
        # Look for all data specs in the section (starts with #dtype;....)
        spec_starts = [match.start() for match in spec_finder.finditer(section_string)]
        spec_starts.append(len(section_string)) # To also read the last block
        blocks = [section_string[start:next_start] for start, next_start in pairwise(spec_starts)]
        if b''.join(blocks) != section_string:
            raise Exception('Something weird happened, did not split all blocks correctly')
        section_data = {}
        section_info = {}
        track_num = tracking_conversion[section_header] if section_header in tracking_conversion else -1
        if track_num == 3 and data_start_num > 0:
            track_num += (sec_num - data_start_num)
        for block_num, block in enumerate(blocks):
            var_label, block_data, var_info, corrupted = decode_block(block, endianness=endianness, track_num=track_num)
            record = True
            # Check if variable metadata is already known (i.e. corrupted entry is not in the first time slice)
            if corrupted and var_label in known_info:
                var_info = metadata[var_label]
            # Check if data already exists (i.e. duplicate entry in section)
            if var_label in section_info and var_label in section_data:
                record = False
                # Check if existing data is already good, if yes, keep it
                good_entry = ('XBASE' in section_info[var_label] and section_data[var_label] is not None and np.all(np.isfinite(section_data[var_label])))
                if not good_entry and not corrupted:
                    record = True
            if record:
                section_data[var_label] = block_data
                section_info[var_label] = var_info
                section_info[var_label]['SECNUM'] = sec_num + 1
        return section_header, section_data, section_info


    def _read_jintrac_binary_file(self, input_path, output_file=None):

        supported_file_extensions = ['.ex', '.ext', '.jsp', '.jst', '.jss', '.jse', '.jhp', '.jht', '.ssp', '.sst', '.jasp', '.jast', '.jsd']
        ipath = Path(input_path)
        if not ipath.is_file():
            raise Exception("File %s not found. Abort." % (str(ipath.absolute())))

        if (ipath.suffix not in supported_file_extensions)\
        and (not ipath.suffix.startswith('.ssp')) and\
        (not ipath.suffix.startswith('.sst')) and\
        (not ipath.suffix.startswith('.jsp')):
            raise Exception("Extention of file %s not in allowed list. Abort." % (str(ipath.absolute())))

        if isinstance(output_file,str):
            convert_binary_file(str(ipath.absolute()), output_file)

        # Store data from input file in dictionary
        data = collections.OrderedDict()
        data['INFO'] = collections.OrderedDict()

        data_types = {'int': 'i', 'float': 'f', 'double': 'd'}
        data_size = {'int': 4, 'float': 4, 'double': 8}

        with open(str(ipath.absolute()), 'rb') as inh:
            indata = inh.read()

        # First split the huge string into sections. A section starts with *\n*HEADER
        section_header_starts = [match.start() for match in header_finder.finditer(indata)]
        section_header_starts.append(len(indata)) # Also read the last section
        sections = [indata[start:next_start] for start, next_start in pairwise(section_header_starts)]
        if b''.join(sections) != indata:
            raise Exception('Something weird happened, did not split all sections correctly')

        # First section has info we need to decode the rest
        section_header, section_data, header_info = decode_section(sections[0], 0)
        data[section_header] = section_data
        endianness = byte_ordering[section_data['FILE FORMAT']]
        data_start_num = -1

        # Now decode all sections
        for sec_num, section in enumerate(sections[1:-1], start=1):
            metadata = data['INFO'] if data['INFO'] else None
            if 'Profiles' in data and data_start_num < 0:
               data_start_num = sec_num - 1
            section_header, section_data, section_info = decode_section(section, sec_num, endianness=endianness, metadata=metadata, data_start_num=data_start_num)
            if section_header == 'Profiles':
                if 'Profiles' not in data:
                    data['Profiles'] = []
                data[section_header].append(section_data)
            else:
                data[section_header] = section_data

            for k, v in section_info.items():
                if k not in data['INFO']:
                     data['INFO'][k] = v

        # Check length of 1D profile arrays
        if 'Profiles' in data:
            profiles = collections.OrderedDict()
            for slice_num, prof in enumerate(data['Profiles']):
                for key in prof.keys():
                    if key not in profiles:
                        profiles[key] = []
                    prof_data = copy.deepcopy(prof[key])
                    if 'PPF Base Vectors' in data and 'INFO' in data and key in data['INFO'] and 'XBASE' in data['INFO'][key]:
                        if not isinstance(prof_data, np.ndarray):
                            prof_data = np.full(data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size, np.nan)
                        if prof_data.size != data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size:
                            prof_data = np.full(data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size, np.nan)
                    if len(profiles[key]) != slice_num:
                        fill_data = np.full(prof_data.shape, np.nan)
                        profiles[key].append(fill_data)
                        print('Missing {!s} data in time slice number {:d}'.format(key, slice_num - 1))
                    profiles[key].append(prof_data)
            # Merge all 1D profiles of like quantities together into 2D arrays
            for key in list(profiles.keys()):
                profiles[key] = np.stack(profiles[key])
            data['Profiles'] = profiles

        # Check length of 1D time arrays
        if 'Traces' in data:
            traces = collections.OrderedDict()
            for key in data['Traces'].keys():
                if key not in traces:
                    traces[key] = []
                trac_data = copy.deepcopy(data['Traces'][key])
                if 'PPF Base Vectors' in data and 'INFO' in data and key in data['INFO'] and 'XBASE' in data['INFO'][key]:
                    if not isinstance(trac_data, np.ndarray):
                        trac_data = np.full(data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size, np.nan)
                    if trac_data.size != data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size:
                        trac_data = np.full(data['PPF Base Vectors'][data['INFO'][key]['XBASE']].size, np.nan)
                traces[key] = trac_data
            data['Traces'] = traces

        data = remove_duplicate_times(data, keep='last')
        data = standardize_data_representation(data, header_info, endianness)
        return data


    def _write_jintrac_binary_file(self, data, output_file='custom'):
        if not isinstance(data, (dict, collections.OrderedDict)):
            raise TypeError("Invalid input data structure, must be a dictionary. Abort.")
        elif "INFO" not in data or "SECTIONS" not in data:
            raise TypeError("Input data dictionary not correctly formatted. Abort.")

        dsizes = {'int':4, 'float':4, 'double':8, 'char':None}

        di = data["INFO"]
        ds = data["SECTIONS"]
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
        bord = byte_ordering[data["FILE FORMAT"]] if "FILE FORMAT" in data else byte_ordering['b']
        with open(str(opath.absolute()), 'wb') as ofile:
            NL = '\n'.encode()
            for ii in range(1, len(ds) + 1):

                if not re.match('^Profiles$', ds[ii-1], flags=re.IGNORECASE):
                    hstr1 = '*'
                    ofile.write(hstr1.encode() + NL)
                    hstr2 = '*' + ds[ii-1]
                    ofile.write(hstr2.encode() + NL)
                    for key in data:
                        if "SECNUM" in di[key] and di[key]["SECNUM"] == ii:
                            nstr = '1'
                            rnum = 0
                            if not re.match(r'^char$', di[key]["FORM"], flags=re.IGNORECASE):
                                nstr = "%d" % (data[key].shape[1])
                                rnum = -1
                            if "SEQ" in di[key]:
                                rnum = 8
                            elif "DTYPE" in di[key]:
                                rnum = 7
                            elif "DDA" in di[key]:
                                rnum = 6
                            elif "UID" in di[key]:
                                rnum = 5
                            elif "XBASE" in di[key]:
                                rnum = 4
                            elif "SCSTR" in di[key]:
                                rnum = 3
                            elif "DESC" in di[key]:
                                rnum = 2
                            elif "UNITS" in di[key]:
                                rnum = 1
                            rstr = '%d' % (rnum) if rnum >= 0 and not re.match(r'^Shot$', key, flags=re.IGNORECASE) else '0'
                            hdelim = ';'
                            fstr = '#' + hdelim.join((di[key]["FORM"], '1', nstr, di[key]["LABEL"], rstr))
                            ofile.write(fstr.encode() + NL)
                            if rnum >= 1:
                                estr = di[key]["UNITS"] if isinstance(di[key]["UNITS"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 2:
                                estr = di[key]["DESC"] if isinstance(di[key]["DESC"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 3:
                                estr = di[key]["SCSTR"] if isinstance(di[key]["SCSTR"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 4:
                                estr = di[key]["XBASE"] if isinstance(di[key]["XBASE"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 5:
                                estr = di[key]["UID"] if isinstance(di[key]["UID"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 6:
                                estr = di[key]["DDA"] if isinstance(di[key]["DDA"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 7:
                                estr = di[key]["DTYPE"] if isinstance(di[key]["DTYPE"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum >= 8:
                                estr = di[key]["SEQ"] if isinstance(di[key]["SEQ"], str) else ''
                                ofile.write(estr.encode() + NL)
                            if rnum == 0:
                                ofile.write(data[key].encode() + NL)
                            else:
                                scale = di[key]["SCALE"] if "SCALE" in di[key] else 1.0
                                dtype = dtypes[di[key]["FORM"]]
                                dsize = dsizes[di[key]["FORM"]]
                                for jj in range(data[key].shape[1]):
                                    dstr = b''
                                    if re.match(r'^[fd]$', dtype, flags=re.IGNORECASE):
                                        dstr = struct.pack(bord+dtype, float(data[key][0, jj] / scale))
                                    else:
                                        dstr = struct.pack(bord+dtype, int(data[key][0, jj] / scale))
                                    ofile.write(dstr)
                                ofile.write(NL)

                elif "TVEC1" in data and isinstance(data["TVEC1"], np.ndarray):
                    for tt in range(data["TVEC1"].shape[0]):
                        hstr1 = '*'
                        ofile.write(hstr1.encode() + NL)
                        hstr2 = '*' + ds[ii-1]
                        ofile.write(hstr2.encode() + NL)
                        for key in data:
                            if "SECNUM" in di[key] and di[key]["SECNUM"] == ii:
                                nstr = '1'
                                rnum = 0
                                if not re.match(r'^char$', di[key]["FORM"], flags=re.IGNORECASE):
                                    nstr = "%d" % (data[key].shape[1])
                                    rnum = -1
                                if "SEQ" in di[key]:
                                    rnum = 8
                                elif "DTYPE" in di[key]:
                                    rnum = 7
                                elif "DDA" in di[key]:
                                    rnum = 6
                                elif "UID" in di[key]:
                                    rnum = 5
                                elif "XBASE" in di[key]:
                                    rnum = 4
                                elif "SCSTR" in di[key]:
                                    rnum = 3
                                elif "DESC" in di[key]:
                                    rnum = 2
                                elif "UNITS" in di[key]:
                                    rnum = 1
                                rstr = '%d' % (rnum) if rnum >= 0 else '0'
                                hdelim = ';'
                                fstr = '#' + hdelim.join((di[key]["FORM"], '1', nstr, di[key]["LABEL"], rstr))
                                ofile.write(fstr.encode() + NL)
                                if rnum >= 1:
                                    estr = di[key]["UNITS"] if isinstance(di[key]["UNITS"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 2:
                                    estr = di[key]["DESC"] if isinstance(di[key]["DESC"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 3:
                                    estr = di[key]["SCSTR"] if isinstance(di[key]["SCSTR"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 4:
                                    estr = di[key]["XBASE"] if isinstance(di[key]["XBASE"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 5:
                                    estr = di[key]["UID"] if isinstance(di[key]["UID"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 6:
                                    estr = di[key]["DDA"] if isinstance(di[key]["DDA"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 7:
                                    estr = di[key]["DTYPE"] if isinstance(di[key]["DTYPE"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum >= 8:
                                    estr = di[key]["SEQ"] if isinstance(di[key]["SEQ"], str) else ''
                                    ofile.write(estr.encode() + NL)
                                if rnum == 0:
                                    ofile.write(data[key].encode() + NL)
                                else:
                                    scale = di[key]["SCALE"] if "SCALE" in di[key] else 1.0
                                    dtype = dtypes[di[key]["FORM"]]
                                    dsize = dsizes[di[key]["FORM"]]
                                    for jj in range(data[key].shape[1]):
                                        dstr = b''
                                        if re.match(r'^[fd]$', dtype, flags=re.IGNORECASE):
                                            if data[key].shape[0] > tt:
                                                dstr = struct.pack(bord+dtype, float(data[key][tt, jj] / scale))
                                            else:
                                                dstr = struct.pack(bord+dtype, float(data[key][-1, jj] / scale))
                                                print("   Less time slices than expected in field: %10s" % (di[key]["LABEL"]))
                                        else:
                                            if data[key].shape[0] > tt:
                                                dstr = struct.pack(bord+dtype, int(data[key][tt, jj] / scale))
                                            else:
                                                dstr = struct.pack(bord+dtype, int(data[key][-1, jj] / scale))
                                                print("   Less time slices than expected in field: %10s" % (di[key]["LABEL"]))
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
        if 'Traces' in data and 'PPF Base Vectors' in data and 'TVEC1' in data['PPF Base Vectors']:
            time_vector = data['PPF Base Vectors']['TVEC1'].flatten()
        if 'Profiles' in data and 'TIME' in data['Profiles']:
            time_vector = data['Profiles']['TIME'].flatten()
        if time_vector is not None:
            full_length = len(time_vector)
            unique_values, unique_indices = np.unique(time_vector, return_index=True)
            if len(unique_indices) != full_length:
                if keep == 'last':
                    index_diff = np.hstack((np.diff(unique_indices), 1)).astype(np.int64)
                    unique_indices = unique_indices + index_diff - 1
                if 'Traces' in data:
                    data['PPF Base Vectors']['TVEC1'] = np.take(data['PPF Base Vectors']['TVEC1'], unique_indices, axis=0)
                    for key in data['Traces']:
                        data['Traces'][key] = np.take(data['Traces'][key], unique_indices, axis=0)
                if 'Profiles' in data:
                    for key in data['Profiles']:
                        data['Profiles'][key] = np.take(data['Profiles'][key], unique_indices, axis=0)
        return data


    def _standardize_data_representation(self, data, header_info, endianness):
        data['SECTIONS'] = [key for key in data.keys() if key != 'INFO']
        for key in data['SECTIONS']:
            dateval = None
            timeval = None
            if key == 'File Header':
                dateval = data['File Header'].pop('DATE')
                timeval = data['File Header'].pop('TIME')
            data.update(data.pop(key))
            if dateval is not None:
                data['CREATION_DATE'] = dateval
            if timeval is not None:
                data['CREATION_TIME'] = timeval
        for timename in ['TIME', 'TVEC1']:
            if timename in data:
                data[timename] = data[timename][:, np.newaxis]
        dateinf = header_info.pop('DATE')
        timeinf = header_info.pop('TIME')
        data['INFO'].update(header_info)
        data['INFO']['FILE FORMAT']['FULLNAME'] = byte_orderingtag[endianness]
        data['INFO']['CREATION_DATE'] = dateinf
        data['INFO']['CREATION_TIME'] = timeinf
        data['INFO']['INFO'] = {'DESC': 'Additional information on data fields'}
        data['INFO']['SECTIONS'] = {'DESC': 'Labelled section and order within EX-FILE'}
        for key, val in data.items():
            if key not in ['INFO', 'SECTIONS']:
                if isinstance(val, np.ndarray) and val.ndim < 2:
                    data[key] = np.atleast_2d(val)
        return data


    @classmethod
    def create_empty_structure(cls, database, shot, version_tag=None, extfile=False):
        # Required user input - forced crash if improper
        if not isinstance(database,str):
            raise ValueError('Database field for ex-file generation must be a string')
        if not isinstance(shot,(int,float,np.int8,np.int16,np.int32,np.float16,np.float32,np.float64)):
            raise ValueError('Shot number field for ex-file generation must be numeric')
        datablock_tag = 'Traces' if extfile else 'Profiles'

        # Initialize structure
        data = {}
        data['INFO'] = {}
        data['SECTIONS'] = ['File Header', 'General Info', 'PPF Attributes', 'PPF Base Vectors', datablock_tag]

        # Add standard descriptions of required metadata - covers until end of 'PPF Attributes' section
        data['INFO']['INFO'] = {'DESC': 'Additional information on data fields'}
        data['INFO']['SECTIONS'] = {'DESC': 'Labelled section and order within EX-FILE'}
        data['INFO']['FILE FORMAT'] = {'FORM': 'char', 'FULLNAME': 'Big-endian', 'LABEL': 'File Format', 'SECNUM': 1}
        data['INFO']['FILE DESCRIPTION'] = {'FORM': 'char', 'LABEL': 'File Description', 'SECNUM': 1}
        data['INFO']['VERSION'] = {'FORM': 'char', 'LABEL': 'Version', 'SECNUM': 1}
        data['INFO']['CREATION_DATE'] = {'FORM': 'char', 'LABEL': 'Date', 'SECNUM': 1}
        data['INFO']['CREATION_TIME'] = {'FORM': 'char', 'LABEL': 'Time', 'SECNUM': 1}
        data['INFO']['DATABASE NAME'] = {'FORM': 'char', 'LABEL': 'Database Name', 'SECNUM': 2}
        data['INFO']['USER EX-FILE'] = {'FORM': 'char', 'LABEL': 'User EX-file', 'SECNUM': 2}
        data['INFO']['USER PRE-MODEX EX-FILE'] = {'FORM': 'char', 'LABEL': 'User Pre-Modex EX-file', 'SECNUM': 2}
        data['INFO']['SHOT'] = {'FORM': 'int', 'LABEL': 'Shot', 'SECNUM': 3}
        data['INFO']['DDA NAME'] = {'FORM': 'char', 'LABEL': 'DDA Name', 'SECNUM': 3}
        if not extfile:
            data['INFO']['XVEC1'] = {'UNITS': None, 'DESC': 'RHO normalised', 'SCSTR': '1.0', 'SCALE': 1.0, 'LABEL': 'XVEC1', 'FORM': 'float', 'SECNUM': 4}
            data['INFO']['TVEC1'] = {'UNITS': 'secs', 'DESC': 'TIME', 'SCSTR': '1.0', 'SCALE': 1.0, 'LABEL': 'TVEC1', 'FORM': 'float', 'SECNUM': 5}
        else:
            data['INFO']['TVEC1'] = {'UNITS': 'secs', 'DESC': 'TIME', 'SCSTR': '1.0', 'SCALE': 1.0, 'LABEL': 'TVEC1', 'FORM': 'float', 'SECNUM': 4}

        # Add required metadata - covers until end of 'PPF Attributes' section
        data['FILE FORMAT'] = 'b'
        data['FILE DESCRIPTION'] = 'EX-FILE'
        data['VERSION'] = version_tag if isinstance(version_tag,str) else '1.0 : JETTO Python tools'
        data['CREATION_DATE'] = datetime.datetime.now().strftime("%d/%m/%Y")
        data['CREATION_TIME'] = datetime.datetime.now().strftime("%H:%M:%S")
        data['DATABASE NAME'] = database
        data['USER EX-FILE'] = ''
        data['USER PRE-MODEX EX-FILE'] = ''
        data['SHOT'] = np.atleast_2d([int(shot)])
        data['DDA NAME'] = 'EX'
        if not extfile:
            data['XVEC1'] = np.atleast_2d([])
            data['TVEC1'] = np.atleast_3d([])
        else:
            data['TVEC1'] = np.atleast_2d([])

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
#    if isinstance(data, (dict, collections.OrderedDict)) and ikey is not None:
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
#    if isinstance(data, (dict, collections.OrderedDict)) and ikey in data:
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
#    elif isinstance(data, (dict, collections.OrderedDict)):
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
#    if isinstance(inputdata, (dict, collections.OrderedDict)):
#        pdata = copy.deepcopy(inputdata)
#    if isinstance(qlist, (list, tuple)) and len(qlist) > 0:
#        qqlist = []
#        for item in qlist:
#            if isinstance(pdata, (dict, collections.OrderedDict)) and item in pdata:
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
#    if isinstance(data, (dict, collections.OrderedDict)):
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
            'RA': (None, 'Minor radius, normalised', 'XVEC1', '1.0', 1.0),
            'XRHO': (None, 'Normalised toroidal flux', 'XVEC1', '1.0', 1.0),
            'PSI': (None, 'Normalised poloidal flux', 'XVEC1', '1.0', 1.0),
            'SPSI': (None, 'Sqrt of normalised poloidal flux', 'XVEC1', '1.0', 1.0),
            'R': ('m', 'Minor radius', 'XVEC1', '0.01', 0.01),
            'RHO': ('m', 'JETTO rho coordinate', 'XVEC1', '0.01', 0.01),
            'PR': ('Pa', 'Pressure (from equilibrium)', 'XVEC1', '1.0', 1.0),
            'Q': (None, 'q (safety factor)', 'XVEC1', '1.0', 1.0),
            'NE': ('m-3', 'Electron Density', 'XVEC1', '1000000.0', 1000000.0),
            'TE': ('eV', 'Electron Temperature', 'XVEC1', '1.0', 1.0),
            'TI': ('eV', 'Ion Temperature', 'XVEC1', '1.0', 1.0),
            'ZEFF': (None, 'Z-effective', 'XVEC1', '1.0', 1.0),
            'ANGF': ('s-1', 'Angular Frequency', 'XVEC1', '1.0', 1.0),
            'NIMP': ('m-3', 'Impurity 1 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP2': ('m-3', 'Impurity 2 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP3': ('m-3', 'Impurity 3 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP4': ('m-3', 'Impurity 4 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP5': ('m-3', 'Impurity 5 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP6': ('m-3', 'Impurity 6 Density', 'XVEC1', '1000000.0', 1000000.0),
            'NIMP7': ('m-3', 'Impurity 7 Density', 'XVEC1', '1000000.0', 1000000.0),
            'TRQI': ('N m-2', 'Intrinsic Torque', 'XVEC1', '0.1', 0.1),
            'PRAD': ('W m-3', 'Radiation', 'XVEC1', '0.1', 0.1),
            'QNBE': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'QNBI': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'SB1': ('m-3 s-1', 'Particle Source 1', 'XVEC1', '1000000.0', 1000000.0),
            'SB2': ('m-3 s-1', 'Particle Source 2', 'XVEC1', '1000000.0', 1000000.0),
            'JZNB': ('A m-2', 'NB Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0),
            'NB': ('m-3', 'Fast Ion Density', 'XVEC1', '1000000.0', 1000000.0),
            'WFNB': ('J m-3', 'Fast Ion Energy Density', 'XVEC1', '0.1', 0.1),
            'TORQ': ('N m-2', 'Torque', 'XVEC1', '0.1', 0.1),
            'QRFE': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'QRFI': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'RF': ('m-3', 'Fast Ion Density', 'XVEC1', '1000000.0', 1000000.0),
            'WFRF': ('J m-3', 'Fast Ion Energy Density', 'XVEC1', '0.1', 0.1),
            'QECE': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'JZEC': ('A m-2', 'ECRH Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0),
            'QEBE': ('W m-3', 'Power Density Electrons', 'XVEC1', '0.1', 0.1),
            'QEBI': ('W m-3', 'Power Density Ions', 'XVEC1', '0.1', 0.1),
            'JZEB': ('A m-2', 'EBW Driven Curr.Dens', 'XVEC1', '1.0E7', 10000000.0)
        }
        lookup_jst = {
            'CUR': ('A', 'Plasma Current', 'TVEC1', '1.0', 1.0)
        }
        lookup = lookup_jsp
        if target == 'jst':
            lookup = lookup_jst
        entry = None
        if fieldname in lookup:
            entry = {'UNITS': None, 'DESC': '', 'SCSTR': '', 'XBASE': '', 'UID': '', 'DDA': '', 'DTYPE': '', 'SEQ': '0', 'SCALE': 1.0, 'LABEL': '', 'FORM': 'float', 'SECNUM': 5}
            unit, desc, xbase, scstr, sc = lookup[fieldname]
            entry['UNITS'] = unit
            entry['DESC'] = desc
            entry['SCSTR'] = scstr
            entry['XBASE'] = xbase
            entry['SCALE'] = sc
            entry['LABEL'] = fieldname
        return entry


    @classmethod
    def convert_jsp_to_exfile(cls, outname, runfolder='./', filename='jetto.jsp', outdir='./'):
        lookup = {
            'XA': 'RA',
            'XRHO': 'XRHO',
            'XPSI': 'PSI',
            'XPSQ': 'SPSI',
            'R': 'R',
            'RHO': 'RHO',
            'PR': 'PR',
            'Q': 'Q',
            'NE': 'NE',
            'TE': 'TE',
            'TI': 'TI',
            'ZEFF': 'ZEFF',
            'ANGF': 'ANGF',
            'NIM1': 'NIMP',
            'NIM2': 'NIMP2',
            'NIM3': 'NIMP3',
            'NIM4': 'NIMP4',
            'NIM5': 'NIMP5',
            'NIM6': 'NIMP6',
            'NIM7': 'NIMP7',
            'TRQI': 'TRQI',
            'QRAD': 'PRAD',
            'QNBE': 'QNBE',
            'QNBI': 'QNBI',
            'SBD1': 'SB1',
            'SBD2': 'SB2',
            'JZNB': 'JZNB',
            'DNBD': 'NB',
            'WNBD': 'WFNB',
            'TORQ': 'TORQ',
            'QRFE': 'QRFE',
            'QRFI': 'QRFI',
            'DRFD': 'RF',
            'WRFD': 'WFRF',
            'QECE': 'QECE',
            'JZEC': 'JZEC',
            'QEBE': 'QEBE',
            'QEBI': 'QEBI',
            'JZEB': 'JZEB'
        }
        status = 1
        idir = Path(runfolder)
        if idir.is_dir() and (idir / filename).is_file():
            iname = idir / filename
            idata = read_binary_file(str(iname.resolve()))
            if idata is not None:
                odata = create_exfile_structure(idata['DDA NAME'], int(idata['SHOT']))
                odata['XVEC1'] = copy.deepcopy(idata['XVEC1'])
                odata['TVEC1'] = copy.deepcopy(idata['TIME'])
                for var, exvar in lookup.items():
                    if var in idata and np.abs(np.sum(idata[var])) > 1.0e-10:
                        info = generate_entry_info(exvar)
                        info['DDA'] = 'JSP'
                        info['DTYPE'] = var
                        odata['INFO'][exvar] = info
                        odata[exvar] = copy.deepcopy(idata[var])
                odir = Path(outdir)
                if not odir.is_dir():
                    odir.mkdir(parents=True)
                oname = odir / outname
                status = write_binary_exfile(odata, output_file=str(oname))
        return status

