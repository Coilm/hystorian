try:
    xrdml_bool = True
    from . import xrdml_files
except ImportError:
    xrdml_bool = False

from . import ibw_files
from . import ardf_files
from . import sxm_files
from . import gsf_files
from . import csv_files
from . import nanoscope_files
from . import img_files
from . import shg_files
from ..processing import core
import h5py
import os
import re
import numpy as np

def tohdf5(filename,filepath=None, params=None):
    '''
    Used to convert a datafile to an hdf5 file.

    Parameters
    ----------
    filename: str
        Name of the file to be converted. MUST contain the extension.
    params: str, optional
        Name of the file containing metadatas

    Returns
    -------
        filetype: str
            string containing the converted file extension
    '''
    filetype = 'unknown'
    if type(filename) == list:
        merge_hdf5(filename, 'merged_file', erase_file='partial')
    else:
        if filename.split('.')[-1] == 'ibw':
            ibw_files.ibw2hdf5(filename, filepath)
            filetype = 'ibw'
        elif filename.split('.')[-1] == 'xrdml':
            if xrdml_bool:
                xrdml_files.xrdml2hdf5(filename, filepath)
            else:
                print('xrdml_files was not imported, probably due to the missing xrd_tools package. Please install it.')
                
            filetype = 'xrdml'
        elif filename.split('.')[-1] == 'ardf' or filename.split('.')[-1] == 'ARDF':
            ardf_files.ardf2hdf5(filename, filepath)
            filetype = 'ardf'
        elif filename.split('.')[-1] == 'sxm':
            sxm_files.sxm2hdf5(filename, filepath)
            filetype = 'sxm'
        elif filename.split('.')[-1] == 'gsf':
            gsf_files.gsf2hdf5(filename, filepath)
            filetype = 'gsf'
        elif filename.split('.')[-1] == 'csv':
            csv_files.csv2hdf5(filename, filepath)
            filetype = 'csv'
        elif filename.split('.')[-1] == 'jpg' or filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpeg' \
                or filename.split('.')[-1] == 'bmp':
            img_files.image2hdf5(filename, filepath)
            filetype = 'jpg'
        elif re.match('\d{3}', filename.split('.')[-1]) is not None:
            try:
                nanoscope_files.nanoscope2hdf5(filename, filepath)
            except:
                print('Your file is ending with three digits, but is not a Nanoscope file, please change the '
                      'extension to the correct one') 
            filetype = 'nanoscope'
        elif filename.split('.')[-1] == 'dat':
            try:
                shg_files.dat2hdf5(filename, filepath, params)
            except:
                print('This dat file has a shape which is not supported, for the moment dat file are used for SHG datas')
            filetype= 'dat'
        else:
            print('file type not yet supported')
    return filetype

def merge_hdf5(filelist, combined_name,shortend = True, erase_file='partial', merge_process=False):
    '''
    Take a list of file to convert into a single hdf5 file

    Parameters
    ----------
    filelist: list of strings
        list of the file to be converted. MUST contain the extensions. There can be multiple type of file in the list
    combined_name: str
        Name of the hdf5 file
    erase_file : str, optional
        merge_hdf5 create temporary hdf5 file for each element in filelist and then merge them.
        If set to 'partial', it erases the temp files.
    merge_process : bool, optional
        If set to True, merge_hdf5 will attempt to merge the process folders of hdf5 files together

    Returns
    -------
        None
    '''
    i = 0
    temporary = False
    if shortend:
        file_path = shorten_path(filelist)
    else:
        file_path = None

    with h5py.File(combined_name + '.hdf5', 'w') as f:
        metadatagrp = f.create_group('metadata')
        datagrp = f.create_group('datasets')
        procgrp = f.create_group('process')
        # print(f.keys())

    for i, filename in enumerate(filelist):

        if filename.split('.')[0] == combined_name:
            print('Warning: \'' + filename + '\' already exists. File has been ignored and will be overwritten.')
            continue
        else:
            if filename.split('.')[-1] != 'hdf5':
                if not (filename.split('.')[0] + '.hdf5' in filelist):
                    try:
                        if shortend:
                            filetype = tohdf5(filename, filepath=file_path[i])
                        else:
                            filetype = tohdf5(filename)
                        if filetype == 'nanoscope':
                            filename = filename.replace('.', '_') + ".hdf5"
                        else:
                            filename = filename.split('.')[0] + '.hdf5'
                        temporary = True
                    except:
                        print('Warning: \'' + filename + '\' as impossible to convert. File has been ignored.')
                        continue
                else:
                    print(
                        'Warning: \'' + filename + '\' is a non-hdf5 file, but the list containt a hdf5 file with the same name. File has been ignored.')
                    continue
        with h5py.File(combined_name + '.hdf5', 'a') as f:
            metadatagrp = f['metadata']
            datagrp = f['datasets']
            procgrp = f['process']
            with h5py.File(filename, 'r') as source_f:
                #if file_path is not None:
                #    p = file_path[i].split('.')[0]
                #else:
                #    p = filename.split('.hdf5')[0]
                #f.require_group('metadata/' + '/'.join(p.split('/')[:-1]))
                #source_f.copy('metadata/' + p, f['metadata/' + '/'.join(p.split('/')[:-1])])
                #f.require_group('datasets/' + '/'.join(p.split('/')[:-1]))
                #source_f.copy('datasets/' + p, f['datasets/' + '/'.join(p.split('/')[:-1])])
                
                for source_metadata in source_f['metadata']:
                    source_f.copy('metadata/' + source_metadata, metadatagrp)
                for source_data in source_f['datasets']:
                    source_f.copy('datasets/' + source_data, datagrp)
                
                if merge_process:
                    f.require_group('process/' + '/'.join(p.split('/')[:-1]))
                    source_f.copy('process/' + p, f['process/' + '/'.join(p.split('/')[:-1])])
                else:
                    for source_proc in source_f['process']:
                        dest_proc = source_proc.split('-', 1)[1]
                        num_proc = len(procgrp.keys())
                        num_proc = num_proc + 1
                        dest_proc = str(num_proc).zfill(3)+'-'+dest_proc
                        core.create_group_path(procgrp, dest_proc)
                        for source_proc_proc in source_f['process'][source_proc]:
                            source_f.copy('process/'+source_proc+'/'+source_proc_proc, procgrp[dest_proc])
                i = i + 1

        print('\'' + filename + '\' successfully merged')
        if erase_file == 'partial' and temporary:
            temporary = False
            if filename.split('.')[-1] == 'hdf5':
                os.remove(filename)

    print(str(i) + '/' + str(len(filelist)) + ' files merged into \'' + combined_name + '.hdf5\'')

    
def create_hdf5(filename, image):
    '''
    Create an hdf5 file from an array-like.

    Parameters
    ----------
    filename: str
        Name of the resulting file
    image: array-like
        data to be converted into hdf5
    Returns
    -------
        None
    '''
    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        metadatagrp = f.create_group("metadata")
        datagrp = f.create_group("datasets/" + filename.split('.')[0])
        f.create_group("process")
        label = filename.split('.')[0]
        datagrp.create_dataset(label, data=image)
        datagrp[label].attrs['name'] = filename
        datagrp[label].attrs['shape'] = image.shape
        datagrp[label].attrs['path'] = ("datasets/" + filename.split('.')[0])
    print('file successfully converted')


def shorten_path(filelist):
    if type(filelist) == str:
        filelist = [filelist]
    filelist = [i.replace('\\', '/') for i in filelist]
    pad = max([len(i.split('/')) for i in filelist])
    splitted = np.array([i.split('/') + ['NAN'] * (pad - len(i.split('/'))) for i in filelist])
    for i in range(np.shape(splitted)[1]):
        if len(set(splitted[:, i])) != 1:
            breakpoint = i
            break
        else:
            breakpoint = -1
    result = []
    for i in range(np.shape(splitted)[0]):
        unpadded = np.delete(splitted[i, breakpoint:], np.where(splitted[i, breakpoint:] == 'NAN'))
        result.append('/'.join(unpadded))

    return result