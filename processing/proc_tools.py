#234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

import h5py
from datetime import datetime
import matplotlib.pyplot as plt


#   FUNCTION m_apply
# Take any function and handles the inputs of the function by looking into the hdf5 file
# The input must be in the form datasets/name/channel or process/proc_name/name/channel
# Also write the output of the function into the hdf5. The name of the path can be given into 
# outputs_names
#   INPUTS:
# filname : name of the hdf5 file where the datas are stored
# function : Custom function that you want to call
# inputs : List of the datas inputs required by the function
# outputs_names : list of the names of the channels for the writting of the resuls
# ** kwargs : All the non-data inputs to give to the function
#    OUTPUTS:
# null

def m_apply(filename, function, inputs=[], outputs_names=None, **kwargs):
    inputs_data = []

    if filename.split('.')[-1] != 'hdf5':
        if os.path.isfile(filename.split('.')[0] + '.hdf5'):
            filename = filename.split('.')[0] + '.hdf5'
        else:
            try:
                read_file.tohdf5(filename)
                filename = filename.split('.')[0] + '.hdf5' 
                print('The file does not have an hdf5 extension. It has been converted.')
            except:
                print('The given filename does not have an hdf5 extension, and it was not possible\
                        to convert it. Please use an hdf5 file with m_apply')
    
    with h5py.File(filename, 'r') as f:
        if type(inputs)==str:
            inputs_data.append(f[inputs][:])
        else:
            for i in inputs:
                inputs_data.append(f[i][:]) 

    result = function(*inputs_data, **kwargs)
    if type(result) == type(None):
        return None
    
    with h5py.File(filename, 'a') as f:
        num_proc = len(f['process'].keys()) + 1
        fproc = f.require_group('process/' + str(num_proc) + '-' + function.__name__ + '/')
        out_dim = np.shape(result)[0]
        for n in range(out_dim):
            if outputs_names == None:
                if np.shape(inputs_data)[0] == out_dim:
                    output_path = inputs_data[n].split('/', 2)
                else:
                    output_path = 'result' + str(n)
                    print('The number of outputs names does not correspond to the number of inputs,\
                            using the default output names')
            elif len(outputs_names) == out_dim:
                output_path = outputs_names[n]
            elif len(outputs_names) > out_dim:
                print('The number of outputs names is bigger than the number of outputs, using the\
                        first output names')
                output_path = outputs_names[n]
            else:
                print('The number of outputs names is smaller than the number of outputs, using the\
                        default output names')
                output_path = 'result' + str(n)
            
            fproc[output_path] = result[n]
            write_generic_attributes(fproc[output_path], 
                        'process/' + str(num_proc) + '-' + function.__name__ + '/' + output_path, 
                        inputs, 
                        output_path)
            

#   FUNCTION path_inputs
# Find paths to all input files
#   INPUTS:
# filename or f: either the open datafile, or the filename of an .hdf5 file that can be opened.
# data_folder (default: 'datasets'): folder searched for source data
# selection (default: None): determines the name of folders or files to be used.
# criteria: determines what category selection refers to. Can be either:
#     'process': an entire process folder
#     'sample': the next subfile after 'process' or 'dataset'; originally used for samples
#     'channel': the next subfile after 'sample'; originally the actual dataset, named after the 
#         source channel.
#   OUTPUTS:
# in_path_list: list of paths to datafiles
    
    
def path_inputs(filename_or_f, data_folder = 'datasets', selection = None, criteria=None):
    if type(filename_or_f) == str:
        filename = filename_or_f
        with h5py.File(filename, 'a') as f:
            in_path_list = path_inputs(f, data_folder, selection, criteria)
    elif type(filename_or_f) == h5py._hl.files.File:
        f = filename_or_f
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = criteria_selection(in_path_list, selection, criteria)
    else:
        print('Error: First argument should either be the filename of an hdf5 file, or an open\
                hdf5 file.')
        in_path_list = []
    return in_path_list


#   FUNCTION find_paths_of_all_subgroups
# Recursively determines list of paths for all datafiles in current_path, as well as datafiles in
# all  subfolders (and sub-subfolders and...) of current path
#   INPUTS:
# f: open hdf5 file
# current_path: current group searched
#   OUTPUTS:
# path_list: list of paths to datafiles

def find_paths_of_all_subgroups(f, current_path):
    path_list = []
    for sub_group in f[current_path]:
        if isinstance(f[current_path+'/'+sub_group], h5py.Group):
            path_list.extend(find_paths_of_all_subgroups(f, current_path+'/'+sub_group))
        elif isinstance(f[current_path+'/'+sub_group], h5py.Dataset):
            path_list.append(current_path+'/'+sub_group)
    return path_list


#   FUNCTION criteria_selection
# From a given list of paths, returns a shorter list that has been refined by particular criteria
#   INPUTS:
# path_list: a list of all possible paths
# selection: determines the name of folders or files to be used.
# criteria: determines what category selection refers to. Can be either:
#     'process': an entire process folder
#     'sample': the next subfile after 'process' or 'dataset'; originally used for samples
#     'channel': the next subfile after 'sample'; originally the actual dataset, named after the 
#         source channel.
#   OUTPUTS:
# valid_path_list: shorter list of paths

def criteria_selection(path_list, selection, criteria):
    valid_path_list = []
    if type(selection) != list:
        selection = [selection]
    if criteria == 'process' or criteria == 'Process' or criteria == 'p':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 1):
                    valid_path_list.append(path)
            else:
                print('Error: \'process\' criteria given, but path not linking to \'process\'\
                        folder. Path considered invalid.')
                
    elif criteria == 'sample' or criteria == 'Sample' or criteria == 's':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 2):
                    valid_path_list.append(path)
            elif split_path[0] == 'datasets':
                if is_valid_path(split_path, selection, 1):
                    valid_path_list.append(path)
            else:
                print('Error: Path not linking to \'process\' or \'datasets\' folder. Path\
                        considered invalid.')
    elif criteria == 'channel' or criteria == 'Channel' or criteria == 'c':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 3):
                    valid_path_list.append(path)
            elif split_path[0] == 'datasets':
                if is_valid_path(split_path, selection, 2):
                    valid_path_list.append(path)
            else:
                print('Error: Path not linking to \'process\' or \'datasets\' folder. Path\
                        considered invalid.')
    elif criteria is None:
        for path in path_list:
            split_path = path.split('/')
            if is_valid_path(split_path, selection, 0):
                valid_path_list.append(path)
    else:
        print('Error: Criteria should be either \'process\', \'sample\', or \'channel\'. No\
                selection was performed.')
        valid_path_list = path_list
    return valid_path_list


#   FUNCTION is_valid_path
# Determines if the path is one of the selected, valid paths.
#   INPUTS:
# split_path: the path, split in a list along each iteration of the character '/'
# selection: string that is required for the path to be valid
# check_index: index of the split path checked to contain the selection
#   OUTPUTS:
# valid: bool that states if the path is valid or not

def is_valid_path(split_path, selection, check_index):
    valid = False
    for valid_name in selection:
        if split_path[check_index] == valid_name:
            valid = True
    return valid


#   FUNCTION find_output_folder_location
# Creates a list of paths to the locations of the output folder. These paths lead to the process
# folder, contain a number corresponding to the operation number (starting from 1), and contain the
# process name. The 'sample' folder can be passed manually, or inherited from the source folder.
#   INPUTS:
# filename or f: either the open datafile, or the filename of a .hdf5 file that can be opened.
# process_name: the name of the process folder
# folder_names: The 'sample' folder name as a string. Alternatively, the list of source folders
#     can instead be passed. The folder name then directly copies from these source folders.
# overwrite if same (default: False): if set to True, if this function was the last process run, the
#     last run will be overwritten and replaced with this. To be used sparingly, and only if 
#     function parameters must be guessed and checked
#   OUTPUTS:
# out_folder_location_list: list of paths to output folders

def find_output_folder_location(filename_or_f, process_name, folder_names,
                                overwrite_if_same = False):
    out_folder_location_list = []
    if type(filename_or_f) == str:
        filename = filename_or_f
        with h5py.File(filename, 'a') as f:
            out_folder_location_list = find_output_folder_location(f, process_name, folder_names,
                                                                   overwrite_if_same)
    elif type(filename_or_f) == h5py._hl.files.File:
        f = filename_or_f
        operation_number = len(f['process'])+1
        if overwrite_if_same == True:
            if str(operation_number-1)+'-'+process_name in f['process'].keys():
                operation_number = operation_number-1
        if type(folder_names) != list:
            folder_names = [folder_names]
        for folder in folder_names:
            if '/' in folder:
                folder_root, output_filename = folder.rsplit('/', 1)
                if folder_root.split('/', 1)[0] == 'datasets':
                    folder_centre = folder_root.split('/', 1)[1]
                elif folder_root.split('/', 1)[0] == 'process':
                    folder_centre = folder_root.split('/', 2)[2]
                else:
                    print('Error: folder_names should not contain a slash unless a path to either\
                          datasets or process')
            else:
                folder_centre = folder
            out_folder_location = ('process/'+str(operation_number).zfill(2)+'-'+process_name+'/'
                                   +folder_centre+'/')
            out_folder_location_list.append(out_folder_location)
    return out_folder_location_list


#   FUNCTION write_output_f
# Writes the output to an open datafile
#   INPUTS:
# f: the open datafile
# data: the data to be written
# out_folder_location: location the dataset is written to
# in_paths: list of paths for the sources of the data
# output_name (default: None): the name of the datafile to be written. If left as None, the output
#     name is inherited from the name of the first entry of in_paths
#   OUTPUTS
# dataset: the directory to the dataset, such that f[dataset] would yield data

def write_output_f(f, data, out_folder_location, in_paths, output_name = None):
    f.require_group(out_folder_location)
    if type(in_paths) != list:
        in_paths = [in_paths]
    if output_name is None:
        if type(in_paths[0]) == str:
            output_name = in_paths[0].rsplit('/', 1)[1]
        else:
            output_name = in_paths[0][0].rsplit('/', 1)[1]
    try:
        # By default doesn't allow overwrite, so delete before writing
        del f[out_folder_location][output_name]
    except:
        pass
    f[out_folder_location].create_dataset(output_name, data = data)
    dataset = f[out_folder_location][output_name]
    write_generic_attributes(dataset, out_folder_location, in_paths, output_name)
    return dataset


#   FUNCTION write_generic_attributes
# Writes necessary and generic attributes to a dataset. This includes the dataset shape, its name,
# operation name and number, time of writing, and source file(s).
#   INPUTS:
# dataset: the dataset the attributes are written to
# out_folder_location: location of the dataset
# in_paths: the paths to the source files
# output_name: the name of the dataset
#   OUTPUTS
# null

def write_generic_attributes(dataset, out_folder_location, in_paths, output_name):
    if type(in_paths) != list:
        in_paths = [in_paths]
    operation_name = out_folder_location.split('/')[1]
    dataset.attrs['shape'] = dataset.shape
    dataset.attrs['name'] = output_name
    dataset.attrs['operation name'] = operation_name.split('-')[1]
    dataset.attrs['operation number'] = operation_name.split('-')[0]
    dataset.attrs['time'] = str(datetime.now())
    for i in range(len(in_paths)):
        dataset.attrs['source'+str(i)] = in_paths[i]

        
#   FUNCTION intermediate_plot
# Debugging tool that shows basic images in the kernel. The function checks if the keyword 
# 'condition' is in the list 'plotlist'. If so, the data is plotted and text printed. Alternatively,
# if force_plot is True, the plot is shown and text printed with no other checks. Otherwise, the
# function does nothing
#   INPUTS:
# data: data to be plotted
# condition (default: ''): a string checked to be in 'plotlist'
# plotlist (default: []): a list of strings that may contain 'condition'
# text (default: 'Intermediate Plot'): text printed prior to showing the plot.
# force_plot (default: False): if set to True, all other conditions are bypassed and the image is 
#     plotted.
#   OUTPUTS
# null
        
def intermediate_plot(data, condition='', plotlist=[], text='Intermediate Plot', force_plot=False):
    if force_plot:
        print(text)
        plt.imshow(data)
        plt.show()
        plt.close()
    elif type(plotlist) == list:
        if condition in plotlist:
            print(text)
            plt.imshow(data)
            plt.show()
            plt.close()

            
#   FUNCTION propagate_scale_attrs
# Attempts to write the scale attributes to a new dataset. This is done by directly copying from
# an old dataset. If this is not possible, then it attempts to generate this from the old dataset
# by calculating from the 'size' and 'shape' attributes.
#   INPUTS:
# new_data: new dataset to write to
# old_data: old dataset to read from
#   OUTPUTS
# null
            
def propagate_scale_attrs(new_data, old_data):
    if 'scale (m/px)' in old_data.attrs:
        new_data.attrs['scale (m/px)'] = old_data.attrs['scale (m/px)']
    else:
        if ('size' in old_data.attrs) and ('shape' in old_data.attrs):
            scan_size = old_data.attrs['size']
            shape = old_data.attrs['shape']
            new_data.attrs['scale (m/px)'] = scan_size[0]/shape[0]

            
            
            
            
            

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# DEPRACATED FILES
# DO NOT USE
# DELETE WHEN NO LONGER USED BY ANY FUNCTIONS

# FUNCTION initialise_process
## Initialises typical processes by finding paths to inputs, creating output folder, and finding their path
#INPUTS:
## filename: name of hdf5 file containing data
## process_name: name of the folder created
## create_groups: if True, creates folders, otherwise does not. Used for debugging.
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
#OUTPUTS:
## Returns a list for each file input. Each entry of this list is a three entry list.
## This sublist consists of the path to the file input, the path to the file output, and the name of the file output
## If process_name is left as None, the output folder creation is bypassed, and a simple list of in_paths are provided

def initialise_process(filename, process_name = None, data_folder = 'datasets', selection = None, selection_depth = 0, create_groups = True):
    with h5py.File(filename, 'a') as f:
        operation_number = str(len(f['process'])+1)
        # Find Inputs
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
        #print(f['process'])

        if process_name is not None:
            # Create Output Folder
            out_path_list = []
            output_filename_list = []
            for path in in_path_list:
                path_root, output_filename = path.rsplit('/', 1)
                output_filename_list.append(output_filename)
                out_folder = path_root[len(data_folder):]      # Skips initial nesting
                out_folder = 'process/'+operation_number+'-'+process_name+out_folder+'/'
                out_path_list.append(out_folder)
                if create_groups == True:
                    try:
                        curr_group = f.create_group(out_folder)
                    except:
                        pass

            in_out_list_of_lists = []
            for i in range(len(in_path_list)):
                in_out_list_of_lists.append([in_path_list[i], out_path_list[i], output_filename_list[i]])
        else:
            in_out_list_of_lists = in_path_list
    return in_out_list_of_lists


#FUNCTION select_certain_data
## Selects valid paths from a list of larger paths
#INPUTS:
## path_list: initial (longer) list of paths
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
#OUTPUTS:
## shortened list of paths

def select_certain_data(path_list, selection, selection_depth):
    # Given a list of paths, returns a shorter list given the selection_params. See initialise_process
    if type(selection) != list:
        selection = [selection]
    valid_path_list = []
    for i in range(len(path_list)):
        split_path = path_list[i].split('/')
        valid = False
        for valid_name in selection:
            if split_path[selection_depth] == valid_name:
                valid = True
        if valid == True:
            valid_path_list.append(path_list[i])
    return valid_path_list


#FUNCTION generic_write
# Writes output for most "generic" datafiles, overwriting by default
# Logs source folder, written name, shape, operation name and number, and current time in attributes
#INPUTS:
## f: open hdf5 file
## data: data to be written into hdf5
## path: path list of lists; the output of initialise_process
#OUTPUTS:
## null

def generic_write(f, path, data=None, *arg):
    if data!= None:
        try:
            # By default doesn't allow overwrite, so delete before writing
            del f[path[1]][path[2]]
        except:
            pass
        f[path[1]].create_dataset(path[2], data = data)
        f[path[1]][path[2]].attrs['shape'] = data.shape
        
    operation_name = path[1].split('/')[1]
    f[path[1]][path[2]].attrs['name'] = path[2]
    f[path[1]][path[2]].attrs['operation name'] = operation_name.split('-')[1]
    f[path[1]][path[2]].attrs['operation number'] = operation_name.split('-')[0]
    f[path[1]][path[2]].attrs['source'] = path[0]
    f[path[1]][path[2]].attrs['time'] = str(datetime.now())

    if data == None:
        f[path[1]][path[2]].attrs['shape'] = f[path[1]][path[2]][()].shape
    if len(arg)%2 != 0:
        print('Error: Odd amount of input arguments')
    else:
        for i in range(len(arg)//2):
            f[path[1]][path[2]].attrs[str(arg[2*i])] = arg[2*i+1]


#FUNCTION find_data_path_structure
## Helper/debugger function.
## Prints paths to valid datafiles given a filename, a data_folder to look in, and selection params
## Intended to help user to determine selection params for initialise_process
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
#OUTPUTS:
## null
    
def find_data_path_structure(filename, data_folder = 'datasets', selection = None, selection_depth = 0):
    with h5py.File(filename, 'a') as f:
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
    print(in_path_list)