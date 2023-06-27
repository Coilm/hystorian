import inspect
import re
import types
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, KeysView, Optional, overload

import h5py
import numpy as np
import numpy.typing as npt

from . import HyExtractor
from .utils import HyConvertedData

h5pyType = KeysView | h5py.Group | h5py.Dataset | h5py.Datatype


class HyPath:
    def __init__(self, path: str):
        self._path = path

    @property
    def path(self):
        return self._path

    def __str__(self):
        return self._path

    def __repr__(self):
        return f"HyPath({self._path!r})"


class HyApply:
    def __init__(self, file, func: Callable, args: tuple[Any], kwargs: dict[str, Any] = {}):
        self.file = file
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def apply(self):
        result = self.func(
            *self._deeplist_translate(list(deepcopy(self.args)), self._read),
            **self._deepdict_translate(self.kwargs, self._read),
        )

        result_args = self._deeplist_translate(list(self.args), self._path)
        result_kwargs = self._deepdict_translate(self.kwargs, self._path)

        return result, result_args, result_kwargs

    def _deeplist_translate(self, iter_: list[Any], f: Callable):
        for i, item in enumerate(iter_):
            if isinstance(item, list):
                self._deeplist_translate(item, f)
            else:
                if isinstance(item, HyPath):
                    iter_[i] = f(item)
                else:
                    pass
        return tuple(iter_)

    def _read(self, arg):
        return self.file.read(arg)

    def _path(self, arg):
        if isinstance(arg, HyPath):
            return arg.path
        else:
            return arg

    def _deepdict_translate(self, args, f):
        translated = {}
        for key, val in args.items():
            if isinstance(val, list):
                translated[key] = self._deeplist_translate(val, self._read)
            else:
                if isinstance(val, HyPath):
                    translated[key] = f(val)
                else:
                    translated[key] = val
        return translated


class HyFile:
    """HyFile is a class that wraps around the h5py.File class and is used to create and manipulate datafile from proprieteray files

    Attributes
    ----------
    path : Path
        Path to the hdf5 file to be manipulated.
    file : h5py.File
        Handle of the file: the class call h5py.File(path). If the file does not exist, it is generated. (See __init__ docstring for more details).
    attrs : Attributes
        Attributes is an internal class, which allow the manipulation of hdf5 attributes through HyFile.
    """

    class Attributes:
        """Internal class of HyFile which allows for the manipulation of attributes inside an hdf5 file in the same way than h5py does.

        Examples
        --------
        - This will navigate to the Dataset located at 'path/to/data' and read the attribute with the key 'important_attribute'.

        >>> f['path/to/data'].attrs('important_attribute')

        - This will navigate to the Dataset located at 'path/to/data' and write (or overwrite if it already exists) the attribute with key new_attribute and set it to 0.

        >>> f['path/to/data'].attrs('new_attribute') = 0

        """

        def __init__(self, file: h5py.File):
            self.file = file

        def __getitem__(self, path: Optional[str] = None) -> dict:
            if path is None:
                f = self.file
            else:
                f = self.file[path]

            return {key: f.attrs[key] for key in f.attrs.keys()}

        def __setitem__(self, path: str, attribute: tuple[str, Any]) -> None:
            key, val = attribute
            self.file[path].attrs[key] = val

    def __init__(self, path: Path | str, mode: str = "r"):
        """Open the file given by path. If the file does not exist, a new hdf5 file is created with a root structure containing three hdf5 groups: 'datasets', 'metadata' and 'process'.

        Parameters
        ----------
        path : Path | str
            Path to the file to be manipulated.
        mode : str, optional
            Mode in which the file should be opened, Valid modes are:

            - 'r': Readonly, file must exist. (default)
            - 'r+': Read/write, file must exist.
            - 'w': Create file, truncate if exists.
            - 'w-' or 'x': Create file, fail if exists.
            - 'a' : Read/write if exists, create otherwise.

            by default 'r'.

        Raises
        ------
        TypeError
            Raise an error if the mode provided is not correct.
        """

        if mode not in ["r", "r+", "w", "w-", "a"]:
            raise TypeError(
                "{mode} is not a valid file permission.\n Valid permissions are: 'r', 'r+', 'w', 'w-' or 'a'"
            )
        self.path = Path(path)

        if self.path.is_file():
            self.file = h5py.File(self.path, mode)
            root_struct = set(self.file.keys())
            if root_struct != {"datasets", "metadata", "process"}:
                warnings.warn(
                    f"Root structure of the hdf5 file is not composed of 'datasets', 'metadata' and 'process'. \n It may not have been created with Hystorian. \n Current root structure is {root_struct}"
                )

        else:
            self.file = h5py.File(self.path, "a")
            for group in ["datasets", "metadata", "process"]:
                self._require_group(group)

            if mode != "a":
                self.file.close()
                self.file = h5py.File(self.path, mode)

        self.attrs = self.Attributes(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback) -> bool:
        if self.file:
            self.file.close()

        if value is not None:
            warnings.warn(f"File closed with the following error: {exc_type} - {value} \n {traceback}")
            return False

        return True

    def __getitem__(self, path: str | HyPath = "") -> h5pyType:
        if isinstance(path, HyPath):
            path = path.path
        if path == "":
            return self.file.keys()
        else:
            return self.file[path]

    def __setitem__(self, data: tuple[str, Any], overwrite=True) -> None:
        self._create_dataset(data, overwrite)

    def __delitem__(self, path: str) -> None:
        if path not in self.file:
            raise KeyError(f"Path {path} does not exist in the file.")
        del self.file[path]

    def __contains__(self, path: str) -> bool:
        return path in self.file

    @property
    def last_process(self):
        """returns a string which is the path to the last process in the hdf5 file."""
        processes = list(self.file["process"].keys())
        if len(processes) > 0:
            return f"process/{processes[-1]}"
        else:
            return

    @overload
    def read(self) -> list[str]:
        pass

    @overload
    def read(self, path: str | HyPath) -> list[str]:
        pass

    @overload
    def read(self, path: str | HyPath) -> h5py.Datatype:
        pass

    @overload
    def read(self, path: str | HyPath) -> npt.ArrayLike:
        pass

    def read(self, path: Optional[str | HyPath] = None) -> list[str] | h5py.Datatype | npt.ArrayLike:
        """Wrapper around the __getitem__ of h5py. Directly returns the keys of the sub-groups if the path lead to an h5py.Group, otherwise directly load the dataset.
        This allows to get a list of keys to the folders without calling .keys(), and to the data without [()] therefore the way to call the keys or the data are the same.
        And therefore the user does not need to change the call between .keys() and [()] to navigate the hierarchical structure.

        Parameters
        ----------
        path : Optional[str], optional
            Path to the Group or Dataset you want to read. If the value is None, read the root of the folder (should be [datasets, metadata, process] if created with Hystorian), by default None

        Returns
        -------
        list[str] | h5py.Datatype | npt.ArrayLike
            If the path lead to Groups, will return a list of the subgroups, if it lead to a Dataset containing data, it will directly return the data, and if it is an empty Dataset, will return its Datatype.
        """
        if path is None:
            return list(self.file.keys())

        if isinstance(path, HyPath):
            path = path.path

        current = self.file[path]
        if isinstance(current, h5py.Group):
            return list(current.keys())
        if isinstance(current, h5py.Datatype):
            return current
        else:
            return current[()]

    def apply(
        self,
        function: Callable,
        /,
        *args: Any,
        output_names: Optional[list[str] | str] = None,
        increment_proc: bool = True,
        **kwargs: Any,
    ):
        """apply allows to call a function and store all the inputs and outputs in the hdf5 file with the raw data.

        Parameters
        ----------
        function : Callable
            function used to transform the data. Result of the function will be stored in process/XXX-<function-name>, where XXX is an incrementing number for each already existing process.
        *args : Any
            All the positional arguments for the above function. If an HyPath is present in args, then self.file.read() will be called for this argument.
        output_names : Optional[list[str]  |  str], optional
            Name to be given to the result of the function. The number of names should be the same as the number of outputs of the function passed,
            othewise a ValueError will be raised, if None is passed, the name of the function will be used, by default None.
        increment_proc : bool, optional
            if a process/XXX-<function-name> already exist and increment_proc is set to true, the result will be save in the existing folder, otherwise it will generated a new folder, by default True
        **kwargs : Any
            All the keyword arguments for the above function. If an HyPath is present in kwargs, then self.file.read() will be called for this argument.
        """

        def convert_to_list(inputs):
            if isinstance(inputs, list):
                return inputs
            return [inputs]

        if output_names is None:
            output_names = function.__name__
        output_names = convert_to_list(output_names)

        result, list_args, list_kwargs = HyApply(self, function, args, kwargs).apply()

        if result is None:
            return None
        if not isinstance(result, tuple):
            result = tuple([result])

        if len(output_names) != len(result):
            raise ValueError(
                f"Error: Unequal amount of outputs ({len(result)}) and output names ({len(output_names)})."
            )

        num_proc = len(self.read("process"))

        if increment_proc or self._generate_process_folder_name(num_proc, function) not in self.read("process"):
            num_proc += 1

        out_folder_location = self._generate_process_folder_name(num_proc, function)

        for name, data in zip(output_names, result):
            self._create_dataset((f"process/{out_folder_location}/{name}", data))

            self._write_generic_attributes(
                f"process/{out_folder_location}/{name}",
                list_args,
                output_name=name,
                function=function,
            )
            self._write_kwargs_as_attributes(
                f"process/{out_folder_location}/{name}", function, list_kwargs, first_kwarg=len(list_args)
            )

    def multiple_apply(self, function, list_args, /, output_names=None, smart=False, increment_proc=True, **kwargs):
        if output_names is None:
            output_names = [arg.path.split("/")[-1] for arg in list_args]

        for args, output in zip(list_args, output_names):
            self.apply(function, args, increment_proc=increment_proc, output_names=output, **kwargs)
            increment_proc = False

    def _generate_process_folder_name(self, num_proc: int, function: Callable) -> str:
        return f"{str(num_proc).zfill(3)}-{function.__name__}"

    def _write_generic_attributes(
        self, out_folder_location: str, args: tuple[Any], output_name: str, function: Callable
    ) -> None:
        operation_name = out_folder_location.split("/")[1]
        new_attrs = {
            "path": out_folder_location + output_name,
            "shape": np.shape(self.read(out_folder_location)),
            "name": output_name,
        }

        new_attrs["operation name"] = (function.__module__ or "None") + "." + function.__name__

        if function.__module__ == "__main__":
            new_attrs["function code"] = inspect.getsource(function)

        new_attrs["operation number"] = operation_name.split("-")[0]
        new_attrs["time"] = str(datetime.now())
        for i, arg in enumerate(args):
            new_attrs[f"args_{i}"] = arg

        for k, v in new_attrs.items():
            if isinstance(v, HyPath):
                v = v.path
            print(k, v)
            self.attrs[out_folder_location] = (k, v)

    def _write_kwargs_as_attributes(
        self, path: str, func: Callable, all_variables: dict[str, Any], first_kwarg: int = 1
    ) -> None:
        attr_dict = {}
        if isinstance(func, types.BuiltinFunctionType):
            attr_dict["BuiltinFunctionType"] = True
        else:
            signature = inspect.signature(func).parameters
            var_names = list(signature.keys())[first_kwarg:]
            for key in var_names:
                if key in all_variables:
                    value = all_variables[key]
                elif isinstance(signature[key].default, np._globals._NoValueType):  # type: ignore
                    value = "None"
                else:
                    value = signature[key].default

                if callable(value):
                    value = value.__name__
                elif value is None:
                    value = "None"

                try:
                    attr_dict[f"kwargs_{key}"] = value
                except RuntimeError:
                    RuntimeWarning("Attribute was not able to be saved, probably because the attribute is too large")
                    attr_dict[f"kwargs_{key}"] = "None"

        for k, v in attr_dict.items():
            self.attrs[path] = (k, v)

    def extract_data(self, path: str | Path, ignore_if_exist=True, **kwargs) -> None:
        """Extract the data, metadata and attributes from a file given by path.
        Currently supported files are:

        - .gsf (Gwyddion Simple Field): generated by Gwyddion
        - .ibw (Igor Binary Wave): generated by Asylum AFM. (might work for other kind of ibw files)
        - .000 : Nanoscope files
        - .ARDF : generated by Asylum AFM (for ForceMaps and SSPFM)

        Parameters
        ----------
        path : str | Path
            path to the file to be converted. If a string is provided it is converted to Path.

        Raises
        ------
        TypeError
            If the file you pass through path does not have a conversion function, will raise an error.
        """
        if isinstance(path, str):
            path = Path(path)

        if ignore_if_exist and Path.exists(path):
            warnings.warn(
                f"File {path} already exists, ignoring it. If you want to overwrite it, set ignore_if_exist=False."
            )
            return

        extracted = HyExtractor.extract(path, **kwargs)

        self._write_extracted_data(path, extracted)

    def path_search(self, criterion: str | list[str] = "*"):
        if not isinstance(criterion, list):
            criterion = [criterion]

        criterion = [crit.path if isinstance(crit, HyPath) else crit for crit in criterion]
        all_path = self._find_paths_of_all_subgroups()
        correct_path = []
        for criteria in criterion:
            for path in all_path:
                if re.match(criteria, path) is not None:
                    correct_path.append(HyPath(path))

        return correct_path

    def _expand_path(self, list_of_paths: list[list[str]], mode: str = "block"):
        max_length = len(max(list_of_paths, key=len))
        for paths in list_of_paths:
            if mode in ["b", "block"]:
                paths = np.repeat(paths, max_length // len(paths))
            elif mode in ["a", "alt"]:
                ...
            else:
                raise ValueError(f"Mode {mode} is not a valid one. Valid one is 'block' ('b') or 'alt' ('a').")

    def _find_paths_of_all_subgroups(self, current_path=""):
        """
        Recursively determines list of paths for all datafiles in current_path, as well as datafiles in
        all subfolders (and sub-subfolders and...) of current path. If no path given, will find all
        subfolders in entire file.

        Parameters
        ----------
        f : open file
            open hdf5 file
        current_path : string
            current group searched

        Returns
        ------
        path_list : list
            list of paths to datafiles
        """
        path_list = []
        if current_path == "":
            curr_group = self.file
        else:
            curr_group = self.file[current_path]

        for sub_group in curr_group:
            if current_path == "":
                new_path = f"{sub_group}"
            else:
                new_path = f"{current_path}/{sub_group}"
            if isinstance(self.file[new_path], h5py.Group):
                path_list.extend(self._find_paths_of_all_subgroups(new_path))
            elif isinstance(self.file[new_path], h5py.Dataset):
                path_list.append(new_path)
        return path_list

    def _require_group(self, name: str, f=None):
        if f is None:
            f = self.file
        f.require_group(name)

    def _create_dataset(self, dataset: tuple[str, Any], f=None, overwrite=True) -> None:
        if f is None:
            f = self.file

        key, data = dataset
        if key == "":
            warnings.warn(f"There is an empty key, the value '{key}':'{data}' will be ignored.")
        else:
            if key in f:
                if overwrite:
                    del f[key]
                else:
                    raise KeyError("Key already exist and overwriste is set to False.")

            f.create_dataset(key, data=data)

    def _generate_deep_groups(self, deep_dict: dict[str, dict], f: Optional[h5pyType] = None):
        """_generate_deep_groups
        Given a nested dictionnary and an hdf5 file opened with h5py,
        it creates the necessary groups and subgroups (folder) in the hdf5 to mimic the structure of the nested dict.
        The leaf of the nested dict should be the data you want to save.

        Parameters
        ----------
        deep_dict : dict[str, dict]
            Nested dictionnary containing the folders, subfolders and data that you want to save in the hdf5 file
        f : Optional[h5py.File], optional
            current position in the file, by default None: it will take the file self.file and start from its root.
        """
        if f is None:
            f = self.file

        for key, val in deep_dict.items():
            if isinstance(deep_dict[key], dict):
                self._require_group(key, f)
                self._generate_deep_groups(val, f[key])
            else:
                try:
                    self._create_dataset((key, val), f)
                except:
                    raise Exception(
                        f"'{key}':'{val}' are not working. Types are:'{type(key)}' and '{type(val)}' \n full dict is : {deep_dict}"
                    )

    def _generate_deep_attributes(self, deep_dict, f=None):
        if f is None:
            f = self.file
        for key, val in deep_dict.items():
            if isinstance(deep_dict[key], dict):
                self._generate_deep_attributes(val, f[key])
            else:
                if key != "":
                    f.attrs[key] = val

    def _write_extracted_data(self, path: Path, extracted_values: HyConvertedData) -> None:
        self._require_group(f"datasets/{path.stem}")
        self._generate_deep_groups(extracted_values.data, self.file[f"datasets/{path.stem}"])

        self.file["metadata"].create_dataset(path.stem, data=str(extracted_values.metadata))

        self._generate_deep_attributes(extracted_values.metadata, self.file[f"metadata/{path.stem}"])
        self._generate_deep_attributes(extracted_values.attributes, self.file[f"datasets/{path.stem}"])
