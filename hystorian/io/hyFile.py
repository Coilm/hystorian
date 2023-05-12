import inspect
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, KeysView, Optional, overload

import h5py
import numpy as np
import numpy.typing as npt

from . import ardf_files, gsf_files, ibw_files
from .utils import HyConvertedData


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

    def __getitem__(self, path: str = "") -> KeysView | h5py.Group | h5py.Dataset | h5py.Datatype:
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

    @overload
    def read(self) -> list[str]:
        pass

    @overload
    def read(self, path: str) -> list[str]:
        pass

    @overload
    def read(self, path: str) -> h5py.Datatype:
        pass

    @overload
    def read(self, path: str) -> npt.ArrayLike:
        pass

    def read(self, path: Optional[str] = None) -> list[str] | h5py.Datatype | npt.ArrayLike:
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
        else:
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
        inputs: list[str] | str,
        output_names: Optional[list[str] | str] = None,
        increment_proc: bool = True,
        **kwargs: dict[str, Any],
    ):
        """apply allows to call a function and store all the inputs and outputs in the hdf5 file with the raw data.

        Parameters
        ----------
        function : Callable
            function used to transform the data. Result of the function will be stored in process/XXX-<function-name>, where XXX is an incrementing number for each already existing process.
        inputs : list[str] | str
            path into the hdf5 to the input of the function, most of the time it should be the data that need processing.
        output_names : Optional[list[str]  |  str], optional
            Name to be given to the result of the function. The number of names should be the same as the number of outputs of the function passed,
            othewise a ValueError will be raised, if None is passed, the name of the data passed in inputs will be used, by default None.
        increment_proc : bool, optional
            if a process/XXX-<function-name> already exist and increment_proc is set to true, the result will be save in the existing folder, otherwise it will generated a new folder, by default True
        """

        def convert_to_list(inputs):
            if isinstance(inputs, list):
                return inputs
            return [inputs]

        inputs = convert_to_list(inputs)

        if output_names is None:
            output_names = inputs[0].rsplit("/", 1)[1]
        output_names = convert_to_list(output_names)

        result = function(*inputs, **kwargs)

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
            self._create_dataset((f"{out_folder_location}/{name}", data))

            self._write_generic_attributes(f"{out_folder_location}/{name}", inputs, name, function)
            self._write_kwargs_as_attributes(
                f"{out_folder_location}/{name}", function, kwargs, first_kwarg=len(inputs)
            )

    def _generate_process_folder_name(self, num_proc: int, function: Callable) -> str:
        return f"{str(num_proc).zfill(3)}-{function.__name__}"

    def _write_generic_attributes(
        self, out_folder_location: str, in_paths: list[str] | str, output_name: str, function: Callable
    ) -> None:
        if not isinstance(in_paths, list):
            in_paths = [in_paths]

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
        new_attrs["source"] = in_paths

        for k, v in new_attrs:
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

        for k, v in attr_dict:
            self.attrs[path] = (k, v)

    def extract_data(self, path: str | Path) -> None:
        """Extract the data, metadata and attributes from a file given by path.
        Currently supported files are:
        - .gsf (Gwyddion Simple Field): generated by Gwyddion
        - .ibw (Igor Binary Wave): generated by Asylum Cypher AFM. (might work for other kind of ibw files)

        Parameters
        ----------
        path : str | Path
            path to the file to be converted. If a string is provided it is converted to Path.

        Raises
        ------
        TypeError
            If the file you pass through path does not have a conversion function, will raise an error.
        """
        conversion_functions = {
            ".ardf": ardf_files.extract_ardf,
            ".gsf": gsf_files.extract_gsf,
            ".ibw": ibw_files.extract_ibw,
        }

        if isinstance(path, str):
            path = Path(path)
        suffix = path.suffix.lower()
        if suffix in conversion_functions:
            # data, metadata, attributes = conversion_functions[path.suffix](path)
            extracted = conversion_functions[suffix](path)
            self._write_extracted_data(path, extracted)
        else:
            raise TypeError(f"{suffix} file doesn't have a conversion function.")

    def _require_group(self, name: str, f=None):
        if f is None:
            f = self.file
        f.require_group(name)

    def _create_dataset(self, dataset: tuple[str, Any], f=None, overwrite=True) -> None:
        if f is None:
            f = self.file

        key, data = dataset
        if key in f:
            if overwrite:
                del f[key]
            else:
                raise KeyError("Key already exist and overwriste is set to False.")

        f.create_dataset(key, data=data)

    def _generate_deep_groups(self, deep_dict, f=None):
        if f is None:
            f = self.file

        for key, val in deep_dict.items():
            if isinstance(deep_dict[key], dict):
                self._require_group(key, f)
                self._generate_deep_groups(val, f[key])
            else:
                self._create_dataset((key, val), f)

    def _generate_deep_attributes(self, deep_dict, f=None):
        if f is None:
            f = self.file
        for key, val in deep_dict.items():
            if isinstance(deep_dict[key], dict):
                self._generate_deep_attributes(val, f[key])
            else:
                f.attrs[key] = val

    def _write_extracted_data(self, path: Path, extracted_values: HyConvertedData) -> None:
        self._require_group(f"datasets/{path.stem}")
        self._generate_deep_groups(extracted_values.data, self.file[f"datasets/{path.stem}"])

        self._require_group(f"metadata/{path.stem}")
        self._generate_deep_groups(extracted_values.metadata, self.file[f"metadata/{path.stem}"])

        self._generate_deep_attributes(extracted_values.attributes, self.file[f"datasets/{path.stem}"])
