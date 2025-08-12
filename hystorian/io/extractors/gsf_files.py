from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..utils import HyConvertedData


def extract(filename: Path) -> HyConvertedData:
    """extract Read a Gwyddion Simple Field 1.0 file format
    http://gwyddion.net/documentation/user-guide-en/gsf.html

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData

    Raises
    ------
    ValueError
        Returned if the header of the file is not "Gwyddion Simple Field 1.0", thus indicating that the gwyddion version used to generated the file is not supported.
    """
    # if filename.rpartition(".")[1] == ".":
    #    filename = filename[0 : filename.rfind(".")]

    gsfFile = open(filename, "rb")  # + ".gsf", "rb")

    metadata = {}

    # check if header is OK
    if not (gsfFile.readline().decode("UTF-8") == "Gwyddion Simple Field 1.0\n"):
        gsfFile.close()
        raise ValueError("File has wrong header")

    term = b"00"
    # read metadata header
    while term != b"\x00":
        line_string = gsfFile.readline().decode("UTF-8")
        metadata[line_string.rpartition("=")[0].strip()] = line_string.rpartition("=")[2].strip()
        term = gsfFile.read(1)

        gsfFile.seek(-1, 1)

    gsfFile.read(4 - gsfFile.tell() % 4)

    # fix known metadata types from .gsf file specs
    # first the mandatory ones...
    for key in metadata:
        try:
            if metadata[key].isdigit():
                metadata[key] = int(metadata[key])
            else:
                metadata[key] = float(metadata[key])
        except ValueError:
            continue

    data = np.frombuffer(gsfFile.read(), dtype="float32").reshape(metadata["YRes"], metadata["XRes"])

    gsfFile.close()

    name = filename.stem
    data = {name: data}

    attributes = {name: {}}
    attributes[name]["name"] = name
    attributes[name]["size"] = (metadata["XRes"], metadata["YRes"])
    if "XOffset" in metadata and "YOffset" in metadata:
        attributes[name]["offset"] = (metadata["XOffset"], metadata["YOffset"])
    if "WavenumberScaling" in metadata:
        attributes[name]["WavenumberScaling"] = float(metadata["Neaspec_WavenumberScaling"])
    extracted = HyConvertedData(data, metadata, attributes)
    return extracted
