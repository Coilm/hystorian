from pathlib import Path

import h5py
import numpy as np

from ..utils import HyConvertedData, conversion_metadata, is_number


def load_nanoscope(path):
    with open(path, "rb") as fn:
        file_str = fn.read().decode("iso-8859-1")

        header = file_str.split("File list end\r\n")[0]
        scan_info = extract_scan_info(header)

        header_of_images = header.split("\\*Ciao image list\r\n")[1:]
        image_infos = [extract_image_info(head.split("\r\n")) for head in header_of_images]

        data_offset = int(image_infos[0]["Data offset"])
        with open(path, "rb") as fn:
            data_orig = np.frombuffer(fn.read(), dtype="<h", offset=data_offset)
        data = {}
        start = 0
        for chan in image_infos:
            tempX = int(chan["Valid data len X"])
            tempY = int(chan["Valid data len X"])
            tempD = data_orig[start : start + tempX * tempY].reshape((tempY, tempX))
            data[chan["@2:Image Data"].split('"')[1].replace(" ", "") + chan["Line Direction"]] = tempD
            start = start + tempX * tempY
        image_infos = {
            chan["@2:Image Data"].split('"')[1].replace(" ", "") + chan["Line Direction"]: chan for chan in image_infos
        }
        return data, scan_info, image_infos


def extract_scan_info(header):
    scan_header = header.split("\\*Ciao scan list")[1].split("\\*")[0]
    scan_dict = {}

    for line in scan_header.split("\r\n"):
        if len(line.split(":")) < 2:
            continue
        key, val = ":".join(line.split(":")[:-1])[1:], line.split(":")[-1].strip()

        scan_dict[key] = conversion_metadata(conversion_units(val))
    return scan_dict


def extract_image_info(header):
    header_dict = {}

    for line in header:
        line = line.strip("\\")
        if len(line.split(":")) < 2:
            continue

        key = ":".join(line.split(":")[:-1])
        value = line.split(":")[-1]

        header_dict[key] = conversion_metadata(conversion_units(value))

    return header_dict


def conversion_units(dat):
    unit_dic = {
        "am": 1e-18,
        "fm": 1e-15,
        "pm": 1e-12,
        "nm": 1e-9,
        "~m": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "dm": 1e-1,
        "m": 1.0,
        "km": 1e3,
    }
    if len(dat.split(" ")) == 2:
        value, unit = dat.split(" ")
        if unit in unit_dic and is_number(value):
            return float(value) * unit_dic[unit]

    return dat


def extract(path: Path) -> HyConvertedData:
    """extract_nanoscope to convert Nanonis files to hdf5

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData
        The metadata contains the scan_info from the orignal file, where the image_info are saved as attribute for each channel respectively
    """
    data, scan_info, image_info = load_nanoscope(path)

    metadata = scan_info
    attributes = {}
    for chan in data:
        attributes[chan] = {}
        attributes[chan]["name"] = chan
        attributes[chan]["size"] = np.shape(data[chan])
        for key, value in image_info[chan].items():
            attributes[chan][key] = value

    extracted = HyConvertedData(data, metadata, attributes)
    return extracted
