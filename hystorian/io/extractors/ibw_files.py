from pathlib import Path
from typing import Any

import h5py
import numpy as np
from igor2 import binarywave

from ..utils import HyConvertedData, conversion_metadata

# ==========================================
# IBW conversion


def extract(filename: Path) -> HyConvertedData:
    """extract_ibw uses igor2 (https://github.com/AFM-analysis/igor2/) to convert ibw files from Asylum AFM.

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData
        The extra saved attributes are: the scale (in m/px), the image offset and the units of each channel.
    """

    def correct_label(label):
        label = [x for x in label if x]  # Remove the empty lists
        label = label[0]  # Remove the unnecessary inception

        # Correct the duplicate letters
        label = [i.decode("UTF-8") for i in label if len(i) > 0]
        label = [x.split("Trace")[0] + "Trace" if "Trace" in x else x for x in label]
        label = [x.split("Retrace")[0] + "Retrace" if "Retrace" in x else x for x in label]
        label = [x.encode() for x in label if len(x) > 0]

        return label

    tmpdata = binarywave.load(filename)["wave"]

    metadata = {}

    for meta in tmpdata["note"].decode("ISO-8859-1").split("\r")[:-1]:
        if len(meta.split(":")) == 2:
            metadata[meta.split(":")[0]] = conversion_metadata(meta.split(":")[1])

    label_list = correct_label(tmpdata["labels"])

    data = {}
    attributes = {}

    fastsize = float(metadata["FastScanSize"])
    slowsize = float(metadata["SlowScanSize"])
    xoffset = float(metadata["XOffset"])
    yoffset = float(metadata["YOffset"])

    for i, k in enumerate(label_list):
        k = k.decode("UTF-8")
        if len(np.shape(tmpdata["wData"])) == 2:
            data[k] = np.flipud(tmpdata["wData"][:, i].T)
            shape = tmpdata["wData"][:, i].T.shape
        else:
            data[k] = np.flipud(tmpdata["wData"][:, :, i].T)
            shape = tmpdata["wData"][:, :, i].T.shape

        attributes[k] = {}

        attributes[k]["shape"] = shape
        attributes[k]["scale_m_per_px"] = fastsize / shape[0]
        attributes[k]["name"] = k
        attributes[k]["size"] = (fastsize, slowsize)
        attributes[k]["offset"] = (xoffset, yoffset)

        if "Phase" in str(k):
            attributes[k]["unit"] = ("m", "m", "deg")
        elif "Amplitude" in str(k):
            attributes[k]["unit"] = ("m", "m", "V")
        elif "Height" in str(k):
            attributes[k]["unit"] = ("m", "m", "m")
        else:
            attributes[k]["unit"] = ("m", "m", "unknown")
    converted = HyConvertedData(data, metadata, attributes)
    return converted
