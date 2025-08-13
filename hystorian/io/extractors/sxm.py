from pathlib import Path
from typing import Any

import numpy as np

from ..utils import HyConvertedData
# ==========================================
# SXM conversion


def extract(filename: Path) -> HyConvertedData:
    """extract_sxm converts files acquired with Nanonis (*.sxm).

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData
        The extra saved attributes are: the scale (in m/px), the image offset and the units of each channel.
    """

    d = open(filename, 'rb')
    
    metadata = {}
    currkey = ""
    hdrend = 0
    while hdrend == 0:
        line = d.readline().strip().decode('ISO-8859-1')
        if line == ":SCANIT_END:":
            hdrend = 1
            continue
        if (line.startswith(':')):
            currkey = line
            metadata[currkey] = []
        else:
            metadata[currkey].append(line)
    for k in metadata.keys():
        metadata[k] = "\n".join(metadata[k]).strip()
    junk = d.read(4)
    dataraw = d.read()

    clist = metadata[":DATA_INFO:"].split("\n")
    chans = []
    for c in clist[1:]:
        ca = c.split("\t")
        if ca[3] == 'both':
            chans.append([ca[1] + " forward", ca[2], ca[4], ca[5]])
            chans.append([ca[1] + " backward", ca[2], ca[4], ca[5]])
        else:
            chans.append([ca[1] + " " + ca[3], ca[2], ca[4], ca[5]])

    data = {}
    attributes = {}

    xsizepx = int(metadata[":SCAN_PIXELS:"].split()[0])
    ysizepx = int(metadata[":SCAN_PIXELS:"].split()[1])
    xsizem = float(metadata[":SCAN_RANGE:"].split()[0])
    ysizem = float(metadata[":SCAN_RANGE:"].split()[1])
    xoffset = float(metadata[":SCAN_OFFSET:"].split()[1])
    yoffset = float(metadata[":SCAN_OFFSET:"].split()[1])
    scandir = metadata[":SCAN_DIR:"]

    for i in range(len(chans)):
        cdata = np.reshape(np.frombuffer(dataraw[i*4*xsizepx*ysizepx:(i+1)*4*xsizepx*ysizepx], dtype='>f4'),(xsizepx, ysizepx))
        if scandir == 'up':
            cdata = np.flipud(cdata)
        if (chans[i][0].endswith(" backward")):
            cdata = np.fliplr(cdata)
        data[chans[i][0]] = np.copy(cdata)
        attributes[chans[i][0]] = {}

        attributes[chans[i][0]]["shape"] = (xsizepx,ysizepx)
        attributes[chans[i][0]]["scale_m_per_px"] = xsizem/xsizepx
        attributes[chans[i][0]]["name"] = chans[i][0]
        attributes[chans[i][0]]["size"] = (xsizem, ysizem)
        attributes[chans[i][0]]["offset"] = (xoffset, yoffset)
        attributes[chans[i][0]]["unit"] = ("m","m",chans[i][1])
    converted = HyConvertedData(data, metadata, attributes)
    return converted