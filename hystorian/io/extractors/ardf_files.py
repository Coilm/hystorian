import gc
import struct
from pathlib import Path

import chardet
import numpy as np

from ..utils import HyConvertedData, conversion_metadata

"""
Following code was translated from the original Matlab source code written by Matthew Poss available here: https://ch.mathworks.com/matlabcentral/fileexchange/80212-ardf-to-matlab/?s_tid=LandingPageTabfx
"""


def smart_decode(b):
    try:
        decoded = b.decode("UTF-8")
        return decoded.replace("\x00", "")
    except:
        try:
            encoding = chardet.detect(b)["encoding"]
            return b.decode(encoding).replace("\x00", "")
        except:
            return b


def read_convert(fid, bitLen, type_):
    result = np.fromfile(fid, type_, count=bitLen)
    if len(result) == 1:
        return result.item()
    else:
        return result


# ==================
# readARDF Functions
# ==================
def readARDF(FN):
    fid = open(FN, "rb")

    _, _, lastType, _ = local_readARDFpointer(fid, 0)
    local_checkType(lastType, "ARDF", fid)
    F = {}
    D = {}

    F["ftoc"] = local_readTOC(fid, -1, "FTOC")

    loc_TTOC = F["ftoc"]["sizeTable"] + 16
    F["ttoc"] = local_readTOC(fid, loc_TTOC, "TTOC")

    F["ttoc"]["numbNotes"] = len(F["ttoc"]["pntText"])

    if F["ttoc"]["numbNotes"] > 0:
        noteMain = local_readTEXT(fid, F["ttoc"]["pntText"][0])
        D = extractImages(fid, F, noteMain)

    F["numbVolm"] = len(F["ftoc"]["pntVolm"])
    D["channelList"] = []
    for n in range(0, F["numbVolm"]):
        volmN = f"volm{n + 1}"

        F[volmN] = local_readTOC(fid, F["ftoc"]["pntVolm"][n], "VOLM")

        loc_VOLM_TTOC = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"]
        F[volmN]["ttoc"] = local_readTOC(fid, loc_VOLM_TTOC, "TTOC")

        loc_VDEF_IMAG = F["ftoc"]["pntVolm"][n] + F[volmN]["sizeTable"] + F[volmN]["ttoc"]["sizeTable"]
        F[volmN]["vdef"] = local_readDEF(fid, loc_VDEF_IMAG, "VDEF")

        F[volmN]["vchn"] = []
        F[volmN]["xdef"] = {}

        done = False
        while not done:
            _, lastSize, lastType, _ = local_readARDFpointer(fid, -1)
            if lastType == "VCHN":
                textSize = 32
                theChannel = smart_decode(fid.read(textSize))
                F[volmN]["vchn"].append(theChannel)

                remainingSize = lastSize - 16 - textSize
                _ = fid.read(remainingSize)
            elif lastType == "XDEF":
                _ = fid.read(4)

                F[volmN]["xdef"]["sizeTable"] = read_convert(fid, 1, np.uint32)
                F[volmN]["xdef"]["text"] = smart_decode(fid.read(F[volmN]["xdef"]["sizeTable"]))

                _ = fid.read(lastSize - 16 - 8 - F[volmN]["xdef"]["sizeTable"])
                done = True
            else:
                raise ValueError(f"'{lastType}' is not recognized!")

        D["channelList"] = D["channelList"] + F[volmN]["vchn"]
        F[volmN]["idx"] = local_readTOC(fid, -1, "VTOC")

        _, lastSize, lastType, _ = local_readARDFpointer(fid, -1)
        local_checkType(lastType, "MLOV", fid)

        F[volmN]["line"] = {}
        for r in range(F[volmN]["vdef"]["lines"]):
            vsetN = f"vset{r + 1}"
            loc = F[volmN]["idx"]["linPointer"][r]

            if loc != 0:
                F[volmN]["line"][vsetN] = local_readVSET(fid, loc)

                if F[volmN]["line"][vsetN]["line"] != r:
                    F[volmN]["scanDown"] = 1
                else:
                    F[volmN]["scanDown"] = 0

                if F[volmN]["line"][vsetN]["point"] == 0:
                    F[volmN]["trace"] = 1
                else:
                    F[volmN]["trace"] = 0

        try:
            idxZero = [i for i, val in enumerate(F[volmN]["idx"]["linPointer"]) if val == 0]
            incMin = 1
            incMax = 0

            if F[volmN]["scanDown"] == 1:
                idxZero = F[volmN]["vdef"]["lines"] - idxZero + 1
                incMin = 0
                incMax = 1
        except:
            pass
        if len(idxZero) > 0:
            idxZeroMin = min(idxZero) - incMin
            idxZeroMax = max(idxZero) + incMax
            for q in range(len(D["y"])):
                D["y"][q][idxZeroMin:idxZeroMax] = []

    D["endNote"] = {}
    D["endNote"]["IsImage"] = "1"
    D["FileStructure"] = F
    fid.close()

    return D


def local_readVSET(fid, address):
    vset = {}
    if address != -1:
        fid.seek(address)

    _, _, lastType, _ = local_readARDFpointer(fid, -1)
    local_checkType(lastType, "VSET", fid)

    vset["force"] = read_convert(fid, 1, np.uint32)
    vset["line"] = read_convert(fid, 1, np.uint32)
    vset["point"] = read_convert(fid, 1, np.uint32)
    _ = read_convert(fid, 1, np.uint32)
    vset["prev"] = read_convert(fid, 1, np.uint64)
    vset["next"] = read_convert(fid, 1, np.uint64)

    return vset


def extractImages(fid, F, noteMain):
    D = {}

    F["numbImag"] = len(F["ftoc"]["pntImag"])

    D["Notes"] = parseNotes(noteMain)

    D["imageList"] = []
    D["y"] = []

    for n in range(F["numbImag"]):
        imagN = f"imag{n+1}"

        F[imagN] = local_readTOC(fid, F["ftoc"]["pntImag"][n], "IMAG")

        loc_IMAG_TTOC = F["ftoc"]["pntImag"][n] + F[imagN]["sizeTable"]
        F[imagN]["ttoc"] = local_readTOC(fid, loc_IMAG_TTOC, "TTOC")

        loc_IMAG_IDEF = F["ftoc"]["pntImag"][n] + F[imagN]["sizeTable"] + F[imagN]["ttoc"]["sizeTable"]
        F[imagN]["idef"] = local_readDEF(fid, loc_IMAG_IDEF, "IDEF")

        D["imageList"].append(F[imagN]["idef"]["imageTitle"])

        idat = local_readTOC(fid, -1, "IBOX")
        D["y"].append(idat["data"])

        _, _, lastType, _ = local_readARDFpointer(fid, -1)
        local_checkType(lastType, "GAMI", fid)

        numbImagText = len(F[imagN]["ttoc"]["pntText"])
        noteQuick = None
        noteThumb = None

        for r in range(numbImagText):
            theNote = local_readTEXT(fid, F[imagN]["ttoc"]["pntText"][r])

            if (numbImagText > 1) or (n == 1):
                if r == 1:
                    noteThumb = theNote
                elif r == 2:
                    F[imagN]["note"] = parseNotes(theNote)
                elif r == 3:
                    noteQuick = theNote
                else:
                    raise ValueError(f"'r' should be between 1 and 3. 'r' value is {r}.")
            else:
                F[imagN]["note"] = parseNotes(theNote)

        theNote = noteMain
        if noteQuick:
            theNote += noteQuick
        elif noteThumb:
            theNote += noteThumb

    D["Notes"] = parseNotes(theNote)

    return D


def local_readDEF(fid, address, type):
    def_ = {}

    if address != -1:
        fid.seek(address, 0)

    _, sizeDEF, typeDEF, _ = local_readARDFpointer(fid, -1)
    local_checkType(typeDEF, type, fid)

    def_["points"] = read_convert(fid, 1, np.uint32)
    def_["lines"] = read_convert(fid, 1, np.uint32)

    if typeDEF == "IDEF":
        skip = 96
    elif typeDEF == "VDEF":
        skip = 144
    else:
        raise ValueError(f"'typeDEF' has value {typeDEF}, was expecting 'IDEF' or 'VDEF'.")

    _ = fid.read(skip)

    sizeText = 32
    def_["imageTitle"] = smart_decode(fid.read(sizeText))

    sizeHead = 16
    remainingSize = sizeDEF - 8 - skip - sizeHead - sizeText
    _ = fid.read(remainingSize)

    return def_


def parseNotes(theNote):
    parsedNotes = {}
    for note in theNote.split("\r"):
        if len(note.split(":")) == 2:
            key, val = note.split(":")
            parsedNotes[key] = conversion_metadata(val)

    return parsedNotes


def local_readTEXT(fid, loc):
    fid.seek(loc)
    _, _, lastType, _ = local_readARDFpointer(fid, -1)
    local_checkType(lastType, "TEXT", fid)

    _ = read_convert(fid, 1, np.uint32)
    sizeNote = read_convert(fid, 1, np.uint32)
    txt = fid.read(sizeNote)
    txt = smart_decode(txt)

    return txt


def local_readTOC(fid, address, _type):
    toc = {}

    if address != -1:
        fid.seek(address, 0)

    _, lastSize, lastType, _ = local_readARDFpointer(fid, -1)
    if lastSize == 0:
        return toc
    local_checkType(lastType, _type, fid)

    toc["sizeTable"] = read_convert(fid, 1, np.uint64)
    toc["numbEntry"] = read_convert(fid, 1, np.uint32)
    toc["sizeEntry"] = read_convert(fid, 1, np.uint32)

    if _type in ["FTOC", "IMAG", "VOLM", "NEXT", "THMB", "NSET"]:  # sizeEntry: 24
        toc["pntImag"] = []
        toc["pntVolm"] = []
        toc["pntNext"] = []
        toc["pntNset"] = []
        toc["pntThmb"] = []
    elif _type in ["TTOC", "IBOX", "TOFF"]:  # sizeEntry: 32
        toc["idxText"] = []
        toc["pntText"] = []
    elif _type in ["VOFF", "VTOC"]:
        toc["pntCounter"] = []
        toc["linCounter"] = []
        toc["linPointer"] = []
    elif _type in ["IDAT"]:
        toc["data"] = []
        sizeRead = (toc["sizeEntry"] - 16) // 4
    else:
        raise ValueError(f"199 - '{_type}' is not recognized!")

    done = 0
    numbRead = 1

    while (done == 0) and (numbRead <= toc["numbEntry"]):
        _, tmpSize, typeEntry, _ = local_readARDFpointer(fid, -1)
        if tmpSize == 0:
            pass
        elif typeEntry in ["FTOC", "IMAG", "VOLM", "NEXT", "THMB", "NSET"]:
            lastPointer = read_convert(fid, 1, np.uint64)
        elif typeEntry in ["TTOC", "IBOX", "TOFF", "VTOC"]:
            lastIndex = read_convert(fid, 1, np.uint64)
            lastPointer = read_convert(fid, 1, np.uint64)
        elif typeEntry in ["VOFF"]:
            lastPntCount = read_convert(fid, 1, np.uint32)
            lastLinCount = read_convert(fid, 1, np.uint32)
            dum = read_convert(fid, 1, np.uint64)
            lastLinPoint = read_convert(fid, 1, np.uint64)
        elif typeEntry in ["IDAT"]:  # IDAT
            if "sizeRead" not in locals():
                sizeRead = (toc["sizeEntry"] - 16) // 4
            lastData = read_convert(fid, sizeRead, np.uint32)
        else:
            raise ValueError(f"'{typeEntry}' is not recognized!")

        if typeEntry in ["IMAG", "VOLM", "NEXT", "NSET", "THMB"]:
            toc[f"pnt{typeEntry.capitalize()}"].append(lastPointer)
        elif typeEntry in ["TOFF"]:
            toc["idxText"].append(lastIndex)
            toc["pntText"].append(lastPointer)
        elif typeEntry in ["IDAT"]:
            if "data" not in toc:
                toc["data"] = []
            toc["data"].append(lastData)
        elif typeEntry in ["VOFF"]:
            toc["pntCounter"].append(lastPntCount)
            toc["linCounter"].append(lastLinCount)
            toc["linPointer"].append(lastLinPoint)
        else:
            if lastType in ["IBOX"]:
                toc["data"].append(lastData)
            elif lastType in ["VTOC"]:
                toc["pntCounter"].append(lastPntCount)
                toc["linCounter"].append(lastLinCount)
                toc["linPointer"].append(lastLinPoint)
            else:
                done = 1

        numbRead += 1

    return toc


def local_readARDFpointer(fid, address):
    if address != -1:
        fid.seek(address, 0)

    checkCRC32 = read_convert(fid, 1, np.uint32)
    sizeBytes = read_convert(fid, 1, np.uint32)
    typePnt = smart_decode(fid.read(4))
    miscNum = read_convert(fid, 1, np.uint32)

    return checkCRC32, sizeBytes, typePnt, miscNum


def local_checkType(found, test, fid):
    if found != test:
        raise ValueError(f"Error: No '{test}' here! Found: '{found}' at Location: {fid.tell()-16}")


# ==================
# getARDFdata Functions
# ==================


def getARDFdata(FN, getLine=0, trace=0, fileStruct=None):
    fid = open(FN, "rb")

    if fileStruct:
        F = fileStruct
    else:
        F = readARDF(FN)["FileStructure"]

    getVolm = "volm1"
    if F["numbVolm"] > 1:
        if trace == F["volm1"]["trace"]:
            getVolm = "volm1"
        else:
            getVolm = "volm2"

    numbPoints = F[getVolm]["vdef"]["points"]
    numbLines = F[getVolm]["vdef"]["lines"]

    if F[getVolm]["scanDown"] == 1:
        adjLine = numbLines - getLine - 1
    else:
        adjLine = getLine

    numbChannels = len(F["volm1"]["vchn"])

    locLine = F[getVolm]["idx"]["linPointer"][adjLine]

    G = {}

    if locLine == 0:
        fid.close()
        return G

    fid.seek(locLine, 0)
    for key in ["numbForce", "numbLine", "numbPoint", "locPrev", "locNext", "name", "y", "pnt0", "pnt1", "pnt2"]:
        G[key] = []

    for _ in range(numbPoints):
        vset = local_readVSET(fid, -1)
        G["numbForce"].append(vset["force"])
        G["numbLine"].append(vset["line"])
        G["numbPoint"].append(vset["point"])
        G["locPrev"].append(vset["prev"])
        G["locNext"].append(vset["next"])

        vnam = local_readVNAM(fid, -1)
        G["name"].append(vnam["name"])

        theData = []

        for _ in range(numbChannels):
            vdat = local_readVDAT(fid, -1)
            theData.append(vdat["data"])

        local_readXDAT(fid, -1)

        G["y"].append(theData)
        G["pnt0"].append(vdat["pnt0"])
        G["pnt1"].append(vdat["pnt1"])
        G["pnt2"].append(vdat["pnt2"])

        if G["numbPoint"][0] != 0:
            for key in [
                "numbForce",
                "numbLine",
                "numbPoint",
                "locPrev",
                "locNext",
                "y",
                "name",
                "pnt0",
                "pnt1",
                "pnt2",
            ]:
                G[key].reverse()

    fid.close()
    return G


def local_readXDAT(fid, address):
    if address != -1:
        fid.seek(-1)

    _, lastSize, lastType, _ = local_readARDFpointer(fid, -1)

    if lastType == "XDAT":
        stepDist = lastSize - 16
        fid.seek(stepDist, 1)
    elif lastType == "VSET":
        fid.seek(-16, 1)

    else:
        raise ValueError(f"Error: No 'XDAT' or 'VSET' here! Found: '{lastType}' at Location: {fid.tell()-16}")


def local_readVNAM(fid, address):
    vnam = {}

    if address != -1:
        fid.seek(-1)

    _, lastSize, lastType, _ = local_readARDFpointer(fid, -1)
    local_checkType(lastType, "VNAM", fid)

    vnam["force"] = read_convert(fid, 1, np.uint32)
    vnam["line"] = read_convert(fid, 1, np.uint32)
    vnam["point"] = read_convert(fid, 1, np.uint32)
    vnam["sizeText"] = read_convert(fid, 1, np.uint32)
    vnam["name"] = smart_decode(fid.read(vnam["sizeText"]))

    remainingSize = lastSize - 16 - vnam["sizeText"] - 16
    _ = fid.read(remainingSize)

    return vnam


def local_readVDAT(fid, address):
    vdat = {}

    if address != -1:
        fid.seek(address, 0)

    _, _, lastType, _ = local_readARDFpointer(fid, -1)
    local_checkType(lastType, "VDAT", fid)

    for key in ["force", "line", "point", "sizeData", "forceType", "pnt0", "pnt1", "pnt2"]:
        vdat[key] = read_convert(fid, 1, np.uint32)

    _ = fid.read(8)

    vdat["data"] = np.fromfile(fid, np.single, count=vdat["sizeData"])
    return vdat


def extract(filename: Path) -> HyConvertedData:
    """Extract datas based on original Matlab source code written by Matthew Poss available here: https://ch.mathworks.com/matlabcentral/fileexchange/80212-ardf-to-matlab/?s_tid=LandingPageTabfx

    Parameters
    ----------
    filename : Path
        The name of the input file to be converted

    Returns
    -------
    HyConvertedData
    """

    gc.enable()
    F = readARDF(filename)
    data = {}
    attributes = {}
    metadata = F["Notes"]

    for idx, channelName in enumerate(F["imageList"]):
        data[channelName] = F["y"][idx]

        attributes[channelName] = {}

        attributes[channelName]["name"] = channelName
        attributes[channelName]["shape"] = np.shape(F["y"][idx])
        attributes[channelName]["unit"] = "unknown"

    lines = F["FileStructure"]["volm1"]["vdef"]["lines"]
    points = F["FileStructure"]["volm1"]["vdef"]["points"]
    for idx, channelName in enumerate(F["FileStructure"]["volm1"]["vchn"]):
        data[channelName] = {}
        attributes[channelName] = {}

        channel_result = {}
        max_size = 0
        for tracetype, key in enumerate(["retrace", "trace"]):
            channel_result[key] = []

            for point in range(points):
                result_line = getARDFdata(filename, point, tracetype, F["FileStructure"])
                result_line = list(np.array(result_line["y"], dtype=object)[:, idx])

                length = max(list(map(len, result_line)))
                if length > max_size:
                    max_size = length

                result_line = list(
                    map(
                        lambda elem: list(elem) + [np.nan] * (max_size - len(elem)),
                        result_line,
                    )
                )
                channel_result[key].append(result_line)

            for line in range(lines):
                for point in range(points):
                    if len(channel_result[key][line][point]) < max_size:
                        channel_result[key][line][point] = channel_result[key][line][point] + [np.nan] * (
                            max_size - len(channel_result[key][line][point])
                        )

            data[channelName][key] = np.array(channel_result[key])

            attributes[channelName]["name"] = channelName

            attributes[channelName][key] = {}

            attributes[channelName][key]["shape"] = data[channelName][key].shape

    gc.disable()
    extracted = HyConvertedData(data, metadata, attributes)
    return extracted
