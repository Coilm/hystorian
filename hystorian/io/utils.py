import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HyConvertedData:
    """Dataclass containing the converted data from proprietary files. It contains:

    - data: a dict containing all the raw experimental datas.
    - metadata: all the extra informations extracted from the file, usually in a raw form.
    - Some default attributes (like the size of the data), and when convenient some extra attributes (usually extracted from the metadata) for ease of use.

    """

    data: dict[str, Any]
    metadata: dict[str, Any]
    attributes: dict[str, dict[str, Any]]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def conversion_metadata(value):
    if is_number(value):
        if float(value).is_integer():
            return int(float(value))
        else:
            return float(value)
    else:
        return value.strip()


def check_extension(regex: str, filepath: Path) -> bool:
    return re.match(regex, filepath.suffix) is not None


def find_common_path(pathlist):
    com_pref = os.path.commonprefix([str(pathlist[i]) for i in range(len(pathlist))])
    if "/" not in com_pref:
        return pathlist
    else:
        n = com_pref.rfind("/")
        return [str(str(pathlist[i])[n + 1 :]) for i in range(len(pathlist))]
