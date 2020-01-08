from .io.read_file import *
from .io.ardf_files import *
from .io.ibw_files import *
from .io.xrdml_files import *

from .processing import core
from .processing import ndim
from .processing import twodim

try:
    import h5py
    major, minor, patch = [int(x, 10) for x in h5py.__version__.split('.')]
    if major <2:
        if minor < 8:
            print('Carefull, h5py version is lower than 2.9.x, please consider updating it, since it changes the way the attributes are handeled')
except:
    print('please install h5py version 2.9.x or higher')