File Conversion
===============

Hystorian provides utilities to convert raw data files from common SPM instruments into HDF5 format.

HyConvertedData
---------------

.. automodule:: hystorian.io.utils
    :members: HyConvertedData

Conversion Functions
--------------------

These functions convert standard raw data files from SPM labs into HDF5 files.

Supported formats:

- GSF files
- IBW files
- ARDF files
- Nanoscope files

.. automodule:: hystorian.io.extractors.gsf_files
    :members:

.. automodule:: hystorian.io.extractors.ibw_files
    :members:

.. automodule:: hystorian.io.extractors.ardf_files
    :members:

.. automodule:: hystorian.io.extractors.nanoscope_files
    :members:

Example: Convert a raw file
---------------------------
.. code-block:: python

    from hystorian.io.hyFile import HyFile
    with HyFile('output.hdf5', 'r+') as f:
        f.extract_data('/path/to/raw_file.ibw')