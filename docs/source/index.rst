Hystorian Documentation
=======================

Hystorian is a Python package based on h5py, designed for scientists to help manage, archive, and analyze data from Scanning Probe Microscopy (SPM) experiments and related sources.

Overview
--------
Hystorian provides a unified framework for:

- Storing experimental data in HDF5 format for efficient access and archiving.
- Converting raw data files from common SPM instruments (e.g., GSF, IBW, Nanoscope, ARDF) into HDF5.
- Extracting, processing, and analyzing experimental datasets.
- Applying advanced image alignment and distortion correction algorithms.
- Managing experiment metadata, results, and reproducibility.
- Extensible processing: apply custom and built-in processing functions to datasets.

Key Features
------------
- **Data Conversion:** Import and convert raw files from various SPM instruments.
- **Unified Data Storage:** Organize and archive data using HDF5, supporting metadata and hierarchical structures.
- **Distortion Correction:** Correct image distortions for accurate quantitative analysis.
- **Experiment Management:** Tools for handling experiment metadata, results, and reproducibility.
- **Extensible Processing:** Apply custom and built-in processing functions to datasets.

Installation
------------
.. code-block:: bash

    pip install hystorian

Quick Start
-----------
.. code-block:: python

    from hystorian.io.hyFile import HyFile
    with HyFile('mydata.hdf5', 'r+') as f:
        f.extract_data('/path/to/file.ibw')

API Reference
-------------
This section provides a comprehensive overview of the Hystorian Python package API, organized by module. Use the links below to explore the available classes, functions, and utilities for data management, conversion, processing, and experiment analysis.

Core Modules
------------
- :doc:`hyfile` — HDF5 file management and data extraction
- :doc:`conversion` — Raw file conversion utilities
- :doc:`processing` — Distortion correction and data processing
- :doc:`experiments` — Experiment organization and analysis


How to Use
~~~~~~~~~~

.. code-block:: python

    # Open and extract data from an HDF5 file
    from hystorian.io.hyFile import HyFile
    with HyFile('data.hdf5', 'r+') as f:
        f.extract_data('/path/to/raw/file.ibw')

    # Apply processing functions
    from hystorian.processing.distortion import find_transform_ecc
    warp_matrix = find_transform_ecc(reference_image, image_to_align)

Documentation Contents
----------------------
.. toctree::
    :maxdepth: 2

    hyfile
    conversion
    experiments
    processing

Publication Source
------------------
For a detailed description of the concepts and algorithms implemented in Hystorian, see:

Loïc Musy, Ralph Bulanadi, Iaroslav Gaponenko, Patrycja Paruch,  
*Hystorian: A processing tool for scanning probe microscopy and other n-dimensional datasets*
Ultramicroscopy, Volume 228, 2021  
https://www.sciencedirect.com/science/article/pii/S0304399121001273


.. note::

    This project is under active development.