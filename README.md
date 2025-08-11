# What is Hystorian?

Keeping track of post-treatment done on scientific data can be cumbersome.

**Hystorian** is a Python package developed by our AFM group to address this issue. It provides two main functionalities:

1. **Unified Data Format:** Handles multiple file formats (AFM, SPM, etc.) and converts them into uniform HDF5 files.
2. **Processing History Tracking:** Keeps track of all processing steps performed on the data, storing parameters and results in the HDF5 file.

Find the link to the paper here: [http://dx.doi.org/10.1016/j.ultramic.2021.113345](http://dx.doi.org/10.1016/j.ultramic.2021.113345)

If you use Hystorian in your research, please cite the paper.

## Installation

You can install the package using pip:

```pip install hystorian```

Documentation is available at: [https://hystorian.readthedocs.io/en/latest/](https://hystorian.readthedocs.io/en/latest/)

## Structure of an HDF5 File

An HDF5 file has a tree-like structure consisting of groups (folders) and datasets. Each group and dataset can have attributes attached, which are used to store metadata and processing parameters.

The structure used in Hystorian is:

1. **Dataset group:** Contains all raw data grouped inside a single HDF5 file.
2. **Metadata group:** Contains metadata from the original files.
3. **Process group:** Contains data processed from the original or other processed data, along with processing parameters.

## Usage Overview

Hystorian provides modules for:

- **File I/O:** Import and export data from various formats (see `io/` and `io/extractors/`).
- **Processing:** Apply corrections, distortions, and operations to your data (see `processing/`).
- **Machine Specific Tools:** Manipulate data file from specific AFMs. (see `experiments/`).

All processing steps and their parameters are automatically tracked and stored in the HDF5 file, ensuring reproducibility.

For detailed usage examples and API reference, please visit the [documentation](https://hystorian.readthedocs.io/en/latest/).

## Contributing

Feel free to open issues or pull requests to help improve Hystorian!

---