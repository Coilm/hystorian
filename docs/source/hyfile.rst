HyFile
======

HyFile is the main interface for reading, writing, and processing HDF5 files in Hystorian.

.. automodule:: hystorian.io.hyFile
    :members:
    :undoc-members:

Example: Extract and process data
---------------------------------
.. code-block:: python

    from hystorian.io.hyFile import HyFile
    with HyFile('data.hdf5', 'r+') as f:
        f.extract_data('/path/to/file.ibw')
        # Apply a function to a dataset
        import numpy as np
        from hystorian.io.hyFile import HyPath
        f.apply(np.mean, HyPath('datasets/data/grid'), output_names='grid_mean')

    with HyFile('random.hdf5', 'r+') as f:
        f.apply(np.sum, HyPath('datasets/data/grid'), output_names='grid_sum')
        f.apply(np.sum, HyPath('datasets/data/grid'), output_names='grid_sum', axis=0)
        f.apply(np.sum, [HyPath('datasets/data/grid'), HyPath('datasets/data/grid2')], output_names='grid_sum')
        f.multiple_apply(np.sum, [HyPath('datasets/data/grid'), HyPath('datasets/data/grid2')], output_names=['grid_sum', 'grid_sum2'])
