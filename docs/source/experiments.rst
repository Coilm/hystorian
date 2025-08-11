Experiments
===========

Tools and classes for managing and analyzing experiment data.

.. automodule:: hystorian.experiments.cypher
    :members:

Example: Analyze an experiment
------------------------------
.. code-block:: python

    from hystorian.experiments.cypher import CypherExperiment
    exp = CypherExperiment('experiment_data.hdf5')
    exp.analyze()