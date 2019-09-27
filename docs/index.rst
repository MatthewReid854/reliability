.. image:: images/logo.png
-------------------------------------

reliability
=====================================

*reliability* is a Python library for reliability engineering and survival analysis. It offers the ability to create and fit probability distributions intuitively and to explore and plot their properties. *reliability* is designed to be much easier to use than scipy.stats  whilst also extending the functionality to include many of the same tools that are typically only found in proprietary software such as Minitab, Reliasoft, and JMP Pro.

Contents:
============

.. toctree::
  :maxdepth: 1
  :caption: Quickstart & Intro

  Quickstart for reliability
  Introduction to the field of reliability engineering
  Recommended resources

.. toctree::
  :maxdepth: 1
  :caption: Parametric Models

  Equations of supported distributions
  Creating and plotting distributions
  Fitting a specific distribution to data
  Fitting all available distributions to data
  Weibull mixture models

.. toctree::
  :maxdepth: 1
  :caption: Probability Plotting

  Probability plots
  Quantile-Quantile plots
  Probability-Probability plots

.. toctree::
  :maxdepth: 1
  :caption: Non-parametric models

  Kaplan-Meier estimate of reliability
  Nelson-Aalen estimate of reliability

.. toctree::
  :maxdepth: 1
  :caption: Stress-Strength Interference
  
  Stress-Strength interference for any distributions
  Stress-Strength interference for normal distributions

.. toctree::
  :maxdepth: 1
  :caption: Repairable systems
  
  Reliability growth
  Optimal replacement time
  ROCOF

.. toctree::
  :maxdepth: 1
  :caption: Other functions
  
  One sample proportion
  Two proportion test
  Sample size required for no failures
  Sequential sampling chart
  Convert dataframe to grouped lists

.. toctree::
  :maxdepth: 1
  :caption: Physics of failure

  SN diagram
  Stress strain life diagram
  Creep
  Wear
  Acceleration factor
  Simultaneous equation solvers

.. toctree::
  :maxdepth: 1
  :caption: Administration
  
  Citing reliability in your work
  How to request or contribute a new feature
  How to get help
  About the author
  Credits to those who helped

Installation
------------------------------

.. code-block:: console

    pip install reliability

Source code and issue tracker
------------------------------

Available on Github, `MatthewReid854/reliability <https://github.com/MatthewReid854/reliability/>`_.
If you find any errors, bugs, or would like to request any feature extensions, please raise an `issue <https://github.com/MatthewReid854/reliability/issues/>`_ on GitHub
