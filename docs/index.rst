.. image:: images/logo.png

-------------------------------------

|pypi| |docs| |build| |code quality| |LOC| |downloads| |license| |colab| |DOI| |donate|

*reliability* is a Python library for reliability engineering and survival analysis. It significantly extends the functionality of scipy.stats and also includes many specialist tools that are otherwise only available in proprietary software.

.. image:: images/readme_image_V2.png

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
  Mixture models
  Competing risk models

.. toctree::
  :maxdepth: 1
  :caption: Probability Plotting

  Probability plots
  Quantile-Quantile plots
  Probability-Probability plots

.. toctree::
  :maxdepth: 1
  :caption: Non-parametric models

  Kaplan-Meier
  Nelson-Aalen
  Rank Adjustment

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
  Mean cumulative function

.. toctree::
  :maxdepth: 1
  :caption: Physics of failure

  SN diagram
  Stress-strain and strain-life
  Fracture mechanics
  Creep
  Palmgren-Miner linear damage model
  Acceleration factor
  Solving simultaneous equations with sympy

.. toctree::
  :maxdepth: 1
  :caption: Accelerated Life Testing

  ALT probability plots
  Fitting a model to ALT data

.. toctree::
  :maxdepth: 1
  :caption: Other functions and data
  
  Similar Distributions
  Convert dataframe to grouped lists
  Crosshairs
  Distribution explorer
  Histogram
  Make right censored data
  Datasets

.. toctree::
  :maxdepth: 1
  :caption: Reliability testing
  
  One sample proportion
  Two proportion test
  Sample size required for no failures
  Sequential sampling chart
  Reliability test planner
  Reliability test duration
  Chi-squared test
  Kolmogorov-Smirnov test

.. toctree::
  :maxdepth: 1
  :caption: Administration
  
  Changelog
  Development roadmap
  Citing reliability in your work
  How to request or contribute a new feature
  How to get help
  How to donate to the project
  About the author
  Credits to those who helped

.. |pypi| image:: https://img.shields.io/pypi/v/reliability?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package
    :alt: PyPI package
    :scale: 100%
    :target: https://pypi.org/project/reliability/

.. |docs| image:: https://img.shields.io/readthedocs/reliability/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=latest
    :alt: docs
    :scale: 100%
    :target: https://reliability.readthedocs.io/en/latest/

.. |build| image:: https://img.shields.io/travis/MatthewReid854/reliability/master?logo=travis&label=Travis%20CI
    :alt: build
    :scale: 100%
    :target: https://travis-ci.com/github/MatthewReid854/reliability

.. |code quality| image:: https://img.shields.io/lgtm/grade/python/g/MatthewReid854/reliability.svg?logo=lgtm&logoWidth=18
    :alt: code quality: python
    :scale: 100%
    :target: https://lgtm.com/projects/g/MatthewReid854/reliability/context:python

.. |LOC| image:: https://sloc.xyz/github/MatthewReid854/reliability/?category=code
    :alt: lines of code
    :scale: 100%
    :target: https://github.com/MatthewReid854/reliability

.. |downloads| image:: https://img.shields.io/pypi/dm/reliability?color=brightgreen&logo=Docusign&logoColor=white&label=PyPI%20downloads
    :alt: downloads
    :scale: 100%
    :target: https://pypistats.org/packages/reliability

.. |license| image:: https://img.shields.io/badge/License-LGPLv3-blue.svg?logo=GNU
    :alt: license
    :scale: 100%
    :target: https://www.gnu.org/licenses/lgpl-3.0.txt

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Colab
    :scale: 100%
    :target: https://colab.research.google.com/github/MatthewReid854/reliability/blob/master/notebooks/Intro2.ipynb

.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281/zenodo.3937999-blue.svg?logo=Buffer&logoColor=white
    :alt: DOI
    :scale: 100%
    :target: https://doi.org/10.5281/zenodo.3937999

.. |donate| image:: https://img.shields.io/badge/Donate-$$$-blueviolet.svg?logo=paypal
    :alt: donate
    :scale: 100%
    :target: https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html
