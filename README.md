![](https://github.com/MatthewReid854/reliability/blob/master/docs/images/logo.png)

[![PyPI version](https://badge.fury.io/py/reliability.svg)](https://badge.fury.io/py/reliability)
[![Documentation Status](https://readthedocs.org/projects/reliability/badge/?version=latest)](http://reliability.readthedocs.io/?badge=latest)
[![Build Status](https://travis-ci.com/MatthewReid854/reliability.svg?branch=master)](https://travis-ci.com/github/MatthewReid854/reliability)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MatthewReid854/reliability.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MatthewReid854/reliability/context:python)
[![Downloads](https://pepy.tech/badge/reliability)](https://pepy.tech/project/reliability)
[![LGPLv3 license](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.txt)
[![Lines Of Code](https://tokei.rs/b1/github/MatthewReid854/reliability?category=code)](https://github.com/MatthewReid854/reliability)
[![DOI](https://zenodo.org/badge/198305660.svg)](https://zenodo.org/badge/latestdoi/198305660)
[![Donate](https://img.shields.io/badge/donate-$€¥£-blueviolet.svg)](https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatthewReid854/reliability/blob/master/notebooks/Intro3.ipynb)

*reliability* is a Python library for reliability engineering and survival analysis. It significantly extends the functionality of scipy.stats and also includes many specialist tools that are otherwise only available in proprietary software.

![](https://github.com/MatthewReid854/reliability/blob/master/docs/images/readme_image_V2.png)

## Key features
- Fitting probability distributions to data including right censored data
- Fitting Weibull mixture models
- Calculating the probability of failure for stress-strength interference between any combination of the supported distributions
- Support for Exponential, Weibull, Gamma, Normal, Lognormal, Loglogistic, and Beta probability distributions
- Mean residual life, quantiles, descriptive statistics summaries, random sampling from distributions
- Plots of probability density function (PDF), cumulative distribution function (CDF), survival function (SF), hazard function (HF), and cumulative hazard function (CHF)
- Easy creation of distribution objects. Eg. dist = Weibull_Distribution(alpha=4,beta=2)
- Non-parametric estimation of survival function using Kaplan-Meier and Nelson-Aalen
- Goodness of fit tests (AICc, BIC)
- Probability plots on probability paper for all supported distributions
- Quantile-Quantile plots and Probability-Probability plots
- Reliability growth, optimal replacement time, sequential sampling charts, similar distributions
- Physics of Failure (SN diagram, stress-strain, fracture mechanics, creep)
- Accelerated Life Testing probability plots (Weibull, Exponential, Normal, Lognormal)
- Accelerated Life Testing Models (Exponential, Eyring, Power, Dual-Exponential, Power-Exponential).
- Mean cumulative function for repairable systems

## Installation
```
pip install reliability
```
## Documentation
Check out [readthedocs](https://reliability.readthedocs.io/en/latest/) for detailed documentation and examples.
If you find any errors, have any suggestions, or would like to request that something be added, please email me: m.reid854@gmail.com.
