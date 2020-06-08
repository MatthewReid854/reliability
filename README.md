![](https://github.com/MatthewReid854/reliability/blob/master/docs/images/logo.png)

[![PyPI version](https://badge.fury.io/py/reliability.svg)](https://badge.fury.io/py/reliability)
[![Documentation Status](https://readthedocs.org/projects/reliability/badge/?version=latest)](http://reliability.readthedocs.io/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MatthewReid854/reliability.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MatthewReid854/reliability/context:python)
[![Downloads](https://pepy.tech/badge/reliability)](https://pepy.tech/project/reliability)
[![LGPLv3 license](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.txt)
[![HitCount](http://hits.dwyl.com/MatthewReid854/reliability.svg)](http://hits.dwyl.com/MatthewReid854/reliability)

# reliability
*reliability* is a Python library for reliability engineering and survival analysis. It offers the ability to create and fit probability distributions intuitively and to explore and plot their properties. *reliability* is designed to be much easier to use than scipy.stats whilst also extending the functionality to include many of the same tools that are typically only found in proprietary software such as Minitab, Reliasoft, and JMP Pro.

## Key features
- Fitting probability distributions to data including right censored data
- Fitting Weibull mixture models
- Calculating the probability of failure for stress-strength interference between any combination of the supported distributions
- Support for Exponential, Weibull, Gamma, Normal, Lognormal, and Beta probability distributions
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
