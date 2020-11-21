![Logo](https://raw.githubusercontent.com/MatthewReid854/reliability/master/docs/images/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/reliability?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/reliability/)
[![Documentation Status](https://img.shields.io/readthedocs/reliability/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=latest)](http://reliability.readthedocs.io/?badge=latest)
[![Build Status](https://img.shields.io/travis/MatthewReid854/reliability/master?logo=travis&label=Travis%20CI)](https://travis-ci.com/github/MatthewReid854/reliability)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MatthewReid854/reliability.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MatthewReid854/reliability/context:python)
[![Scc Count Badge](https://sloc.xyz/github/MatthewReid854/reliability/?category=code)](https://github.com/MatthewReid854/reliability/)
[![Downloads](https://img.shields.io/pypi/dm/reliability?color=brightgreen&logo=Docusign&logoColor=white&label=PyPI%20downloads)](https://pypistats.org/packages/reliability)
[![LGPLv3 license](https://img.shields.io/badge/License-LGPLv3-blue.svg?logo=GNU&logoColor=white)](https://www.gnu.org/licenses/lgpl-3.0.txt)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3937999-blue.svg?logo=Buffer&logoColor=white)](https://doi.org/10.5281/zenodo.3937999)
[![Donate](https://img.shields.io/badge/Donate-$$$-blueviolet.svg?logo=paypal)](https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html)
[![survey](https://img.shields.io/badge/Provide-feedback-9cf.svg?logo=Verizon)](https://form.jotform.com/203156856636058)
[![survey](https://img.shields.io/badge/feedback-9cf.svg?logo=Verizon)](https://form.jotform.com/203156856636058)
[![survey](https://img.shields.io/badge/-feedback-9cf.svg?logo=Verizon)](https://form.jotform.com/203156856636058)
[![survey](https://img.shields.io/badge/ -feedback-9cf.svg?logo=Verizon)](https://form.jotform.com/203156856636058)

*reliability* is a Python library for [reliability engineering](https://en.wikipedia.org/wiki/Reliability_engineering) and [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis). It significantly extends the functionality of scipy.stats and also includes many specialist tools that are otherwise only available in proprietary software.

![](https://raw.githubusercontent.com/MatthewReid854/reliability/master/docs/images/readme_image_V2.png)

## Key features
- Fitting probability distributions to data including right censored data
- Fitting Weibull mixture models and Weibull Competing risks models
- Calculating the probability of failure for stress-strength interference between any combination of the supported distributions
- Support for Exponential, Weibull, Gamma, Gumbel, Normal, Lognormal, Loglogistic, and Beta probability distributions
- Mean residual life, quantiles, descriptive statistics summaries, random sampling from distributions
- Plots of probability density function (PDF), cumulative distribution function (CDF), survival function (SF), hazard function (HF), and cumulative hazard function (CHF)
- Easy creation of distribution objects. Eg. dist = Weibull_Distribution(alpha=4,beta=2)
- Non-parametric estimation of survival function using Kaplan-Meier, Nelson-Aalen, and Rank Adjustment
- Goodness of fit tests (AICc, BIC, AD)
- Probability plots on probability paper for all supported distributions
- Quantile-Quantile plots and Probability-Probability plots
- Reliability growth, optimal replacement time, sequential sampling charts, similar distributions, reliability test planners
- Interactive matplotlib functions including crosshairs and distribution explorer
- Physics of Failure (SN diagram, stress-strain, fracture mechanics, creep)
- Accelerated Life Testing probability plots (Weibull, Exponential, Normal, Lognormal)
- Accelerated Life Testing Models (Exponential, Eyring, Power, Dual-Exponential, Power-Exponential).
- Mean cumulative function and ROCOF for repairable systems

## Installation and upgrading

To install *reliability* for the first time, open your command prompt and type:

```
pip install reliability
```

To upgrade a previous installation of *reliability* to the most recent version, open your command prompt and type:

```
pip install --upgrade reliability
```

If you would like to receive an email notification when a new release of *reliability* is uploaded to PyPI, [NewReleases.io](https://newreleases.io/) provides this service for free.

## Documentation
Check out [readthedocs](https://reliability.readthedocs.io/en/latest/) for detailed documentation and examples.
If you find any errors, have any suggestions, or would like to request that something be added, please email alpha.reliability@gmail.com.
