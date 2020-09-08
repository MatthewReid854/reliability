![](https://github.com/MatthewReid854/reliability/blob/master/docs/images/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/reliability?color=brightgreen&logo=Python&logoColor=white&label=PyPI%20package)](https://pypi.org/project/reliability/)
[![Documentation Status](https://img.shields.io/readthedocs/reliability/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=latest)](http://reliability.readthedocs.io/?badge=latest)
[![Build Status](https://img.shields.io/travis/MatthewReid854/reliability/master?logo=travis%20ci&logoColor=white&label=Travis%20CI)](https://travis-ci.com/github/MatthewReid854/reliability)
[![Scc Count Badge](https://sloc.xyz/github/MatthewReid854/reliability/?category=code)](https://github.com/MatthewReid854/reliability/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MatthewReid854/reliability.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MatthewReid854/reliability/context:python)
[![Downloads](https://img.shields.io/pypi/dm/reliability?logo=Docusign&logoColor=white&label=PyPI%20downloads)](https://pypistats.org/packages/reliability)
[![LGPLv3 license](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.txt)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatthewReid854/reliability/blob/master/notebooks/Intro2.ipynb)
[![DOI](https://zenodo.org/badge/198305660.svg)](https://zenodo.org/badge/latestdoi/198305660)
[![Donate](https://img.shields.io/badge/Donate-$%20€%20¥%20£-blueviolet.svg?logo=PayPal&logoColor=white)](https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html)

[![test](https://img.shields.io/badge/test-blueviolet.svg?logo=data:image/svg;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iMTQyLjAwMDAwMHB0IiBoZWlnaHQ9IjQ5LjAwMDAwMHB0IiB2aWV3Qm94PSIwIDAgMTQyLjAwMDAwMCA0OS4wMDAwMDAiCiBwcmVzZXJ2ZUFzcGVjdFJhdGlvPSJ4TWlkWU1pZCBtZWV0Ij4KCjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLDQ5LjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTEzMTAgNDgwIGMtMTQgLTQgLTEyMyAtOCAtMjQ0IC05IGwtMjE5IC0xIDcgLTU2IGM0IC0zMiAxNCAtODYgMjMKLTEyMSAxMiAtNTAgMTMgLTcwIDUgLTkxIC0xMCAtMjIgLTEwIC0yNCAxIC04IDU3IDg3IDE0OSAxNjkgMjQ4IDIyMSA4NiA0NQoxMDIgNDUgMjMgMCAtMTU5IC05MCAtMjY3IC0yMzcgLTIwOSAtMjg1IDQxIC0zNCAyMzEgMjcgMjk0IDk0IDMxIDMzIDIyIDQyCi0yNiAyOCAtOTEgLTI2IC05MSAtMjYgLTU4IDAgMTcgMTMgNTMgNDUgODAgNzMgNTAgNDkgNTAgNTAgMjMgNTMgLTE0IDIgLTUwCi05IC03OCAtMjMgLTI5IC0xNSAtNTUgLTI0IC01OCAtMjEgLTMgNCAxNSAyNiA0MCA0OSAxNDQgMTM4IDMzNyAxMTYgMTk5IC0yMgotMzQgLTM0IC0zNyAtNDAgLTIwIC00NCA3MCAtMTkgLTYwIC0xNjkgLTIxNCAtMjQ3IC00NSAtMjMgLTcxIC0zOCAtNTcgLTM1Cjc4IDIxIDIyNiAxMjggMjcxIDE5NCAzMSA0NyAzNiA3MCAxOCA4OCAtOCA4IC0xIDI0IDI1IDU4IDQyIDU0IDQ1IDc4IDE0IDk5Ci0yNCAxNyAtNDggMTkgLTg4IDZ6Ii8+CjxwYXRoIGQ9Ik0xMjAgNDY4IGMwIC0yIC0yMiAtMTAwIC00OSAtMjE4IGwtNDggLTIxNSA0MzYgLTcgYzI0MCAtMyA0NjEgLTgKNDkxIC0xMSBsNTUgLTUgLTUwIDcgYy0yNyA0IC01OCAxMSAtNjcgMTUgLTIxIDEwIC0zOCA1NyAtMzEgODggNCAxOCAyIDIwIC01CjkgLTkgLTEyIC0xMiAtOSAtMTkgMTUgLTEwIDM4IC0xMCAyNTAgMCAyOTIgbDkgMzIgLTM2MSAwIGMtMTk5IDAgLTM2MSAtMQotMzYxIC0yeiBtNTAgLTY1IGMwIC0xMCAtOSAtNTUgLTIwIC0xMDEgLTI4IC0xMTggLTMxIC0xMTIgNDYgLTExMiA0OCAwIDY1Ci0zIDYyIC0xMiAtNSAtMTQgLTE3OCAtMjUgLTE3OCAtMTEgMCA0IDEyIDYzIDI3IDEzMSAyMiAxMDEgMzAgMTIyIDQ1IDEyMiAxMAowIDE4IC03IDE4IC0xN3ogbTMxOSA4IGMyMiAtMTQgNiAtODEgLTE5IC04MSAtMTYgMCAtMjAgNyAtMjAgMzEgMCAyOSAtMSAzMAotNDIgMjcgbC00MyAtMyAtMjIgLTk1IGMtMTIgLTUyIC0yMiAtOTggLTIzIC0xMDIgMCAtNSAyMCAtOCA0NCAtOCA0MiAwIDQ0IDEKNTQgMzkgOSAzNiA4IDQwIC0xMCA0MyAtMzYgNiAtMTggMjMgMjcgMjYgNDUgMyA0NiAzIDM5IC0yNSAtMTggLTgxIC0yNiAtMTAxCi00MCAtMTA3IC0zMCAtMTEgLTE0MSAtNiAtMTUyIDcgLTE0IDE3IDM0IDIyNyA1NiAyNDUgMTggMTQgMTMwIDE2IDE1MSAzegptMjAwIDAgYzExIC03IDEyIC0yMCAxIC03NSAtNyAtMzYgLTE3IC02OSAtMjMgLTczIC02IC0zIC0zOCAtOSAtNzAgLTEyIGwtNjAKLTYgLTExIC00NSBjLTkgLTM1IC0xNiAtNDUgLTMxIC00NSAtMTIgMCAtMTkgNiAtMTggMTUgMiAxNyA0MiAyMDcgNDkgMjMzIDUKMTQgMTcgMTcgNzcgMTcgNDAgMCA3OCAtNCA4NiAtOXogbTcxIC0xMSBjMCAtMTAgLTkgLTU4IC0yMCAtMTA1IC0yNyAtMTE2Ci0yNyAtMTEzIDIzIC0xMTcgNjYgLTUgNDkgLTIzIC0yNSAtMjYgLTYxIC0zIC02OCAtMSAtNjggMTYgMCAyNSA0NiAyMjcgNTQKMjQxIDEyIDE4IDM2IDEyIDM2IC05eiBtLTU2OCAtMjgxIGMyIC01IDI3IC0xMCA1NiAtMTEgNDkgMCA1MyAtMiA1MCAtMjEgLTIKLTE4IC0xMCAtMjIgLTQzIC0yMyAtMjYgLTEgLTQ0IDQgLTUyIDE1IC0xMSAxNSAtMTIgMTQgLTEzIC0xIDAgLTIyIC0yMCAtMjQKLTIwIC0zIDAgMjAgLTE2IDE5IC0yNCAtMSAtMTAgLTI3IC0xOCAtOCAtMTEgMjUgNSAyNCAxMiAzMSAzMCAzMSAxMyAwIDI1IC01CjI3IC0xMXogbTE5MiAtMTIgYy0yIC0xNCAxIC0xNSAxMiAtNiA3IDYgMTkgOSAyNiA3IDcgLTMgMTggMiAyNCAxMCA5IDEyIDE1CjEyIDM0IDIgMTQgLTcgNTYgLTEyIDEwNiAtMTEgODggMiAxMDkgLTggODAgLTM3IC0xNiAtMTYgLTE4IC0xNiAtMzAgMSAtMTAKMTMgLTE1IDE0IC0xOSA1IC02IC0xNiAtNjcgLTIzIC02NyAtOCAwIDcgLTUgOCAtMTIgMyAtMjUgLTE1IC00OCAtMTMgLTQ5IDUKLTEgMTYgLTIgMTUgLTYgLTEgLTQgLTE0IC04IC0xNiAtMTkgLTYgLTggNiAtMTQgOCAtMTQgMyAwIC01IC0xNCAtMTAgLTMwCi0xMSAtMTcgLTEgLTMwIDIgLTMwIDcgMCA2IC05IDUgLTIwIC0yIC0yOCAtMTggLTQ0IDAgLTMwIDM2IDExIDMxIDQ2IDMzIDQ0CjN6Ii8+CjxwYXRoIGQ9Ik01NjEgMzUzIGMtMTkgLTY2IC0xNiAtNzMgMjggLTczIDQ2IDAgNDcgMCA1NiA2MyBsNyA0NyAtNDAgMCBjLTM5CjAgLTQyIC0yIC01MSAtMzd6Ii8+CjxwYXRoIGQ9Ik0xMDE4IDIzIGM3IC0zIDE2IC0yIDE5IDEgNCAzIC0yIDYgLTEzIDUgLTExIDAgLTE0IC0zIC02IC02eiIvPgo8L2c+Cjwvc3ZnPgo=&logoColor=white)](https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html)



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
If you find any errors, have any suggestions, or would like to request that something be added, please email me: alpha.reliability@gmail.com.
