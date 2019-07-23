![](https://github.com/MatthewReid854/reliability/blob/master/docs/logo3.png)

# reliability
*reliability* is a Python library for reliability engineering and survival analysis. It offers the ability to create and fit probability distributions intuitively and to explore and plot their properties. *reliability* is designed to be much easier to use than scipy.stats  whilst also extending the functionality to include many of the same tools that are typically only found in proprietary software such as Minitab, Reliasoft, and JMP Pro. It is somewhat similar to [lifelines](https://github.com/CamDavidsonPilon/lifelines/blob/master/README.md) but with a greater focus on the application of survival analysis to reliability engineering.

## Key features
- Ability to fit probability distributions to data including left or right censored data
- Ability to fit Weibull mixture models
- Calculating the probability of failure for stress-strength interference between any combination of the supported distributions
- Support for Exponential, Weibull, Gamma, Normal, Lognormal, and Beta probability distributions
- Mean residual life, quantiles, descriptive statistics summaries, random sampling from distributions
- Plots of PDF, CDF, survival function, hazard function, and cumulative hazard function
- Easy creation of distribution objects. Eg. dist = Weibull_Distribution(alpha=4,beta=2)
- Non-parametric estimation of reliability function using Kaplan-Meier
- Q-Q plots and goodness of fit tests (AICc, BIC)
- Reliability growth, optimal replacement time, sequential sampling charts, and many more functions.

## Installation
```
pip install reliability
```
## Documentation
Check out [readthedocs](https://reliability.readthedocs.io/en/latest/) for detailed documentation and examples.
