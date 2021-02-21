.. image:: images/logo.png

-------------------------------------

Distributions2
''''''''''''''

Weibull_Distribution
--------------------

Creates a Weibull Distribution object.

inputs:

-    alpha - scale parameter
-    beta - shape parameter
-    gamma - threshold (offset) parameter. Default = 0

methods:
    
-    name - 'Weibull'
-    name2 = 'Weibull_2P' or 'Weibull_3P' depending on the value of the gamma parameter
-    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Weibull Distribution (α=5,β=2)'
-    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
-    parameters - [alpha,beta,gamma]
-    alpha
-    beta
-    gamma
-    mean
-    variance
-    standard_deviation
-    skewness
-    kurtosis
-    excess_kurtosis
-    median
-    mode
-    b5
-    b95
-    plot() - plots all functions (PDF,CDF,SF,HF,CHF)
-    PDF() - plots the probability density function
-    CDF() - plots the cumulative distribution function
-    SF() - plots the survival function (also known as reliability function)
-    HF() - plots the hazard function
-    CHF() - plots the cumulative hazard function
-    quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing. Also known as b life where b5 is the time at which 5% have failed.
-    inverse_SF() - the inverse of the Survival Function. This is useful when producing QQ plots.
-    mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time. Effectively the mean of the remaining amount (right side) of a distribution at a given time.
-    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
-    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.

Exponential_Distribution
--------------------

Creates an Exponential Distribution object.