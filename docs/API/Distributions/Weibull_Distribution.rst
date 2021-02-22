.. image:: https://raw.githubusercontent.com/MatthewReid854/reliability/master/docs/images/logo.png

-------------------------------------

Weibull_Distribution
''''''''''''''''''''

Creates a Weibull Distribution object.

Inputs:

-    alpha - scale parameter
-    beta - shape parameter
-    gamma - threshold (offset) parameter. Default = 0

Methods:
    
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

.plot()
"""""""

Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics in a single figure

Inputs:

-   xvals - x-values for plotting
-   xmin - minimum x-value for plotting
-   xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. No plotting keywords are accepted.

Outputs:

-   The plot will be shown. No need to use plt.show()

.PDF()
""""""

Plots the PDF (probability density function)

Inputs:

-   show_plot - True/False. Default is True
-   xvals - x-values for plotting
-   xmin - minimum x-value for plotting
-   xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

-   yvals - this is the y-values of the plot
-   The plot will be shown if show_plot is True (which it is by default).


This is a work in progress and will be written soon
