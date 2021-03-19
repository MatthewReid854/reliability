.. image:: https://raw.githubusercontent.com/MatthewReid854/reliability/master/docs/images/logo.png

-------------------------------------

Gamma_Distribution
''''''''''''''''''

Creates a Gamma Distribution object.

Inputs:

-    alpha - scale parameter
-    beta - shape parameter
-    gamma - threshold (offset) parameter. Default = 0

Methods:
    
-    name - 'Gamma'
-    name2 = 'Gamma_2P' or 'Gamma_3P' depending on the value of the gamma parameter
-    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Gamma Distribution (α=5,β=2)'
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
-    b5 - The time where 5% have failed. Same as quantile(0.05)
-    b95 - The time where 95% have failed. Same as quantile(0.95)
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


.CDF()
""""""

Plots the CDF (cumulative distribution function)
      
Inputs:

- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).
  

.SF()
"""""

Plots the SF (survival function). Also known as the reliability function.
      
Inputs:

- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.HF()
"""""

Plots the HF (hazard function function)
      
Inputs:

- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.CHF()
""""""

Plots the CHF (cumulative hazard function)
      
Inputs:

- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 200 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.inverse_SF()
"""""""""""""

Inverse Survival function calculator

Inputs:

- q - quantile to be calculated

Outputs:

- the inverse of the survival function at q


.mean_residual_life()
"""""""""""""""""""""

Mean Residual Life calculator
    
Inputs:

- t - time at which MRL is to be evaluated

Outputs:

- the mean residual life at t


.quantile()
"""""""""""

Quantile calculator

Inputs:

- q - quantile to be calculated

Outputs:

- the probability (area under the curve) that a random variable from the distribution is < q


.random_samples()
"""""""""""""""""

Draws random samples from the probability distribution.

Inputs:

- number_of_samples - the number of samples to be drawn
- seed - the random seed. Default is None

Outputs:

- list of the random samples


.stats()
""""""""

Descriptive statistics of the probability distribution. Same as the statistics shown using .plot() but printed to console.

Inputs:

- None

Outputs:

- None
- The descriptive statistics (mean, median, etc.) will be printed to the console.
