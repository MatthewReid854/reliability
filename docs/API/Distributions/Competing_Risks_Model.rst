.. image:: https://raw.githubusercontent.com/MatthewReid854/reliability/master/docs/images/logo.png

-------------------------------------

Competing_Risks_Model
'''''''''''''''''''''

Creates a Mixture Model probability distribution object.

The competing risks model is used to model the effect of multiple risks (expressed as probability distributions) that act on a system over time.
The model is obtained using the product of the survival functions: SF_total = SF_1 x SF_2 x SF_3 x ....x SF_n
An equivalent form of this model is to sum the hazard or cumulative hazard functions. The result is the same.
In this way, we see the CDF, HF, and CHF of the overall model being equal to or higher than any of the constituent distributions.
Similarly, the SF of the overall model will always be equal to or lower than any of the constituent distributions.
The PDF occurs earlier in time since the earlier risks cause the population to fail sooner leaving less to fail due to the later risks.

This model should be used when a data set has been divided by failure mode and each failure mode has been modelled separately.
The competing risks model can then be used to recombine the constituent distributions into a single model.
Unlike the mixture model, there are no proportions as the risks are competing to cause failure rather than being mixed.

As this process is multiplicative for the survival function, and may accept many distributions of different types, the mathematical formulation quickly gets complex.
For this reason, the algorithm combines the models numerically rather than empirically so there are no simple formulas for many of the descriptive statistics (mean, median, etc.)
Also, the accuracy of the model is dependent on xvals. If the xvals array is small (<100 values) then the answer will be "blocky" and inaccurate.
The variable xvals is only accepted for PDF, CDF, SF, HF, CHF. The other methods (like random samples) use the default xvals for maximum accuracy.
The default number of values generated when xvals is not given is 1000. Consider this carefully when specifying xvals in order to avoid inaccuracies in the results.

The API is similar to the other probability distributions (Weibull, Normal, etc.) and has the following Inputs and Methods:

Inputs:

- distributions - a list or array of probability distributions used to construct the model

Methods:

-    name - 'Competing risks'
-    name2 - 'Competing risks using 3 distributions'
-    mean
-    median
-    mode
-    variance
-    standard_deviation
-    skewness
-    kurtosis
-    excess_kurtosis
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

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. No plotting keywords are accepted.

Outputs:

-   The plot will be shown. No need to use plt.show()

.PDF()
""""""

Plots the PDF (probability density function)

Inputs:

-   plot_components - option to plot the components of the model. Default is False.
-   show_plot - True/False. Default is True
-   xvals - x-values for plotting
-   xmin - minimum x-value for plotting
-   xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

-   yvals - this is the y-values of the plot
-   The plot will be shown if show_plot is True (which it is by default).


.CDF()
""""""

Plots the CDF (cumulative distribution function)
      
Inputs:

- plot_components - option to plot the components of the model. Default is False.
- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.SF()
"""""

Plots the SF (survival function). Also known as the reliability function.
      
Inputs:

- plot_components - option to plot the components of the model. Default is False.
- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.HF()
"""""

Plots the HF (hazard function function)
      
Inputs:

- plot_components - option to plot the components of the model. Default is False.
- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

Outputs:

- yvals - this is the y-values of the plot
- The plot will be shown if show_plot is True (which it is by default).


.CHF()
""""""

Plots the CHF (cumulative hazard function)
      
Inputs:

- plot_components - option to plot the components of the model. Default is False.
- show_plot - True/False. Default is True
- xvals - x-values for plotting
- xmin - minimum x-value for plotting
- xmax - maximum x-value for plotting.

If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 1000 elements will be created using these ranges. If nothing is specified then the range will be based on the distribution's parameters. Plotting keywords are also accepted (eg. color, linestyle)

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
