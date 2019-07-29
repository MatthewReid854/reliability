.. _code_directive:

.. image:: images/logo.png

-------------------------------------


Creating and plotting distributions
'''''''''''''''''''''''''''''''''''

Probability distributions within ``reliability`` are Python objects, which allows us to specify just the type of distribution and parameters, and from that we can access a large number of methods, some of which will require additional input. There are 6 different probability distributions available in ``reliability``. These are:

-   Weibull Distribution
-   Exponential Distribution
-   Gamma Distribution
-   Normal Distribution
-   Lognormal Distribution
-   Beta Distribution

Understanding how to create and plot distributions is easiest with an example. The following code will create a Lognormal Distribution with parameters mu=5 and sigma=1. From this distribution, we will use the plot() method which provides a quick way to visualise the five functions and also provides a summary of the descriptive statistics.

.. code:: python

    from reliability.Distributions import Lognormal_Distribution
    dist = Lognormal_Distribution(mu=5,sigma=1)
    dist.plot()

.. image:: images/Lognormal_plot1.png

The following methods are available for all distributions:

-   parameter names - varies by distribution
-   parameters - returns an array of parameters
-   mean
-   variance
-   standard_deviation
-   skewness
-   kurtosis
-   excess_kurtosis
-   median
-   mode
-   plot() - plots all functions (PDF,CDF,SF,HF,CHF)
-   PDF() - plots the probability density function
-   CDF() - plots the cumulative distribution function
-   SF() - plots the survival function (also known as reliability function)
-   HF() - plots the hazard function
-   CHF() - plots the cumulative hazard function
-   quantile() - Calculates the quantile (time until a fraction has failed) for a given fraction failing. Also known as 'b' life where b5 is the time at which 5% have failed.
-   mean_residual_life() - Average residual lifetime of an item given that the item has survived up to a given time. Effectively the mean of the remaining amount (right side) of a distribution at a given time.
-   stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
-   random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.


Section title
-------------

