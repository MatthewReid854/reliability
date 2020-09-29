.. image:: images/logo.png

-------------------------------------

Chi-squared test
''''''''''''''''

The Chi-squared test is a statistical test for goodness of fit to determine whether we can accept or reject the hypothesis that the data is from the specified distribution at the specified level of significance. This method is not a means of comparing distributions (which can be done with AICc, BIC, and AD), but instead allows us to accept or reject a hypothesis that data come from a distribution. Note that the result is sensitive to the bins. For this reason, it is recommended to leave bins as the default value.

The procedure for the test involves comparing the fitted CDF (from a hypothesised distribution) against the empirical CDF (from a cumulative histogram of the data). As with all histograms, the exact shape of the histogram depends on the bins. The difference between the fitted CDF and the empirical CDF is used to find the Chi-squared statistic. The specified level of significance (analogous to confidence level), the number of parameters in the hypothesised distribution, and the number of data points is used to obtain the Chi-squared critical value from the chi-squared distribution. By comparing the chi-squared statistic with the chi-squared critical value, we can determine whether the hypothesis (that the data are from the specified distribution) should be rejected or accepted. The acceptance criteria is when the the chi-squared statistic is below the critical value.

Inputs:

-   distribution - a distribution object created using the reliability.Distributions module
-   data - an array or list of data that are hypothesised to come from the distribution
-   significance - This is the complement of confidence. 0.05 significance is the same as 95% confidence. Must be between 0 and 0.5. Default is 0.05.
-   bins - an array or list of the bin edges from which to group the data OR a string for the bin edge method from numpy. String options are 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'. For more information see the numpy documentation on `numpy.histogram_bin_edges <https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html>`_. Default is 'auto'.
-   print_results - if True the results will be printed. Default is True.
-   show_plot - if True a plot of the distribution and histogram will be shown. Default is True.

Outputs:

-   chisquared_statistic - the chi-squared statistic
-   chisquared_critical_value - the chi-squared critical value
-   hypothesis - 'ACCEPT' or 'REJECT'. If chisquared_statistic < chisquared_critical_value then we can accept the hypothesis that the data is from the specified distribution
-   bin_edges - the bin edges used. If bins is a list or array then bin_edges = bins. If bins is a string then you can find the bin_edges that were calculated using this output.

In the example below we import a dataset called mileage which contains 100 values that appear to be normally distributed. Using the function chi2test we can determine whether we should accept the hypothesis that the data are from a Normal distribution with parameters mu=30011 and sigma=10472. This example is based on Example 2.31 (page 63) of Reliability Engineering and Risk Analysis (listed in `recommended resources <https://reliability.readthedocs.io/en/latest/Recommended%20resources.html>`_).

.. code:: python

    from reliability.Datasets import mileage
    from reliability.Distributions import Normal_Distribution
    from reliability.Reliability_testing import chi2test
    import numpy as np

    data = mileage().failures
    dist = Normal_Distribution(mu=30011, sigma=10472)
    bins = [0, 13417, 18104, 22791, 27478, 32165, 36852, 41539, 46226, np.inf] #it is not necessary to specify the bins and leaving them unspecified is usually best
    chi2test(distribution=dist, data=data, bins=bins)
    
    '''
    Chi-squared statistic: 3.1294947845652
    Chi-squared critical value: 12.591587243743977
    At the 0.05 significance level, we can ACCEPT the hypothesis that the data comes from a Normal Distribution (μ=30011,σ=10472)
    '''

.. image:: images/chi2test.png
