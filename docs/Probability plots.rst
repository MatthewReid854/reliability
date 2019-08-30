.. image:: images/logo.png

-------------------------------------

Probability plots
'''''''''''''''''

Proabability plots are a general term for several different plotting techniques. One of these is a graphical technique for comparing two data sets and includes probability-probability (PP) plots and quantile-quantile (QQ) plots. PP and QQ plots are discussed in ___link___. The second technique is for assessing the goodness of fit of a distribution by plotting the enpirical CDF of the failures against their failure time and scaling the axes in such as way that the distribution appears linear. This method was the original "least squares" fitting method used before computers were capable of calculating the MLE estimates of the parameters. While we do not typically favour the use of least squares as a fitting method, we can still use probability plots to assess the goodness of fit.
The module ``reliability.Probability_plotting`` contains functions for each of the six distributions supported in ``reliability``. These functions are:

- Weibull_probability_plot
- Normal_probability_plot
- Lognormal_probability_plot
- Gamma_probability_plot
- Beta_probability_plot
- Exponential_probability_plot

There is also a function to obtain the plotting positions as well as the functions for custom axes scaling. These are explained more in the help file for ``reliability.Probability_plotting`` and will not be discussed further here.

Within each of the above probability plotting functions you may enter failures as well as left or right censored data (either but not both). For those distributions that have a function in ``reliability.Fitters`` for fitting location shifted distributions (Weibull_3P, Gamma_3P, Exponential_2P), you can explicitly tell the probability plotting function to fit the gamma parameter using fit_gamma=True. By default the gamma parameter is not fitted. Fitting the gamma parameter will also change the x-axis to time-gamma such that everything will appear linear.

Inputs:

- failures - the array or list of failure times
- right_censored - the array or list of right censored failure times
- left_censored - the array or list of left censored failure times
- fit_gamma - this is only included for Weibull, Gamma, and Exponential probability plots. Specify fit_gamma=True to fit the location shifted distribution.

Outputs:

- The plot is the only output. Use plt.show() to show it.

In the example below we generate some samples from a Normal Distribution and provide these to the probability plotting function.

.. code:: python

    from reliability.Distributions import Normal_Distribution
    from reliability.Probability_plotting import Normal_probability_plot
    import matplotlib.pyplot as plt
    dist = Normal_Distribution(mu=50,sigma=10)
    dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
    failures = dist.random_samples(100)
    Normal_probability_plot(failures=failures) #generates the probability plot
    plt.show()
    
.. image:: images/Normal_probability_plot.png

