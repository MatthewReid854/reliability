.. image:: images/logo.png

-------------------------------------

Fitting a specific distribution to data
'''''''''''''''''''''''''''''''''''''''

The module ``reliability.Fitters`` provides many probability distribution fitting functions. These functions can be thought of in two categories; non-location shifted distributions [eg. Weibull(α,β)], and location shifted distributions [eg. Weibull(α,β,γ)]. All of the non-location shifted distributions can be fitted to data containing left or right censored data (either but not both), however the location shifted distributions cannot be fitted to data containing left censored data. A discussion on why this is the case is presented at the end of this section. All distributions in the Fitters module are named with their number of parameters (eg. Fit_Weibull_2P uses α,β, whereas Fit_Weibull_3P uses α,β,γ). This is intended to remove ambiguity about what distribution you are fitting.

Distributions are fitted simply by using the desired function and specifying the data as failures, right_censored, or left_censored data. You may not specify both left and right censored data at the same time. You must have at least as many failures as there are distribution parameters or the fit would be under-constrained. It is generally advisable to have at least 4 data points as the accuracy of the fit is proportional to the amount of data. Once fitted, the results are assigned to an object and the fitted parameters can be accessed by name, as shown in the examples below. The goodness of fit criterions are also available as AICc (Akaike Information Criterion corrected) and BIC (Bayesian Information Criterion), though these are more useful when comparing the fit of multiple distributions such as in the `Fit_Everything <https://reliability.readthedocs.io/en/latest/Fitting%20all%20available%20distributions%20to%20data.html>`_ function. As a matter of convenience, each of the modules in Fitters also generates a distribution object that has the parameters of the fitted distribution.

The supported distributions for failures and right censored data are:

-   Weibull_2P
-   Weibull_3P
-   Exponential_1P
-   Exponential_2P
-   Gamma_2P
-   Gamma_3P
-   Lognormal_2P
-   Normal_2P
-   Beta_2P
-   Weibull_Mixture (see the `section <https://reliability.readthedocs.io/en/latest/Weibull%20mixture%20models.html>`_ on this)

The supported distributions for failures and left censored data are:

-   Weibull_2P
-   Exponential_1P
-   Gamma_2P
-   Lognormal_2P
-   Normal_2P
-   Beta_2P
-   Weibull_Mixture (see the `section <https://reliability.readthedocs.io/en/latest/Weibull%20mixture%20models.html>`_ on this)

.. note:: The Beta distribution is only for data in the range {0,1}. Specifying data outside of this range will cause an error.

.. note:: The current method of fitting the location shifted distributions (Weibull_3P, Exponential_2P, Gamma_3P) is not perfectly accurate as it uses a shortcut of setting γ equal to the lowest datapoint. The optimisation method is highly sensitive to the quality of the initial guess and this initial guess is provided by scipy which does a rather poor job of estimating the parameters for location shifted distributions. This is planned to be solved in a future release but will require a reqrite of the optimization functions that are curretly used from scipy so the solution is not simple. If you are fitting location shifted distributions to small or heavily censored datasets and your work is important, I would recommend using commercial software for maximum accuracy. If you have a lot of data and not too much censored data then ``reliability.Fitters`` should do a reasonable job with location shifted distributions.

If you do not know which distribution you want to fit, then please see the `section <https://reliability.readthedocs.io/en/latest/Fitting%20all%20available%20distributions%20to%20data.html>`_ on using the Fit_Everything function which will find the best distribution to describe your data.

Each of the fitters listed above (except Fit_Weibull_Mixture) has the following inputs and outputs:

Inputs:

-   failures - an array or list of failure data
-   left_censored - an array or list of left censored data. This may only be specified if fitting non-location shifted distributions.
-   right_censored - an array or list of right censored data
-   show_probability_plot - True/False. Defaults to True. Produces a probability plot of the failure data and fitted distribution.
-   print_results - True/False. Defaults to True. Prints a dataframe of the point estimate, standard error, Lower CI and Upper CI for each parameter.
-   CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.

Outputs (the following example outputs are for the Fit_Weibull_2P distribution but for other distributions the parameter names may be different from alpha and beta):

-   alpha - the fitted Weibull_2P alpha parameter
-   beta - the fitted Weibull_2P beta parameter
-   loglik2 - LogLikelihood*-2
-   AICc - Akaike Information Criterion
-   BIC - Bayesian Information Criterion
-   distribution - a Distribution object with the parameters of the fitted distribution
-   alpha_SE - the standard error (sqrt(variance)) of the parameter
-   beta_SE - the standard error (sqrt(variance)) of the parameter
-   Cov_alpha_beta - the covariance between the parameters
-   alpha_upper - the upper CI estimate of the parameter
-   alpha_lower - the lower CI estimate of the parameter
-   beta_upper - the upper CI estimate of the parameter
-   beta_lower - the lower CI estimate of the parameter
-   results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
-   success - True/False. Indicated whether the solution was found by autograd. If success is False a warning will be printed indicating that scipy's fit was used as autograd failed. This fit will not be accurate if there is censored data as scipy does not have the ability to fit censored data. Failure of autograd to find the solution should be rare and if it occurs, it is likely that the distribution is an extremely bad fit for the data. Try scaling your data, removing extreme values, or using another distribution.

To learn how we can fit a distribution, we will start by using a simple example with 10 failure times. These times were generated from a Weibull distribution with α=50, β=2. Note that the output also provides the confidence intervals and standard error of the parameter estimates. The probability plot is generated be default (you will need to specify plt.show() to show it). See the section on `probability plotting <https://reliability.readthedocs.io/en/latest/Probability%20plots.html#what-does-a-probability-plot-show-me>`_ to learn how to interpret this plot.

.. code:: python

    from reliability.Fitters import Fit_Weibull_2P
    import matplotlib.pyplot as plt
    data = [42.1605147, 51.0479599, 41.424553, 35.0159047, 87.3087644, 30.7435371, 52.2003467, 35.9354271, 71.8373629, 59.171129]
    wb = Fit_Weibull_2P(failures=data)
    plt.show()

    '''
    Results from Fit_Weibull_2P (95% CI):
               Point Estimate  Standard Error   Lower CI   Upper CI
    Parameter                                                      
    Alpha           56.682270        6.062572  45.962661  69.901951
    Beta             3.141684        0.733552   1.987995   4.964890
    '''

.. image:: images/Fit_Weibull_2P.png

It is beneficial to see the effectiveness of the fitted distribution in comparison to the original distribution. In this second example, we are creating 500 samples from a Weibull distribution and then we will right censor all of the data above our chosen threshold. Then we are fitting a Weibull_3P distribution to the data. Note that we need to specify "show_probability_plot=False, print_results=False" in the Fit_Weibull_3P to prevent the normal outputs from the fitting functions from being displayed.

.. code:: python

    from reliability.Distributions import Weibull_Distribution
    from reliability.Fitters import Fit_Weibull_3P
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(2)  # this is just for repeatability in this tutorial
    a = 30
    b = 2
    g = 20
    uncensored_failure_data = Weibull_Distribution(alpha=a, beta=b, gamma=g).random_samples(500)  # create some data
    cens = []
    fail = []
    threshold = 55  # censoring cutoff
    for item in uncensored_failure_data:
        if item >= threshold:  # this will right censor any value above the threshold
            cens.append(threshold)
        else:
            fail.append(item)
    print('There are' ,len(cens) ,'censored items.')
    wbf = Fit_Weibull_3P(failures=fail, right_censored=cens,show_probability_plot=False,print_results=False)  # fit the Weibull_3P distribution
    print('Fit_Weibull_3P parameters:\nAlpha:', wbf.alpha, '\nBeta:', wbf.beta, '\nGamma', wbf.gamma)
    xvals = np.linspace(0 ,150 ,1000)
    N ,bins ,patches = plt.hist(uncensored_failure_data, density=True, alpha=0.2, color='k', bins=30, edgecolor='k')  # histogram of the data
    for i in range(np.argmin(abs(np.array(bins ) -threshold)) ,len(patches)):  # this is to shade the censored part of the histogram as white
        patches[i].set_facecolor('white')
    Weibull_Distribution(alpha=a ,beta=b ,gamma=g).PDF(xvals=xvals ,label='True Distribution')  # plots the true distribution
    Weibull_Distribution(alpha=wbf.alpha, beta=wbf.beta, gamma=wbf.gamma).PDF(xvals=xvals, label='Fit_Weibull_3P' ,linestyle='--')  # plots the fitted Weibull_3P
    plt.title('Fitting comparison for failures and right censored data')
    plt.legend()
    plt.show()

    '''
    There are 118 censored items.
    Fit_Weibull_3P parameters:
    Alpha: 27.732547268103584 
    Beta: 1.8418848813302022 
    Gamma 21.486647464233737
    '''

.. image:: images/Fit_Weibull_3P_right_cens.png

As a final example, we will fit a Gamma_2P distribution to some partially left censored data. To provide a comparison of the fitting accuracy as the number of samples increases, we will do the same experiment with varying sample sizes. The results highlight that the accuracy of the fit is proportional to the amount of samples, so you should always try to obtain more data if possible.

.. code:: python

    from reliability.Distributions import Gamma_Distribution
    from reliability.Fitters import Fit_Gamma_2P
    import matplotlib.pyplot as plt
    import numpy as np
    
    np.random.seed(2)  # this is just for repeatability in this tutorial
    a = 30
    b = 4
    xvals = np.linspace(0, 500, 1000)
    
    trials = [10, 100, 1000, 10000]
    subplot_id = 141
    plt.figure(figsize=(12, 5))
    for t in trials:
        uncensored_failure_data = Gamma_Distribution(alpha=a, beta=b).random_samples(t)  # create some data
        cens = []
        fail = []
        threshold = 100  # censoring cutoff
        for item in uncensored_failure_data:
            if item <= threshold:  # this will left censor any value below the threshold
                cens.append(threshold)
            else:
                fail.append(item)
        gf = Fit_Gamma_2P(failures=fail, left_censored=cens, show_probability_plot=False, print_results=False)  # fit the Gamma_2P distribution
        print('\nFit_Gamma_2P parameters using', t, 'samples:', '\nAlpha:', gf.alpha, '\nBeta:', gf.beta)
        plt.subplot(subplot_id)
        num_bins = min(int(len(fail) / 2), 30)
        N, bins, patches = plt.hist(uncensored_failure_data, density=True, alpha=0.2, color='k', bins=num_bins, edgecolor='k')  # histogram of the data
        for i in range(0, np.argmin(abs(np.array(bins) - threshold))):  # this is to shade the censored part of the histogram as white
            patches[i].set_facecolor('white')
        Gamma_Distribution(alpha=a, beta=b).PDF(xvals=xvals, label='True')  # plots the true distribution
        Gamma_Distribution(alpha=gf.alpha, beta=gf.beta).PDF(xvals=xvals, label='Fitted', linestyle='--')  # plots the fitted Gamma_2P
        plt.title(str(str(t) + ' samples'))
        plt.ylim([0, 0.012])
        plt.xlim([0, 500])
        plt.legend()
        subplot_id += 1
    plt.subplots_adjust(left=0.09, right=0.96, wspace=0.41)
    plt.show()

    '''
    Fit_Gamma_2P parameters using 10 samples: 
    Alpha: 16.826016882071595 
    Beta: 5.534279313290292

    Fit_Gamma_2P parameters using 100 samples: 
    Alpha: 43.204091411221356 
    Beta: 2.84231256528535

    Fit_Gamma_2P parameters using 1000 samples: 
    Alpha: 30.23910765614133 
    Beta: 3.9312509126197566

    Fit_Gamma_2P parameters using 10000 samples: 
    Alpha: 29.911755243578337 
    Beta: 4.028977541477251
    '''

.. image:: images/Fit_Gamma_2P_left_cens.png

Why can't I fit a location shifted distribution to my left censored data?
-------------------------------------------------------------------------

This is because left censored data could occur anywhere to the left of the shifted start point (the gamma value), making the true location of a censored datapoint an impossibility if the gamma parameter is larger than the data. To think of it another way, for the same reason that we can't have a negative failure time on a Weibull_2P distribution, we can't have a failure time less than gamma on a Weibull_3P distribution. While it is certainly possible that left censored data come from a location shifted distribution, we cannot accurately determine what gamma is without a known minimum. In the case of no censoring or right censored data, the gamma parameter is simply set as the lowest failure time, but this convenience breaks down for left censored data.

How does the code work with censored data?
------------------------------------------

All functions in this module work using a Python library called `autograd <https://github.com/HIPS/autograd/blob/master/README.md/>`_ to find the derivative of the log-likelihood function. In this way, the code only needs to specify the log PDF, log CDF, and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained using scipy.stats on all the data as if it wasn't censored (since scipy doesn't accept censored data). If the distribution is an extremely bad fit or is heavily censored then these guesses may be poor and the fit might not be successful. In this case, the scipy fit is used which will be incorrect if there is any censored data. If this occurs, a warning will be printed. Generally the fit achieved by autograd is highly successful.

For location shifted distributions, the fitting of the gamma parameter is done using the lowest failure time, rather than by using a location shifted log-likelihood function. This is a shortcut way that is usually quite effective. It was found to be necessary because scipy's fit (which is used as the initial guess for autograd) was often wildly inaccurate for location shifted log-likelihood functions. This meant that autograd did not converge to a solution for location shifted distributions when given such a poor initial guess. Because the lognormal distribution is initially slow to increase (compared to Weibull, Gamma, and Exponential), there is often a substantial gap between zero and the smallest failure time in a lognormal distribution. This made it unreliable to use the "lowest failure time" method to find gamma, which is why there is no Fit_Lognormal_3P distribution. If you have a solution to this, please let me know.

A special thanks goes to Cameron Davidson-Pilon (author of the Python library `lifelines <https://github.com/CamDavidsonPilon/lifelines/blob/master/README.md/>`_ and website `dataorigami.net <https://dataorigami.net/>`_) for providing help with getting autograd to work, and for writing the python library `autograd-gamma <https://github.com/CamDavidsonPilon/autograd-gamma/blob/master/README.md/>`_, without which it would be impossible to fit the Beta or Gamma distributions using autograd.
