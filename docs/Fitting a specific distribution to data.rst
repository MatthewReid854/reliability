.. _code_directive:

.. image:: images/logo.png

-------------------------------------

Fitting a specific distribution to data
'''''''''''''''''''''''''''''''''''''''

The module ``reliability.Fitters`` provides many probability distribution fitting functions. Many of these functions can accept left or right censored data, although the location shifted distributions (any distribution with γ>0) will not accept left censored data. A discussion on why this is the case is presented at the end of this section. All distributions in the Fitters module are named with their number of parameters. I.e. Weibull_2P uses α,β, whereas Weibull_3P uses α,β,γ. This is intended to remove ambiguity about what distribution you are fitting. A list of the available distribution fitters is provided below.

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
-   Weibull_Mixture (see the `section<https://reliability.readthedocs.io/en/latest/Fitting%20all%20available%20distributions%20to%20data.html>`_ on this)

The supported distributions for failures and left censored data are:

-   Weibull_2P
-   Exponential_1P
-   Gamma_2P
-   Lognormal_2P
-   Normal_2P
-   Beta_2P
-   Weibull_Mixture

Note that the Beta distribution is only for data in the range {0,1}.
If you do not know which distribution you want to fit, then please see the section on using the Fit_Everything function.

To learn how we can fit a distribution, we will use an example. In this example, we are creating some data from a weibull distribution (if you want the same data be sure to provide numpy with a random seed) and then intentionally censoring some of the data. Then we are fitting a distribution to the data.

.. code:: python

    more to come tomorrow :)
    from reliability.Distributions import Weibull_Distribution


Why can't I fit a location shifted distribution to my left censored data?
-------------------------------------------------------------------------

This is because left censored data could occur anywhere to the left of the shifted start point (the gamma value), making the true location of a censored datapoint an impossibility if the gamma parameter is larger than the data. To think of it another way, for the same reason that we can't have a negative failure time on a Weibull_2P distribution, we can't have a failure time less than gamma on a Weibull_3P distribution. While it is certainly possible that left censored data come from a location shifted distribution, we cannot accurately determine what gamma is without a known minimum. In the case of no censoring or right censored data, the gamma parameter is simply set as the lowest failure time, but this convenience breaks down for left censored data.

How do these work?
------------------

All functions in this module work using a Python library called `autograd <https://github.com/HIPS/autograd/blob/master/README.md/>`_ to find the derivative of the log-likelihood function. In this way, the code only needs to specify the log PDF, log CDF, and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained using scipy.stats on all the data as if it wasn't censored (since scipy doesn't accept censored data). If the distribution is an extremely bad fit or is heavily censored then these guesses may be poor and the fit might not be successful. In this case, the Scipy fit is used which will be incorrect if there is any censored data. Generally the fit achieved by autograd is highly successful.

A special thanks goes to Cameron Davidson-Pilon (author of the Python library `lifelines <https://github.com/CamDavidsonPilon/lifelines/blob/master/README.md/>`_ and website `dataorigami.net <https://dataorigami.net/>`_ for providing help with getting autograd to work, and for writing the python library `autograd-gamma<https://github.com/CamDavidsonPilon/autograd-gamma/blob/master/README.md/>`_, without which it would be impossible to fit the Beta or Gamma distributions using autograd.
