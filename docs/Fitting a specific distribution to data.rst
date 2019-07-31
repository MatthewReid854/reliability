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
-   Weibull_Mixture (see the section on this)

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

    from reliability.Distributions import Weibull_Distribution


Why can't I fit a location shifted distribution to my left censored data?
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This is because


How do these work?
''''''''''''''''''

All functions in this module work using a Python library called autograd to find the derivative of the log-likelihood function. In this way, the code only needs to specify the log PDF, log CDF, and log SF in order to obtain the fitted parameters. Initial guesses of the parameters are essential for autograd and are obtained using scipy.stats. If the distribution is an extremely bad fit or is heavily censored then these guesses may be poor and the fit might not be successful. In this case, the Scipy fit ws used which will be incorrect if there is any censored data. Generally the fit achieved by autograd is highly successful.

A special thanks goes to Cameron Davidson-Pilon (author of the Python library "lifelines" and website "dataorigami.net") for providing help with getting autograd to work, and for writing the python library "autograd-gamma", without which it would be impossible to fit the Beta or Gamma distributions using autograd.
