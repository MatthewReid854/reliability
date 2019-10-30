.. image:: images/logo.png

-------------------------------------

Fitting a model to ALT data
'''''''''''''''''''''''''''

.. note:: This document and the associated functions are a work in progress. This notice will be removed when the models are available in the PyPI release of reliability.

Before reading this section, you should be familiar with `ALT probability plots <https://reliability.readthedocs.io/en/latest/ALT%20probability%20plots.html>`_, and `Fitting distributions <https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html>`_ to non-ALT datasets.

The module ``reliability.ALT`` contains fitting function for 15 different ALT life-stress models. Each model is a combination of the life model with the scale or location parameter replaced with life-stress model. For example, the Weibull-Exponential model is found by replacing the :math:`\alpha` parameter with the equation for the exponential life-stress model as follows:

:math:`\text{Weibull PDF:} \hspace{40mm} f(t) = \frac{\beta t^{ \beta - 1}}{ \alpha^ \beta} .exp \left(-(\frac{t}{\alpha })^ \beta \right)`

:math:`\text{Exponential Life-Stress model:} \hspace{5mm} L(T) = b.exp\left(\frac{a}{T} \right)`

Replacing :math:`\alpha` with :math:`L(T)` gives the PDF of the Weibull-Exponential model:

:math:`\text{Weibull-Exponential:} \hspace{25mm} f(t,T) = \frac{\beta t^{ \beta - 1}}{ \left(b.exp\left(\frac{a}{T} \right) \right)^ \beta} .exp \left(-\left(\frac{t}{\left(b.exp\left(\frac{a}{T} \right) \right) }\right)^ \beta \right)` 

The correct substitutions for each type of model are:

:math:`\text{Weibull:} \hspace{12mm} \alpha = L(T)`

:math:`\text{Normal:} \hspace{12mm} \mu = L(T)`

:math:`\text{Lognormal:} \hspace{5mm} \mu = ln \left( L(T) \right)`

The `life models <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_ available are:

- Weibull_2P
- Lognormal_2P
- Normal_2P

The life-stress models available are:

:math:`\text{Exponential (also used for Arrhenius equation):} \hspace{29mm} L(T)=b.exp \left(\frac{a}{T} \right)`

:math:`\text{Eyring:} \hspace{108mm} L(T)= \frac{1}{T} .exp \left( - \left( c - \frac{a}{T} \right) \right)`

:math:`\text{Power (also known as inverse power):} \hspace{48mm} L(S)=a .S^n`

:math:`\text{Dual-Exponential (also known as Temperature-Humidity):} \hspace{8mm} L(T,H)=c.exp \left(\frac{a}{T} + \frac{b}{H} \right)`

:math:`\text{Power-Exponential (also known as Thermal-Non-Thermal):} \hspace{5mm} L(T,S)=c.S^n.exp \left(\frac{a}{T} \right)`

When choosing a model, it is important to consider the physics involved rather than just trying everything to see what fits best. For exxample, the Power-Exponential model is most appropriate for a dataset that was obtained from an ALT test with a thermal and a non-thermal stress (such as temperature and voltage). It would be inappropriate to model the data from a Temperature-Humidity test using a Power-Exponential model as the physics suggests that a Temperature-Humidity test should be modelled using the Dual-Exponential model.

Each of the fitting functions works in a very similar way so the documentation below can be applied to all of the models with minor modifications where specified. The following documentation is for the Weibull-Power model.

Inputs:

-   failures - an array or list of the failure times.
-   failure_stress - an array or list of the corresponding stresses (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
-   right_censored - an array or list of all the right censored failure times
-   right_censored_stress - an array or list of the corresponding stresses (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
-   use_level_stress - The use level stress at which you want to know the mean life. Optional input.
-   print_results - True/False. Default is True
-   show_plot - True/False. Default is True
-   CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
-   initial_guess - starting values for [a,n]. Default is [500000,-1.5]. Optional input. If fitting fails, you will be prompted to try a better initial guess and you can use this input to do it.

Outputs:

-   a - fitted parameter from the Power model
-   n - fitted parameter from the Power model
-   beta - the fitted Weibull_2P beta
-   loglik2 - LogLikelihood*-2
-   AICc - Akaike Information Criterion
-   BIC - Bayesian Information Criterion
-   a_SE - the standard error (sqrt(variance)) of the parameter
-   n_SE - the standard error (sqrt(variance)) of the parameter
-   beta_SE - the standard error (sqrt(variance)) of the parameter
-   a_upper - the upper CI estimate of the parameter
-   a_lower - the lower CI estimate of the parameter
-   n_upper - the upper CI estimate of the parameter
-   n_lower - the lower CI estimate of the parameter
-   beta_upper - the upper CI estimate of the parameter
-   beta_lower - the lower CI estimate of the parameter
-   results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
-   mean_life - the mean life at the use_level_stress. Only calculated if use_level_stress is specified

In the following example, we will fit the Weibull-Power model to an ALT dataset obtained from a fatigue test. This dataset can be found in ``reliability.Datasets``. We want to know the mean life at the use level of 60 so the parameter use_level_stress is specified. All other values are left as defaults and the results and plot are shown.

.. code:: python

    from reliability.ALT import Fit_Weibull_Power
    from reliability.Datasets import ALT_load2
    import matplotlib.pyplot as plt
    data = ALT_load2()
    results = Fit_Weibull_Power(failures=data.failures,failure_stress=data.failure_stresses,right_censored=data.right_censored,right_censored_stress=data.right_censored_stresses,use_level_stress=60)
    plt.show()
    
    '''
    Results from Fit_Weibull_Power (95% CI):
               Point Estimate  Standard Error       Lower CI      Upper CI
    Parameter                                                             
    a           398816.334596   519397.494927 -619184.049122  1.416817e+06
    n               -1.417306        0.243944      -1.895427 -9.391838e-01
    beta             3.017297        0.716426       1.894563  4.805374e+00
    At the use level stress of 60 , the mean life is 1075.32844
    '''
    
.. image:: images/Weibull_power.png
