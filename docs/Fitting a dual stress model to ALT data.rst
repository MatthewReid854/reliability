.. image:: images/logo.png

-------------------------------------

Fitting a dual stress model to ALT data
'''''''''''''''''''''''''''''''''''''''

Before reading this section it is recommended that readers are familiar with the concepts of `fitting probability distributions <https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html>`_, `probability plotting <https://reliability.readthedocs.io/en/latest/Probability%20plots.html>`_, and have an understanding of `what accelerated life testing (ALT) involves <https://reliability.readthedocs.io/en/latest/What%20is%20Accelerated%20Life%20Testing.html>`_.

The module `reliability.ALT_fitters` contains 24 `ALT models <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_; 12 of these models are for single stress and 12 are for dual stress. This section details the dual stress models, though the process for `fitting single stress models <https://reliability.readthedocs.io/en/latest/Fitting%20a%20single%20stress%20model%20to%20ALT%20data.html>`_ is similar. The decision to use a single stress or dual stress model depends entirely on your data. If your data has two stresses that are being changed then you will use a dual stress model.

The following dual stress models are available within ALT_fitters:

-    Fit_Weibull_Dual_Exponential
-    Fit_Weibull_Power_Exponential
-    Fit_Weibull_Dual_Power
-    Fit_Lognormal_Dual_Exponential
-    Fit_Lognormal_Power_Exponential
-    Fit_Lognormal_Dual_Power
-    Fit_Normal_Dual_Exponential
-    Fit_Normal_Power_Exponential
-    Fit_Normal_Dual_Power
-    Fit_Exponential_Dual_Exponential
-    Fit_Exponential_Power_Exponential
-    Fit_Exponential_Dual_Power

Each of the ALT models works in a very similar way so the documentation below can be applied to all of the dual stress models with minor modifications to the parameter names of the outputs. The following documentation is for the Weibull_Dual_Exponential model.

Inputs:

-    failures - an array or list of the failure times.
-    failure_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
-    failure_stress_2 - an array or list of the corresponding stress 2 (such as humidity) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
-    right_censored - an array or list of all the right censored failure times
-    right_censored_stress_1 - an array or list of the corresponding stress 1 (such as temperature) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
-    right_censored_stress_2 - an array or list of the corresponding stress 1 (such as humidity) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
-    use_level_stress - [stress_1, stress_2]. A two element list or array of the use level stresses at which you want to know the mean life. Optional input.
-    print_results - True/False. Default is True
-    show_probability_plot - True/False. Default is True
-    show_life_stress_plot - True/False. Default is True
-    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
-    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.

Outputs:

-    a - fitted parameter from the Dual_Exponential model
-    b - fitted parameter from the Dual_Exponential model
-    c - fitted parameter from the Dual_Exponential model
-    beta - the fitted Weibull_2P beta
-    loglik2 - Log Likelihood*-2 (as used in JMP Pro)
-    loglik - Log Likelihood (as used in Minitab and Reliasoft)
-    AICc - Akaike Information Criterion
-    BIC - Bayesian Information Criterion
-    a_SE - the standard error (sqrt(variance)) of the parameter
-    b_SE - the standard error (sqrt(variance)) of the parameter
-    c_SE - the standard error (sqrt(variance)) of the parameter
-    beta_SE - the standard error (sqrt(variance)) of the parameter
-    a_upper - the upper CI estimate of the parameter
-    a_lower - the lower CI estimate of the parameter
-    b_upper - the upper CI estimate of the parameter
-    b_lower - the lower CI estimate of the parameter
-    c_upper - the upper CI estimate of the parameter
-    c_lower - the lower CI estimate of the parameter
-    beta_upper - the upper CI estimate of the parameter
-    beta_lower - the lower CI estimate of the parameter
-    results - a dataframe of the results (point estimate, standard error, Lower CI and Upper CI for each parameter)
-    goodness_of_fit - a dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
-    change_of_parameters - a dataframe showing the change of the parameters (alpha and beta) at each stress level
-    mean_life - the mean life at the use_level_stress (only provided if use_level_stress is provided)
-    alpha_at_use_stress - the equivalent Weibull alpha parameter at the use level stress (only provided if use_level_stress is provided)
-    distribution_at_use_stress - the Weibull distribution at the use level stress (only provided if use_level_stress is provided)
-    probability_plot - the figure object from the probability plot (only provided if show_probability_plot is True)
-    life_stress_plot - the figure object from the life-stress plot (only provided if show_life_stress_plot is True)
    
Example 1
---------

In the following example, we will fit the Normal-Dual-Exponential model to an ALT dataset obtained from a temperature-voltage dual stress test. This dataset can be found in `reliability.Datasets`. We want to know the mean life at the use level stress of 330 Kelvin, 2.5 Volts so the parameter use_level_stress is specified. All other values are left as defaults and the results and plot are shown.

.. code:: python

    from reliability.Datasets import ALT_temperature_voltage
    from reliability.ALT_fitters import Fit_Normal_Dual_Exponential
    import matplotlib.pyplot as plt
    data = ALT_temperature_voltage()
    Fit_Normal_Dual_Exponential(failures=data.failures, failure_stress_1=data.failure_stress_temp, failure_stress_2=data.failure_stress_voltage,use_level_stress=[330,2.5])
    plt.show()

    '''
    Results from Fit_Normal_Dual_Exponential (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Failures / Right censored: 12/0 (0% right censored) 

    Parameter  Point Estimate  Standard Error    Lower CI  Upper CI
            a         4056.06         752.956     2580.29   5531.83
            b         2.98952        0.851787     1.32005   4.65899
            c      0.00220833      0.00488708 2.88625e-05  0.168963
        sigma         87.3192          17.824     58.5274   130.275 

     stress  original mu  original sigma  new mu  common sigma sigma change  acceleration factor
     378, 3        273.5         98.7258   273.5       87.3192      -11.55%              5.81287
     348, 5          463         81.8475     463       87.3192       +6.69%              3.43374
     348, 3       689.75         80.1759  689.75       87.3192       +8.91%              2.30492

     Goodness of fit    Value
     Log-likelihood -70.6621
               AICc  155.039
                BIC  151.264 

    At the use level stress of 330, 2.5, the mean life is 1589.82043
    '''

.. image:: images/Normal_dual_exponential_probplot.png

.. image:: images/Normal_dual_exponential_lifestress.png

In the results above we see 3 tables of results; the fitted parameters (along with their confidence bounds) dataframe, the change of parameters dataframe, and the goodness of fit dataframe. For the change of parameters dataframe the "original mu" and "original sigma" are the fitted values for the Normal_2P distribution that is fitted to the data at each stress (shown on the probability plot by the dashed lines). The "new mu" and "new sigma" are from the Normal_Dual_Exponential model. The sigma change is extremely important as it allows us to identify whether the fitted ALT model is appropriate at each stress level. A sigma change of over 50% will trigger a warning to be printed informing the user that the failure mode may be changing across different stresses, or that the model is inappropriate for the data. The acceleration factor column will only be returned if the use level stress is provided since acceleration factor is a comparison of the life at the higher stress vs the use stress.

Example 2
---------

In this second example we will fit the Lognormal_Power_Exponential model. Instead of using an existing dataset we will create our own data using the function make_ALT_data. The results show that the fitted parameters agree well with the parameters we used to generate the data, as does the mean life at the use stress. This accuracy improves with more data.

Two of the outputs returned are the figure handles for the probability plot and the life-stress plot. These handles can be used to set certain values. In the example below we see the axes labels being set to custom values after the plots have been generated but before the plots have been displayed.

.. code:: python

    from reliability.Other_functions import make_ALT_data
    from reliability.ALT_fitters import Fit_Lognormal_Power_Exponential
    import matplotlib.pyplot as plt
    use_level_stress = [150,3]
    ALT_data = make_ALT_data(distribution='Lognormal',life_stress_model='Power_Exponential',a=200,c=400,n=-0.5,sigma=0.5,stress_1=[500,400,350,420,245],stress_2=[12,8,6,9,10],number_of_samples=100,fraction_censored=0.5,seed=1,use_level_stress=use_level_stress)
    model = Fit_Lognormal_Power_Exponential(failures=ALT_data.failures, failure_stress_1=ALT_data.failure_stresses_1, failure_stress_2=ALT_data.failure_stresses_2, right_censored=ALT_data.right_censored, right_censored_stress_1=ALT_data.right_censored_stresses_1,right_censored_stress_2=ALT_data.right_censored_stresses_2, use_level_stress=use_level_stress)
    # this will change the xlabel on the probability plot
    model.probability_plot.axes[0].set_xlabel('Time (hours)')
    # this will change the axes labels on the life-stress plot
    model.life_stress_plot.axes[0].set_xlabel('Temperature $(^oK)$')
    model.life_stress_plot.axes[0].set_ylabel('Voltage (kV)')
    model.life_stress_plot.axes[0].set_zlabel('Life (hours)')

    print('The mean life at use stress of the true model is:',ALT_data.mean_life_at_use_stress)
    plt.show()
    
    '''
    Results from Fit_Lognormal_Power_Exponential (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Failures / Right censored: 250/250 (50% right censored) 

    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
            a          192.66         36.7262   120.678   264.642
            c         369.526         100.472   216.875   629.624
            n       -0.463811        0.110597 -0.680578 -0.247044
        sigma        0.466844        0.020649  0.428078  0.509122 

      stress  original mu  original sigma  new mu  common sigma sigma change  acceleration factor
     500, 12      5.11464        0.480696 5.14501      0.466844       -2.88%               4.6742
      420, 9      5.46727        0.491475 5.35184      0.466844       -5.01%              3.80088
      400, 8      5.34327        0.431199  5.4294      0.466844       +8.27%              3.51721
      350, 6      5.64245        0.504774 5.63164      0.466844       -7.51%              2.87321
     245, 10      5.61146        0.413335 5.63062      0.466844      +12.95%              2.87614

     Goodness of fit    Value
     Log-likelihood -1562.46
               AICc  3133.01
                BIC  3149.79 

    At the use level stress of 150, 3, the mean life is 894.30098

    The mean life at use stress of the true model is: 992.7627728988726
    '''

.. image:: images/Lognormal_power_exponential_probplot.png

.. image:: images/Lognormal_power_exponential_lifestress.png

.. note:: The 3D surface plot with scatter plot has a known visibility issue where the 3D surface will appear to be in front of the scatter plot even when it should be shown behind it. This `issue is internal to matplotlib <https://matplotlib.org/mpl_toolkits/mplot3d/faq.html#my-3d-plot-doesn-t-look-right-at-certain-viewing-angles>`_ and the only current fix is to change the plotting library to MayaVi.
