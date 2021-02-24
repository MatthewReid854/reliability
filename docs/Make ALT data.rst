.. image:: images/logo.png

-------------------------------------

Make ALT data
'''''''''''''

This function is used to generate accelerated life testing (ALT) data. It is primarily used for testing the functions within ALT_fitters. The function `Other_functions.make_ALT_data` accepts the life distribution (Weibull, Lognormal, Normal, Exponential) and the life-stress model (Exponential, Eyring, Power, Dual_Exponential, Dual_Power, Power_Exponential), along with the parameters of the model and will create an object with the data in the correct format for the ALT models contained within `reliability.ALT_fitters`. The function contains many more inputs than are required and these inputs will only be used if they are part of the model. Please see the `equations <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_ of the ALT model you are using to determine what parameters are required. The function is designed to automatically censor a fraction of the data using the input fraction_censored.

Inputs:

-    distribution - "Weibull", "Exponential", "Lognormal", or "Normal"
-    life_stress_model - "Exponential", "Eyring", "Power", "Dual_Exponential", "Power_Exponential", "Dual_Power"
-    stress_1 - array or list of the stresses. eg. [100,50,10].
-    stress_2 - array or list of the stresses. eg. [0.8,0.6,0.4]. Required only if using a dual stress model. Must match the length of stress_1.
-    a - parameter from all models
-    b - parameter from Exponential and Dual_Exponential models
-    c - parameter from Eyring, Dual_Exponential, Power_Exponential, and Dual_Power models
-    n - parameter from Power, Power_Exponential, and Dual_Power models
-    m - parameter from Dual_Power model
-    beta - shape parameter for Weibull distributon
-    sigma - shape parameter for Normal or Lognormal distributions
-    use_level_stress - a number (if single stress) or list or array (if dual stress). Optional input.
-    number_of_samples - the number of samples to generate for each stress. Default is 100. The total data points will be equal to the number of samples x number of stress levels
-    fraction_censored - 0 for no censoring or between 0 and 1 for right censoring. Censoring is "multiply censored" meaning that there is no threshold above which all the right censored values will occur.
-    seed - random seed for repeatability

Outputs if using a single stress model:

-    failures - list
-    failure_stresses - list
-    right_censored - list (only provided if fraction_censored > 0)
-    right_censored_stresses - list (only provided if fraction_censored > 0)
-    mean_life_at_use_stress - float (only provided if use_level_stress is provided)

Outputs if using a dual stress model:

-    failures - list
-    failure_stresses_1 - list
-    failure_stresses_2 - list
-    right_censored - list (only provided if fraction_censored > 0)
-    right_censored_stresses_1 - list (only provided if fraction_censored > 0)
-    right_censored_stresses_2 - list (only provided if fraction_censored > 0)
-    mean_life_at_use_stress - float (only provided if use_level_stress is provided)

Example 1
---------

In this first example we will create ALT data from a Weibull_Eyring model. To verify the accuracy of the fitter we can compare the fitted model's parameters to the parameters we used to generate the data. Note that we only need to specify a, c, and beta since these are the three parameters of the Weibull_Exponential model.

.. code:: python

    from reliability.Other_functions import make_ALT_data
    from reliability.ALT_fitters import Fit_Weibull_Eyring

    ALT_data = make_ALT_data(distribution='Weibull',life_stress_model='Eyring',a=1500,c=-10,beta=2,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
    Fit_Weibull_Eyring(failures=ALT_data.failures, failure_stress=ALT_data.failure_stresses, right_censored=ALT_data.right_censored, right_censored_stress=ALT_data.right_censored_stresses, use_level_stress=300, show_probability_plot=False, show_life_stress_plot=False)
    
    '''
    Results from Fit_Weibull_Eyring (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Failures / Right censored: 240/60 (20% right censored) 

    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
            a         1439.38         93.9075   1255.33   1623.44
            c        -10.1164        0.233217  -10.5735  -9.65934
         beta         1.89927       0.0908621   1.72928   2.08598 

     stress  original alpha  original beta  new alpha  common beta beta change  acceleration factor
        500         901.054        1.82169    880.592      1.89927      +4.26%              11.3589
        400         2066.75        1.80167     2260.7      1.89927      +5.42%              4.42454
        350         4479.57        2.09167    4320.06      1.89927       -9.2%              2.31537

     Goodness of fit    Value
     Log-likelihood -1994.75
               AICc  3995.58
                BIC  4006.61 

    At the use level stress of 300, the mean life is 8875.99544
    '''

Example 2
---------

In this second example we will create ALT data from a Lognormal_Dual_Power model. To verify the accuracy of the fitter we can compare the fitted model's parameters to the parameters we used to generate the data. Note that we only need to specify c, m, n, and sigma since these are the four parameters of the Lognormal_Dual_Power model.

.. code:: python

    from reliability.Other_functions import make_ALT_data
    from reliability.ALT_fitters import Fit_Lognormal_Dual_Power

    use_level_stress = [250, 7]
    ALT_data = make_ALT_data(distribution='Lognormal', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, sigma=0.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.5, seed=1,use_level_stress=use_level_stress)
    Fit_Lognormal_Dual_Power(failures=ALT_data.failures, failure_stress_1=ALT_data.failure_stresses_1, failure_stress_2=ALT_data.failure_stresses_2, right_censored=ALT_data.right_censored, right_censored_stress_1=ALT_data.right_censored_stresses_1, right_censored_stress_2=ALT_data.right_censored_stresses_2, use_level_stress=use_level_stress, show_probability_plot=False, show_life_stress_plot=False)
    print('The mean life from the true model is',ALT_data.mean_life_at_use_stress)
    
    '''
    Results from Fit_Lognormal_Dual_Power (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Failures / Right censored: 250/250 (50% right censored) 

    Parameter  Point Estimate  Standard Error    Lower CI    Upper CI
            c     8.12819e+14     5.30872e+14 2.25971e+14 2.92371e+15
            m        -3.98122        0.111488    -4.19973    -3.76271
            n        -1.96541        0.112554    -2.18602    -1.74481
        sigma        0.466856       0.0206494    0.428089    0.509135 

      stress  original mu  original sigma  new mu  common sigma sigma change  acceleration factor
     500, 12      4.67615        0.480696 4.70595      0.466856       -2.88%              45.5551
      420, 9      6.08153        0.491475  5.9655      0.466856       -5.01%              12.9276
      400, 8      6.30556        0.431199 6.39124      0.466856       +8.27%              8.44548
      350, 6      7.49896        0.504774 7.48827      0.466856       -7.51%              2.81961
     245, 10      7.88354        0.413335 7.90429      0.466856      +12.95%              1.86001

     Goodness of fit   Value
     Log-likelihood -1825.8
               AICc 3659.69
                BIC 3676.46 

    At the use level stress of 250, 7, the mean life is 5618.65229

    The mean life from the true model is 5920.122530308318
    '''

*Recommended values*

Some parameters are more suitable than others for these models. The following parameters are recommended for use as a starting point if you are having difficulty in determining the rough order of magnitude of the values you should use:

- Exponential: a=2000, b=10
- Eyring: a=1500, c=-10
- Power: a=5e15, n=-4
- Dual_Exponential: a=50, b=0.1, c=500
- Dual_Power: c=1e15, m=-4, n=-2
- Power_Exponential: a=200, c=400, n=-0.5
