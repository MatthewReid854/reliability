.. image:: images/logo.png

-------------------------------------

Make ALT data
'''''''''''''

This function is used to generate accelerated life testing (ALT) data. It is primarily used for testing the functions within ALT_fitters. The function `Other_functions.make_ALT_data` accepts the life distribution (Weibull, Lognormal, Normal, Exponential) and the life-stress model (Exponential, Eyring, Power, Dual_Exponential, Dual_Power, Power_Exponential), along with the parameters of the model and will create an object with the data in the correct format for the ALT models contained within `reliability.ALT_fitters`. The function contains many more inputs than are required and these inputs will only be used if they are part of the model. Please see the `equations <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_ of the ALT model you are using to determine what parameters are required. The function is designed to automatically censor a fraction of the data using the input fraction_censored.

.. admonition:: API Reference

   For inputs and outputs see the `API reference <https://reliability.readthedocs.io/en/latest/API/Other_functions/make_ALT_data.html>`_.

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
    Optimizer: TNC
    Failures / Right censored: 240/60 (20% right censored) 
    
    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
            a         1458.14         92.7751   1276.31   1639.98
            c        -10.1231        0.230245  -10.5743   -9.6718
         beta         1.92551       0.0921087   1.75318   2.11477 
    
     stress  original alpha  original beta  new alpha  common beta beta change  acceleration factor
        500         945.271        1.94553    920.348      1.92551      -1.03%              11.6466
        400          2193.3        1.75376    2385.03      1.92551      +9.79%              4.49426
        350         4822.96        2.11256    4588.29      1.92551      -8.85%              2.33615
    
     Goodness of fit   Value
     Log-likelihood -2000.5
               AICc 4007.09
                BIC 4018.12 
    
    At the use level stress of 300, the mean life is 9507.77152
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
    Optimizer: TNC
    Failures / Right censored: 250/250 (50% right censored) 
    
    Parameter  Point Estimate  Standard Error    Lower CI    Upper CI
            c     9.48288e+14     6.67128e+14 2.38844e+14 3.76502e+15
            m         -3.9731         0.11982    -4.20795    -3.73826
            n        -1.99518        0.123271    -2.23678    -1.75357
        sigma        0.491039       0.0212097     0.45118    0.534419 
    
     stress  original mu  original sigma  new mu  common sigma sigma change  acceleration factor
    500, 12      4.85616        0.496646 4.83656      0.491039       -1.13%              46.0321
     420, 9      6.15963        0.525041 6.10326      0.491039       -6.48%                12.97
     400, 8      6.39217        0.392671 6.53211      0.491039      +25.05%              8.44684
     350, 6      7.69905        0.550747 7.63662      0.491039      -10.84%              2.79905
    245, 10      8.02546        0.457947 8.03454      0.491039       +7.23%              1.88017
    
     Goodness of fit    Value
     Log-likelihood -1859.62
               AICc  3727.32
                BIC   3744.1 
    
    At the use level stress of 250, 7, the mean life is 6545.04098
    
    The mean life from the true model is 5920.122530308318
    '''

**Recommended values**

Some parameters are more suitable than others for these models. The following parameters are recommended for use as a starting point if you are having difficulty in determining the rough order of magnitude of the values you should use:

- Exponential: a=2000, b=10
- Eyring: a=1500, c=-10
- Power: a=5e15, n=-4
- Dual_Exponential: a=50, b=0.1, c=500
- Dual_Power: c=1e15, m=-4, n=-2
- Power_Exponential: a=200, c=400, n=-0.5
