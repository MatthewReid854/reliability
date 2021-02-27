.. image:: images/logo.png

-------------------------------------

Fitting all available models to ALT data
''''''''''''''''''''''''''''''''''''''''

Just as the function `Fitters.Fit_Everything` provides users with a quick way to `fit all available distributions <https://reliability.readthedocs.io/en/latest/Fitting%20all%20available%20distributions%20to%20data.html>`_ to their dataset, we can do a similar thing using `ALT_fitters.Fit_Everything_ALT` to fit all of the ALT models to an ALT dataset.

There are 24 ALT models available within `reliability`; 12 single stress models and 12 dual stress models. `Fit_Everything_ALT` will automatically fit the single stress or dual stress models based on whether the input includes single or dual stress data. Manual exclusion of certain models is also possible using the `exclude` argument. From the results, the models are sorted based on their goodness of fit test results, where the smaller the goodness of fit value, the better the fit of the model to the data.

Inputs:

-    failures - an array or list of the failure times (this does not need to be sorted).
-    failure_stress_1 - an array or list of the corresponding stresses (such as temperature or voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress.
-    failure_stress_2 - an array or list of the corresponding stresses (such as temperature or voltage) at which each failure occurred. This must match the length of failures as each failure is tied to a failure stress. Optional input. Providing this will trigger the use of dual stress models. Leaving this empty will trigger the use of single stress models.
-    right_censored - an array or list of the right failure times (this does not need to be sorted). Optional Input.
-    right_censored_stress_1 - an array or list of the corresponding stresses (such as temperature or voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress.
-    right_censored_stress_2 - an array or list of the corresponding stresses (such as temperature or voltage) at which each right_censored data point was obtained. This must match the length of right_censored as each right_censored value is tied to a right_censored stress. Conditionally optional input. This must be provided if failure_stress_2 is provided.
-    use_level_stress - The use level stress at which you want to know the mean life. Optional input. This must be a list [stress_1,stress_2] if failure_stress_2 is provided.
-    print_results - True/False. Default is True
-    show_probability_plot - True/False. Default is True. Provides a probability plot of each of the fitted ALT model.
-    show_best_distribution_probability_plot - True/False. Defaults to True. Provides a probability plot in a new figure of the best ALT model.
-    CI - confidence interval for estimating confidence limits on parameters. Must be between 0 and 1. Default is 0.95 for 95% CI.
-    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned with a warning.
-    sort_by - goodness of fit test to sort results by. Must be 'BIC','AICc', or 'Log-likelihood'. Default is BIC.
-    exclude - list or array of strings specifying which distributions to exclude. Default is None. Options are:

     -   Weibull_Exponential
     -   Weibull_Eyring
     -   Weibull_Power
     -   Weibull_Dual_Exponential
     -   Weibull_Power_Exponential
     -   Weibull_Dual_Power
     -   Lognormal_Exponential
     -   Lognormal_Eyring
     -   Lognormal_Power
     -   Lognormal_Dual_Exponential
     -   Lognormal_Power_Exponential
     -   Lognormal_Dual_Power
     -   Normal_Exponential
     -   Normal_Eyring
     -   Normal_Power
     -   Normal_Dual_Exponential
     -   Normal_Power_Exponential
     -   Normal_Dual_Power
     -   Exponential_Exponential
     -   Exponential_Eyring
     -   Exponential_Power
     -   Exponential_Dual_Exponential
     -   Exponential_Power_Exponential
     -   Exponential_Dual_Power

Outputs:

-    results - the dataframe of results. Fitted parameters in this dataframe may be accessed by name. See below example.
-    best_model_name - the name of the best fitting ALT model. E.g. 'Weibull_Exponential'. See above list for exclude.
-    best_model_at_use_stress - a distribution object created based on the parameters of the best fitting ALT model at the use stress. This is only provided if the use_level_stress is provided. This is because use_level_stress is required to find the scale parameter.
-    excluded_models - a list of the models which were excluded. This will always include at least half the models since only single stress OR dual stress can be fitted depending on the data.
-    parameters and goodness of fit results for each fitted model. For example, the Weibull_Exponential model values are:
     
     -   Weibull_Exponential_a
     -   Weibull_Exponential_b
     -   Weibull_Exponential_beta
     -   Weibull_Exponential_BIC
     -   Weibull_Exponential_AICc
     -   Weibull_Exponential_loglik

Example 1
---------

In this first example, we will use `Fit_Everything_ALT` on some data that is generated using the function `make_ALT_data`. We can then compare the fitted results to the input parameters used to create the data. `Fit_Everything_ALT` produces two plots; a grid of all the fitted models (usually 12 models unless you have excluded some) and a larger plot of the best fitting model's probability plot. These are shown by default, so using plt.show() is not required to display the plots.

.. code:: python

     from reliability.Other_functions import make_ALT_data
     from reliability.ALT_fitters import Fit_Everything_ALT

     ALT_data = make_ALT_data(distribution='Normal',life_stress_model='Exponential',a=500,b=1000,sigma=500,stress_1=[500,400,350],number_of_samples=100,fraction_censored=0.2,seed=1)
     Fit_Everything_ALT(failures=ALT_data.failures, failure_stress_1=ALT_data.failure_stresses, right_censored=ALT_data.right_censored, right_censored_stress_1=ALT_data.right_censored_stresses, use_level_stress=300)
     
     '''
     Results from Fit_Everything_ALT:
     Analysis method: Maximum Likelihood Estimation (MLE)
     Failures / Right censored: 240/60 (20% right censored) 

                    ALT_model           a       b        c        n    beta    sigma  Log-likelihood    AICc     BIC
           Normal_Exponential     501.729 985.894                            487.321        -1833.41 3672.89 3683.93
                Normal_Eyring      88.928         -13.9268                   490.833        -1835.23 3676.53 3687.56
                 Normal_Power 4.77851e+06                    -1.205          491.757        -1835.68 3677.45 3688.48
        Lognormal_Exponential     502.086 974.987                           0.151077        -1840.03 3686.14 3697.17
             Lognormal_Eyring     84.8059         -13.9272                  0.151992        -1841.54 3689.16 3700.19
              Lognormal_Power 4.43489e+06                  -1.19428         0.152211        -1841.89 3689.87  3700.9
          Weibull_Exponential     445.079 1206.61                    7.1223                 -1849.68 3705.44 3716.47
               Weibull_Eyring     28.2064         -14.1399          7.05022                 -1851.94 3709.96 3720.99
                Weibull_Power 4.43489e+06                  -1.18188 6.92681                 -1854.25 3714.57  3725.6
      Exponential_Exponential     492.845  1118.8                                           -2214.88  4433.8 4441.16
           Exponential_Eyring     74.9261         -14.0665                                  -2214.93 4433.91 4441.27
            Exponential_Power 4.23394e+06                  -1.16747                         -2214.94 4433.93  4441.3 

     At the use level stress of 300, the Normal_Exponential model has a mean life of 5249.98339
     '''

.. image:: images/Fit_everything_ALT_example1_grid.png

.. image:: images/Fit_everything_ALT_example1_single.png

Example 2
---------

In this second example, we will repeat what we saw in Example 1, but this time we will use a dual stress dataset generated using a Weibull_Dual_Power model.

.. code:: python

     from reliability.Other_functions import make_ALT_data
     from reliability.ALT_fitters import Fit_Everything_ALT

     ALT_data = make_ALT_data(distribution='Weibull', life_stress_model='Dual_Power', c=1e15, m=-4, n=-2, beta=2.5, stress_1=[500, 400, 350, 420, 245], stress_2=[12, 8, 6, 9, 10], number_of_samples=100, fraction_censored=0.2, seed=1)
     Fit_Everything_ALT(failures=ALT_data.failures, failure_stress_1=ALT_data.failure_stresses_1, failure_stress_2=ALT_data.failure_stresses_2, right_censored=ALT_data.right_censored, right_censored_stress_1=ALT_data.right_censored_stresses_1,right_censored_stress_2=ALT_data.right_censored_stresses_2, use_level_stress=[250,7])
     
     '''
     Results from Fit_Everything_ALT:
     Analysis method: Maximum Likelihood Estimation (MLE)
     Failures / Right censored: 400/100 (20% right censored) 

                          ALT_model       a       b           c        m        n    beta    sigma  Log-likelihood    AICc     BIC
                 Weibull_Dual_Power                 1.46475e+15  -4.1208 -1.84314 2.42854                 -2812.38 5632.85 5649.62
          Weibull_Power_Exponential 1356.32              2254.3           -2.2797 2.42384                 -2813.37 5634.81 5651.59
           Weibull_Dual_Exponential 1369.88 18.3903     1.79043                   2.37954                 -2820.27 5648.63  5665.4
               Lognormal_Dual_Power                 1.55721e+15 -4.13775 -1.93076         0.517428        -2833.83 5675.75 5692.52
        Lognormal_Power_Exponential 1362.75             2143.08          -2.37053         0.518215        -2834.45 5676.99 5693.77
         Lognormal_Dual_Exponential 1382.88  19.206     1.24225                            0.52403        -2838.99 5686.05 5702.83
             Exponential_Dual_Power                   1.733e+15 -4.13485 -1.89678                            -2995 5996.05 6008.65
      Exponential_Power_Exponential  1361.7              2429.1          -2.33559                         -2995.18  5996.4    6009
       Exponential_Dual_Exponential 1379.14 18.8852     1.59237                                           -2996.34 5998.72 6011.32
            Normal_Dual_Exponential 1174.35 14.1571     5.26764                            599.737        -3170.79 6349.65 6366.43
           Normal_Power_Exponential 1200.75              1565.9          -1.90507          600.014        -3171.18 6350.45 6367.22
                  Normal_Dual_Power                 1.52648e+15 -4.14955 -1.91248          441.469        -3257.31  6522.7 6539.47 

     At the use level stress of 250, 7, the Weibull_Dual_Power model has a mean life of 4725.71844
     '''

.. image:: images/Fit_everything_ALT_example2_grid.png

.. image:: images/Fit_everything_ALT_example2_single.png

Example 3
---------

In this third example, we will look at how to extract specific parameters from the output. This example uses a dataset from reliability.Datasets. The plots are turned off for this example.

.. code:: python

     from reliability.Datasets import ALT_temperature
     from reliability.ALT_fitters import Fit_Everything_ALT

     model = Fit_Everything_ALT(failures=ALT_temperature().failures, failure_stress_1=ALT_temperature().failure_stresses, right_censored=ALT_temperature().right_censored, right_censored_stress_1=ALT_temperature().right_censored_stresses,show_probability_plot=False,show_best_distribution_probability_plot=False)
     print('The Lognormal_Power model parameters are:\n a:',model.Lognormal_Power_a,'\n n:',model.Lognormal_Power_n,'\n sigma:',model.Lognormal_Power_sigma)
     
     '''
     Results from Fit_Everything_ALT:
     Analysis method: Maximum Likelihood Estimation (MLE)
     Failures / Right censored: 35/102 (74.45255474452554% right censored) 

                    ALT_model           a       b        c        n    beta    sigma  Log-likelihood    AICc     BIC
              Lognormal_Power 1.20893e+10                   -3.6399         0.961922        -339.183 684.546 693.126
             Lognormal_Eyring     142.294         -9.94803                  0.976603        -339.835 685.851  694.43
        Lognormal_Exponential     197.351 134.746                           0.986867        -340.144 686.468 695.047
                Weibull_Power 2.47966e+10                  -3.73283 1.44884                  -340.39  686.96  695.54
            Exponential_Power 3.08769e+12                  -4.85419                         -343.274 690.639 696.389
               Weibull_Eyring     151.091         -10.1367          1.42117                 -341.206 688.592 697.171
           Exponential_Eyring     211.096         -9.31393                                  -343.795 691.679  697.43
      Exponential_Exponential     266.147 71.2215                                           -343.991 692.071 697.821
          Weibull_Exponential     208.334 157.574                   1.39983                 -341.591 689.363 697.942
                Normal_Eyring     37.0322         -11.7653                   2464.67        -353.919 714.018 722.598
           Normal_Exponential     89.9062 855.006                            2439.04        -354.496 715.172 723.751
                 Normal_Power      772496                  -1.48137          2464.82        -465.469 937.119 945.698 

     The Lognormal_Power model parameters are:
      a: 12089297805.310057 
      n: -3.639895486209829 
      sigma: 0.9619219995672486
     '''
