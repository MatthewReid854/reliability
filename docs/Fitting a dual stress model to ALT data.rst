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
-    optimizer - 'TNC', 'L-BFGS-B', 'powell'. Default is 'TNC'. These are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the inital guess (using least squares) will be returned with a warning.

Outputs:

-   a - fitted parameter from the Dual_Exponential model
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

This will be written soon

Example 2
---------

This will be written soon


