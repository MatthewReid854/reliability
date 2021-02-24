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



This section will be written soon
