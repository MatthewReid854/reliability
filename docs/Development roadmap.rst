.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (alpha.reliability@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap. The current release schedule is approximately every 4 to 6 weeks.

**High priority (likely to be part of the next release)**

-    Confidence intervals for Gamma and Beta Distributions. Currently the confidence intervals have been completed for all of the other standard distributions.
-    Writing more automated tests. This will speed up the code development processes and help prevent future changes having unidentified effects.

**Medium priority (expected in early 2021)**

-    New Distributions along with their associated fitters and probability plots:

     - `Defective_Subpopulation_Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF does not reach 1 due to a lot of right censored data.
     - `Zero_Inflated_Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF starts above 0 due to a lot of 'dead on arrival' products in the dataset.

-    Add least squares as a method to obtain the initial guess for all Fitters. Currently this has only been implemented in Weibull and Loglogistic fitters but all the other Fitters use scipy which is slower but more accurate for small datasets.
-    Improvement to the optimisation routine within each of the fitters. Weibull has been optimised very well but the others need to same improvements such as updating the initial guess using the output until the BIC converges.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.
-    Merge Fit_Weibull_2P_grouped functionality into Fit_Weibull_2P. Input format will be failures=[], right_censored=[], n_failures=[], n_right_censored=[]. Once this is done for Weibull it will be replicated for all Fitters so they are faster for large datasets with repeated values.
-    Add converters between 3 data formats:
     
     - FR = failures, right_censored
     - FRNN = failures, right_censored, n_failures, n_right_censored
     - XCN = unit, censoring_code, number_of_events (where unit is time, km, cycles, etc.)

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Parameter Solver using GUI.
-    Speed improvements to fitters by using `JAX <https://github.com/google/jax>`_ to replace `Autograd <https://github.com/HIPS/autograd>`_. This will be done once the `issue <https://github.com/google/jax/issues/438>`_ preventing JAX from being installed on Windows machines is resolved. It is also reliant on approriate functions for Beta and Gamma being written, which is why autograd-gamma is a dependancy in addition to autograd.
