.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (alpha.reliability@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap.

**Currently in development**

-    Writing more automated tests. This will speed up the code development processes and help prevent future changes having unidentified effects.
-    Confidence intervals for Normal and Lognormal distributions (Gamma, Beta and Loglogistic will come later). Currently the confidence intervals have only been completed for Weibull and Exponential distributions.

**High priority (expected by the end of 2020)**

-    New Distributions along with their associated fitters and probability plots:

     - `Defective_Subpopulation_Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF does not reach 1 due to a lot of right censored data.
     - `Zero_Inflated_Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF starts above 0 due to a lot of 'dead on arrival' products in the dataset.
     - `Gumbel_Distribution <http://reliawiki.org/index.php/The_Gumbel/SEV_Distribution>`_.

-    Add least squares as a method to obtain the initial guess for all Fitters. Currently this has only been implemented in Weibull and Loglogistic fitters but all the other Fitters use scipy which is slower but more accurate for small datasets.
-    Merge Fit_Weibull_2P_grouped functionality into Fit_Weibull_2P. Input format will be failures=[], right_censored=[], n_failures=[], n_right_censored=[]. Once this is done for Weibull it will be replicated for all Fitters so they are fast large datasets with repeated values.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add the rank adjustment method to Nonparametric. Rank adjustment is the method used in Probability plotting (eg. to obtain the Median Ranks) and is a common and useful nonparametric estimate of the CDF, SF, and CHF.
-    Parameter Solver using GUI.
-    Speed improvements to fitters by using `JAX <https://github.com/google/jax>`_ to replace `Autograd <https://github.com/HIPS/autograd>`_. This will be done once the `issue <https://github.com/google/jax/issues/438>`_ preventing JAX from being installed on Windows machines is resolved.
