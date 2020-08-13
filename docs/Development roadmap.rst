.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (m.reid854@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap.

**Next task (currently in development)**

-    Plotting enhancements to increase the detail in plots using less points (by generating more points where the plots curve and less where the plots are flat). Using 100 instead of 1000 points will make the plots much faster, particularly when multiple distributions are layered.
-    Plotting enhancements to the x and y scale such that the limits are based on the quantiles. This will ensure more relevant detail is shown, particularly for location shifted distributions.
-    Writing more automated tests. This will speed up the code development processes and help prevent future changes having unidentified effects.
-    Confidence intervals for Normal and Lognormal distributions (Gamma and Beta will come later). Currently the confidence intervals have only been completed for Weibull and Exponential distributions.

**High priority (expected by the end of 2020)**

-    New Distributions

     - `Defective Subpopulation_Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF does not reach 1 due to a lot of right censored data.
     - `Zero Inflated Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF starts above 0 due to a lot of 'dead on arrival' products in the dataset.
     - `Loglogistic Distribution <http://reliawiki.org/index.php/The_Loglogistic_Distribution>`_.
     - `Gumbel Distribution <http://reliawiki.org/index.php/The_Gumbel/SEV_Distribution>`_.

-    New Fitters for the above 4 new distributions

     - Fit_Weibull_DS
     - Fit_Weibull_ZI
     - Fit_Loglogistic
     - Fit_Gumbel
     
-    Add least squares as a method to obtain the initial guess for all Fitters. Currently this has only been implemented in Fit_Weibull_2P, Fit_Weibull_2P_grouped, and Fit_Weibull_3P but all the other Fitters use scipy which is slower but more accurate for small datasets.
-    Amalgamate Fit_Weibull_2P and Fit_Weibull_2P_grouped under a single function. Need to decide the best input format ==> failures=[], right_censored=[], xcn=[[x],[c],[n]] or df. If this works, do it for all Fitters so they are fast for large datasets.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add the rank adjustment method to Nonparametric. Rank adjustment is the method used in Probability plotting (eg. to obtain the Median Ranks) and is a common and useful nonparametric estimate of the CDF, SF, and CHF.
-    3D ALT probability plots (reliability vs time vs stress). This feature is seen in `Reliasoft <http://reliawiki.com/index.php/File:ALTA6.9.png>`_.
-    Simple solver for various parameters - for example, find the parameters of a Weibull Distribution given the b5 and b95 life. Or for an Exponential Distribution find the time until the reliability reaches 50% given the MTTF. There are so many combinations of these problems which are easy to solve but would be nice to have in a simple tool. Reliasoft's `Quick Calculation Pad <https://help.synthesisplatform.net/weibull_alta9/quick_calculation_pad.htm>`_ is a GUI version of this. The free software `Parameter Solver <https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/6>`_ also does a few neat functions that I intend to include. Some of these do not have formulas so need to be solved iteratively.
-    Speed improvements to fitters by using cython for key components.
