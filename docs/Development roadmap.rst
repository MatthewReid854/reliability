.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (m.reid854@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap.

**Next task (currently in development)**

-    Confidence intervals for Normal and Lognormal distributions (Gamma and Beta will come later). Currently the confidence intervals have only been completed for Weibull and Exponential distributions.
-    Improving the documentation to reflect the changes from Version 0.5.0
-    Any bug fixes that are necessary as a result of the large number of new features in Version 0.5.0
-    RAM_test_planners module. This will incorporate all the test planners that are currently included in Other_functions.
-    Move utilities into utils module.

**High priority (expected by the end of 2020)**

-    New Distributions ==> `Mixure Distribution <https://reliability.readthedocs.io/en/latest/Weibull%20mixture%20models.html>`_, Competing Risks Distribution, and `Defective Subpopulation <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_ Distribution. The only one with a Fitter is the Mixture Distribution. Other fitters will be developed.
-    Better visualisation of results from Fit_Weibull_Mixture to use a probability plot and the Mixture_Distribution object.
-    Fitters for DS_Weibull (defective subpopulation) and CR_Weibull (competing risks).
-    Add least squares as a method to obtain the initial guess for all Fitters. Currently this has been implemented in Fit_Weibull_2P, Fit_Weibull_2P_grouped, and Fit_Weibull_3P but all the other Fitters use scipy which is slower but more accurate for small datasets.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Low priority (more of a wish list at this point)**

-    SEV / Gumbel Distribution - This is another useful, but less common `probability distribution <http://reliawiki.org/index.php/The_Gumbel/SEV_Distribution>`_.
-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add the rank adjustment method to Nonparametric. Rank adjustment is the method used in Probability plotting (eg. to obtain the Median Ranks) and is a common and useful nonparametric estimate of the CDF, SF, and CHF.
-    3D ALT probability plots (reliability vs time vs stress). This feature is seen in `Reliasoft <http://reliawiki.com/index.php/File:ALTA6.9.png>`_.
-    Simple solver for various parameters - for example, find the parameters of a Weibull Distribution given the b5 and b95 life. Or for an Exponential Distribution find the time until the reliability reaches 50% given the MTTF. There are so many combinations of these problems which are easy to solve but would be nice to have in a simple tool. Reliasoft's `Quick Calculation Pad <https://help.synthesisplatform.net/weibull_alta9/quick_calculation_pad.htm>`_ is a GUI version of this. The free software `Parameter Solver <https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/6>`_ also does a few neat functions that I intend to include. Some of these do not have formulas so need to be solved iteratively.
-    Speed improvements to fitters by using cython for key components.
