.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or a you find a bug, please email me (m.reid854@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap.

**Coming soon (currently in development)**

-    Fit_Weibull_2P for grouped data - Not only does this accept grouped data (where you can specify the number of failures or right censored at the same time) but it also handles it efficiently. This can reduce some big calculations from a few hours down to a few seconds.
-    Use of least squares as a more efficient method of obtaining the initial guess for all the Fitters. Currently scipy.stats is used which uses MLE, but since both of them don't consider censored data they are both inaccurate (compared to MLE), but least squares is much faster (especially for big data sets). Using least squares as an initial guess also generates the estimates which are accessible later if required for comparison.

**High priority (expected by the end of 2020)**

-    Improvement to how plots set the axes limits by looking at all data in a plot rather than just the current data being plotted. For plots in which the axes are rescaled (such as in all the probability plots), the axes limits are set based on the limits of the data. This is fine unless you call for two plots on top of each other, in which case the second plot will set the axes limits and not take into account data that is already on the plot, potentially leading to some data being hidden by the axes limits.
-    Reliability_test_plan - To determine the test time or sample size required to demonstrate reliability requirements. `Minitab <https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/reliability/supporting-topics/basics/reliability-analyses-in-minitab/>`_ has this feature.
-    Mean_cumulative_function (aka. Cumulative Intensity Function) - Used to determine if a repairable system is improving or worsening. `Minitab <https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/reliability/how-to/parametric-growth-curve/interpret-the-results/mean-cumulative-function/>`_ has this feature.
-    Limited_failure_population for Weibull_2P - Some data sets are from a population which does not all fail. Such populations are best modelled using a limited failure population model. `JMP <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_ has this distribution, but they call it "defective subpopulation". Meeker and Escobar (1998) call it "Limited Failure Population".
-    ALT_probability_plot_Weibull prints new alpha in [] when given list as input.

**Low priority (more of a wish list at this point)**

-    Logistic Distribution - This is another useful, but less common `probability distribution <https://en.wikipedia.org/wiki/Logistic_distribution>`_.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Confidence intervals (both time and reliability) on the probability plots.
-    Add the rank adjustment method to Nonparametric. Rank adjustment is the method used in Probability plotting (eg. to obtain the Median Ranks) and is a common and useful nonparametric estimate of the CDF, SF, and CHF.
-    3D ALT probability plots (reliability vs time vs stress). This feature is seen in `Reliasoft <http://reliawiki.com/index.php/File:ALTA6.9.png>`_.
