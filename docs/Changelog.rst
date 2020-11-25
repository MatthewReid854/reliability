.. image:: images/logo.png

-------------------------------------

Changelog
---------

**Version: 0.5.5 --- Currently unreleased --- due for release around mid-February**
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    All of the standard fitters have been significantly improved with the following features:

    -    Least Squares estimation is now available. Previously the fit was solely achieved using MLE. MLE remains the default.
    -    For the least squares estimation, users may select RRX, RRY, LS. RRX and RRY are rank regression on X and rank regression on Y respectively. LS will perform both RRX and RRY and use the one with the best log-likelihood.
    -    There are 3 optimisers to choose from for all of the standard fitters. These are L-BFGS-B, TNC, powell. Previously there was only an option for some of the fitters and the optimiser was not standardized.
    -    Removal of scipy as the method to obtain the initial guess for MLE. With the inclusion of least squares estimation, the MLE method is much faster since it is not reliant on scipy to provide an initial guess (which failed to account for right censored data and often gave a poor guess).

-    Addition of a new module for converting data between different formats. The module reliability.Convert_data allows for conversion between FR (failures, right censored), FNRN (failures, number of failures, right censored, number of right censored), and XCN (event time, censoring code, number of events). It also provides a streamlined process for importing data from xlsx files, for exporting data to xlsx files, and for printing the dataset in a dataframe for easy visualisation.

**API Changes**

-    All of the standard fitters now include method and optimizer arguments.
-    Fitters.Fit_Weibull_2P, Fitters.Fit_Weibull_3P, Fitters.Fit_Weibull_2P_grouped have had some changes to their input arguments so that they all include method and optimizer. The initial_guess_method option is gone as it has been replaced by least squares estimation.
-    The entire Stress_strength module has been deprecated. This is because there were (and likely only ever would be) two functions in this module which is not enough to justify a separate module. The two function have been moved into Other_functions and renamed. Full deprecation will occur in March 2021 (in version 0.5.6), and until then a DeprecationWarning will be printed and the old functions will still work. The renaming is as follows:

     -    reliability.Stress_strength.Probability_of_failure :math:`\Rightarrow` reliability.Other_functions.stress_strength
     -    reliability.Stress_strength.Probability_of_failure_normdist :math:`\Rightarrow` reliability.Other_functions.stress_strength_normal

**Bug Fixes**

-    fixed a bug in Reliability_testing.reliability_test_duration in which certain inputs resulted in 1 failure and the plot limits caused a crash when left=right limit.

**Other**

-    Utils has 2 new functions (linear_regression and least_squares). These are now used by Fitters to obtain the least squares estimates.
-    The format of all the printed fitters outputs has been improved. More detail is provided and the formatting is better.
-    Dataframes from fitters are formatted better to retain the index but not display it.
-    Text output for sample_size_no_failures
-    Text output for one_sample_proportion
-    Text output for two_proportion_test
-    one_sample_proportion will now return 0 or 1 for the lower and upper reliability estimates instead of NaN in cases when there are all failures or all successes.

**Version: 0.5.4 --- Released: 7 November 2020**
''''''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    Confidence intervals added for Normal, Lognormal, Loglogistic, and Gumbel Distributions. *Confidence intervals for the Gamma and Beta Distributions will be part of 0.5.5 in Feb 2021*
-    Added Gumbel_Distribution to Distributions
-    Added Gumbel_Distribution to Other_functions.distribution_explorer
-    Added Fit_Gumbel_2P to Fitters
-    Added Gumbel_probability_plot to Probability_plotting
-    Added Gumbel Distribution to Fitters.Fit_Everything
-    Added Gumbel Distribution to Other_functions.similar_distributions
-    Added Gumbel Distribution to Stress_strength.Probability_of_failure
-    Added Gumbel Distribution to Reliability_testing.chi2test and Reliability_testing.KStest
-    Added Loglogistic and Gumbel Distributions to PP_plot_parametric, QQ_plot_parametric, PP_plot_semiparametric, and QQ_plot_semiparametric. Loglogistic should have been added in version 0.5.3 but it was missed.
-    Added Loglogistic and Gumbel Distributions to Mixture Model and Competing Risks Model. Loglogistic should have been added in version 0.5.3 but it was missed.
-    Fit_Everything now plots everything in order of best fit for all 3 of the plots generated.
-    Both the Competing Risks Model and Mixture Model now work for negative xvals when the mixture contains one or more Normal and/or Gumbel Distributions. Previously these were be truncated at 0 which could lead to inaccuracies if the model contained Normal Distributions (or Gumbel Distributions, though Gumbel was not available previously).

**API Changes**

-    Confidence intervals were previously available for the Hazard functions of the Weibull and Exponential distributions. This capability has been removed as it was not useful (just as confidence intervals on the PDF are not useful). Any attempt to use confidence interval related keywords (such as CI and CI_type) on the HF of any distribution will generate an error.
-    Fit_Everything now includes an option to exclude distributions.
-    Fit_Expon_1P and Fit_Expon_2P are deprecated. These have been replaced by Fit_Exponential_1P and Fit_Exponential_2P. Using the old functions will still work and will issue a DeprecationWarning printed to the console. Full deprecation/removal will occur in March 2021 (in version 0.5.6). The reason for the change is to minimize the use of abbreviated terms. It was originaly abbreviated because the word Exponential_Distribution seemed too long, but this is no longer valid with Loglogistic_Distribution being added. Also, scipy's function for Exponential is "expon" so Fit_Expon_1P initially seemed like an appropriate abbreviation.
-    percentiles have been added to all fitters (except Gamma and Beta). This will print a table of percentiles (with bounds on time) to the console. This is similar to the output that Minitab gives when fitting a distribution.

**Bug Fixes**

-    Other_functions.distribution_explorer had a bug caused by a recent update to matplotlib. When a non-existent axis was deleted, the error matplotlib generated was a ValueError and that is now changed to AttributeError which was not being appropriately handled by distribution_explorer.
-    All of the standard distributions expected a list or array for their 5 functions (PDF, CDF, SF, HF, CHF). A command like this "dist.SF(1)" would cause an error and should have been entered as dist.SF([1]). This is now fixed such that if the input is not in a list or array then it will no longer produce an error and the output type will be np.float64.
-   Within Fit_Everything if only 3 points were entered some of the AIC values would be 'Insufficient Data'. If the user also specified sort_by='AIC' then an error would be raised by pandas trying to sort by strings and numbers. In this case the sort_by method will automatically be changed to BIC.
-   The Exponential confidence intervals were invisibe if there were only 2 failures for the fit. This was cause by the upper CI reaching 1 which is effectively infinity on a probability plot. 1's are now filtered out so the CI will always appear.

**Other**

-    Removed margins in the stress_strength plots so that the xaxis coincides with the plot window.
-    Changed layout of Fitters.Fit_Everything probability plot and PP plot to be 4x3 without Beta fitted and 5x3 with Beta fitted. This was necessary to include the Gumbel Distribution in the space that Beta previously used.
-    Formatting changes to Fitters.Fit_Everything PP plot so the red line extends to the edges of the plot.
-    The histogram plot in Fitters.Fit_Everything now has its legend in the order of the the results, such that the best fitting distribution will appear first in the legend.
-    Within Other_functions.similar_distributions there were cases when a 3P distribution was fitted and the optimal gamma was 0 (making it the same as its 2P distribution). A filter has been added so the 3P distribution will only be shown if the gamma parameter is non-zero.
-    Improved plots for Stress_strength so the distribution xvals extend beyond the plot xlims. This is only noticable if the plot is moved.
-    Adjusted scaling and line colors for all QQ and PP plots to improve the way they are displayed.
-    PP_plot_parametric now has labels for quantile lines which are linked to the axes coords, so if the plot is moves / zoomed the labels will follow the plotting window.
-    Improved the Mixture Model PDF and HF using the actual formula rather than taking the numerical derivatives of CDF and CHF respectively.
-    Fit_Everything can now accept a minimum of 2 failures (previously the minimum was 3) and it will automatically exclude the 3P distributions
-    All warnings throughout reliability are now printed in red.
-    New Utils function colorprint. This provides a simple API for printing in color, bold, underline and italic.
-    Improved input checking for all the fitters. This has been standardised in a Utils function so nothing is missed for each of the fitters.
-    Probability_plotting.plot_points previously has a minimum of 2 failures required to plot the points. The minimum is now 1 failure required.

**Version: 0.5.3 --- Released: 29 September 2020**
''''''''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    Added Loglogistic_Distribution to Distributions
-    Added Fit_Loglogistic_2P and Fit_Loglogistic_3P to Fitters
-    Added Loglogistic_probability_plot to Probability_plotting
-    Added Fit_Loglogistic_2P and Fit_Loglogistic_3P to Fitters.Fit_Everything
-    Added Loglogistic distribution to Other_functions.similar_distributions
-    Added Loglogistic distribution to Stress_strength.probability_of_failure
-    Added the function Reliability_testing.reliability_test_duration
-    Added the function Other_functions.distribution_explorer
-    Added Utils.probability_plot_xylims and Utils.probability_plot_xyticks which provide better axes limits and tick labels. These are now incorporated into all probability plots, ALT probability plots and ALT Fitters.
-    Added Chi-squared and Kolmogorov-Smirnov goodness of fit tests to Reliability_testing
-    Added Anderson-Darling goodness of fit test statistic into all Fitters (It is not approriate to use for ALT_fitters for the entire model). This now allows users to compare distributions goodness of fit using Log-likelihood, AICc, BIC, or AD. Note that the Anderson-Darling test statistic is the default goodness of fit test statistic in Minitab.
-    Added Utils.anderson_darling to simplify the process of calculating the AD statistic. It's a lot of formulas that are best packaged into a function that is called by each of the Fitters.
-    Added Datasets.mileage which is a simple dataset with no right censored data.
-    Added Nonparametric.RankAdjustment. This method is similar in results to Kaplan-Meier and Nelson-Aalen but very different in the method used.
-    Other_functions.make_right_censored_data can now create either singly-censored or multiply-censored data. Previously it only created singly-censored data.

**API Changes**

-    Reliability_testing.reliability_test_planner has an optional argument of two_sided which was set to True as default. This has been changed to one_sided=True, making the default calculation use the one-sided confidence interval and changing the argument name. The reason for this change was to align the function with the approach more commonly used in industry.
-    All probability plots had h1 and h2 options for the plotting heuristics. These have been replaced by the argument "a" which is the same as what h1 was. h2 can be calculated from h1 and the length of the dataset so it was redundant. "a" was chosen to align with `wikipedia <https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics>`_.
-    Thanks to the addition of the Nonparametric.RankAdjustment, the functions Probability_plotting.QQ_plot_semiparametric and Probability_plotting.PP_plot_semiparametric now allow for 'RA' as the option in their method. Previously the methods were limited to 'KM', and 'NA' for Kaplan-Meier and Nelson-Aalen.
-    Other_functions.make_right_censored_data now has an additional argument of fraction_censored which controls the amount of data to right censor when producing multiply-censored data. There is also a random seed argument added for repeatability.
-    All the ALT_fitters were missing loglik as an output. They had loglik2 which is the same as loglik*-2 but this is added for completeness and to match the outputs from Fitters.

**Bug Fixes**

-    Fixed autoscale for cases where the HF is constant so it no longer lies along the yaxis upper limit
-    Fit_Everything had a bug in the default xvals for the Beta_Distribution's histogram which caused an error in some special cases. This is now resolved.
-    All the quantile functions in each distribution didn't accept np.float64 and raised an error. They now accept this data type.
-    The AICc and BIC in all the ALT_fitters was slightly wrong due to a small coding error. This is now fixed.

**Other**

-    Fixed the HF and CHF equations for Exponential_Distribution to be actual equations. The is preferred than using the HF = PDF/SF and CHF=-ln(SF) relationships which breakdown when SF=0 at high xvals. This has also been implemented for the loglogistic distribution. Can't do it for Normal, Lognormal, Gamma, and Beta distributions as these do not have closed form solutions for HF and CHF which don't involve the SF.
-    Changed the Gamma_Distribution and Weibull_Distribution mode to be self.gamma when beta < 1. Previously it was "No mode exists when beta < 1" which is true from a formula perspective but it is clear that the mode is equal to gamma as that's where the asymptote occurs. The only distribution with "no mode exists..." is the Beta distribution as it can have 2 modes for certain values of alpha and beta.
-    Updated Utils.generate_X_array to use 200 points (rather than 100) and allocated more points to the right hand side of the plot (beyond b99). This was because plots were not displaying smoothly enough for distributions with high skewness.
-    Changed default plotting upper limit to b9999. Previously it was slightly more and was not a round quantile. Done for simplicity and minimal change will be noticed.
-    Changed the layout of the Probability plots and PP plots in Fit_Everything from a 5x2 grid to a 4x3 grid. This made more sense due to the addition of the Loglogistic Distribution which would have made the layout 6x2 which is too long.
-    Plotting enhancements to increase the detail in plots using less points (by generating more points where the plots curve and less where the plots are flat). Using 200 instead of 1000 points will make the plots much faster, particularly when multiple distributions are layered. In version 0.5.2 this was just done for the Weibull Distribution but it has now been implemented for all 7 of the standard probability distributions.
-    Plotting enhancements to the x and y scale such that the limits are based on the quantiles. This will ensure more relevant detail is shown, particularly for location shifted distributions. In version 0.5.2 this was just done for the done for Weibull Distribution but it has now been implemented for all 7 of the standard probability distributions.
-    Within Stress_strength.Probability_of_failure, the integration method has been changed from quad to trapz based on this `issue <https://github.com/MatthewReid854/reliability/issues/8>`_.
-    Within Stress_strength the legend text for both plots no longer formats the probability of failure as a percentage and the format is changed to use scientific notation which is much more appropriate for very small failure probabilities.
-    Within Stress_strength both functions will issue a warning if stress.mean > strength.mean to indicate that the user may have assigned the distributions in the wrong order.
-    The version requirements for all dependancies have been updated to their most recent versions. This is most important for scipy which recently had an update that affects the covariance matrix results.
-    Added __version__ to the __init__.py file so that the version number is recorded in the same way as other packages record it.
-    Other_functions.histogram has an argument for bins. Previously this accepted the exact bins to be used and if left blank calculated them using the `Freedman-Diaconis rule <https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule>`_. In addition to accepting the exact bins to use, the bins argument now accepts strings just like matplotlib and numpy, and the default is now 'auto'. See `numpy <https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html>`_ for more detail on the strings available.
-    KaplanMeier and NelsonAalen now consider previous xlim when plotting. This prevents plot limits from being overridden by the most recent plot.

**Version: 0.5.2 --- Released: 14 August 2020**
'''''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    New distributions

     - Mixture_Distribution
     - Competing_Risks_Distribution

-    A new fitter for the Weibull competing risks model (Fit_Weibull_CR)
-    The output of the Fit_Weibull_Mixture now includes a probability plot instead of a histogram of the PDF and CDF
-    The output of the Fit_Weibull_Mixture now prints the confidence interval estimates of the parameters
-    Added some datasets for use with the mean cumulative function (MCF_1 and MCF_2).

**API Changes**

-    Within Fitters.Fit_Weibull_mixture the option show_plot has been changed to show_probability_plot to align with all the other fitters.

**Bug Fixes**

-    Fixed the autoscale in Weibull and Exponential distributions that locked autoscaling when confidence intervals were plotted sequentially.
-    Automatic removal of zeros for all fitters (except Normal_2P). Previously the zeros were left in the data and resulted in NaNs and crashes. Also added a dedicated error to report input with times below zero.
-    Fixed the confidence interval bounds for Kaplan-Meier and Nelson-Aalen CHF plots. Some of the bounds were inf since the CHF = -ln(SF) which will be inf when SF=0.
-    MCF_Nonparametric and MCF_Parametric had a bug which caused crashes when the dataset included a system with only one censored time. This has now been fixed. 

**Other**

-    Minor clean up of code. Removed unnecessary imports, removed unused variables, etc. Hopefully this will have no noticable effects.
-    Within Fitters.Fit_Everything the histogram output has been improved with better formatting and it now uses the Freedman-Diaconis rule for obtaining optimal bin width.
-    Fixed Weibull HF and CHF equations to use actual equations and not PDF/SF or -ln(SF) as these result in NaN when SF=0 (an issue at high xvals). These changes are currently only implemented for Weibull_Distribution.
-    Improved creation of xvals for PDF,CDF,SF,HF,CHF within the Weibull Distribution. The changes now generate datapoints where there is more detail (between the 0.1% and 99.9% quantiles) such that only 100 datapoints are needed to show more detail than was previously achieved with 1000 datapoints. This is most noticable with Weibull distributions that have high beta values and are significantly location shifted. An example of this is shown in the plot below. These changes are only implemented for Weibull_Distribution but will be extended to all distributions in the very near future.
-    Improved autoscaling for the Weibull Distribution plots. For location shifted distributions, this zooms in on the 0.1% to 99.9% quantiles allowing users to see more detail. The HF and CHF ylimits are also limited based on the quantiles so that they do not obscure the detail if there is an asymptote to large values or infinity. An example of this is shown in the plot below. These changes are only implemented for Weibull_Distribution but will be extended to all distributions in the very near future.

.. image:: images/autoscale_improvement_v052.png

**Version: 0.5.1 --- Released: 08 July 2020**
'''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    More efficient method used within Other_functions.similar_distributions. Results are always consistent and more accurate now.
-    Other_functions.histogram. This plots a histogram with optimal bin width, better default formatting, and an option to shade bins white above a threshold.

**API Changes**

-    Some of the functions in reliability.Other_functions have been moved into reliability.Utils and reliability.Reliability_testing. The new layout is:

     - Utils :math:`\Rightarrow` round_to_decimals, transform_spaced, axes_transforms
     - Other_functions :math:`\Rightarrow` similar_distributions, convert_dataframe_to_grouped_lists, crosshairs, make_right_censored_data
     - Reliability_testing :math:`\Rightarrow` one_sample_proportion, two_proportion_test, sample_size_no_failures, sequential_sampling_chart, reliability_test_planner
     
-    Within Other_functions.similar_distributions the option 'monte_carlo_trials' has been removed as the distribution sampling method is no longer random.

**Bug Fixes**

-    Fixed confidence interval color inheritance for Nonparametric.Kaplan_Meier and Nonparametric.Nelson_Aalen. Previously the color was only inherited if specified rather than left as default.
-    The default axes labels for both Stress_strength.Probability_of_failure and Stress_strength.Probability_of_failure_normdist were reversed. The have now been switched to the correct labels.

**Other**

-    Documentation updates to reflect the API changes in Version 0.5.1


**Version: 0.5.0 --- Released: 04 July 2020**
'''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    Confidence intervals on fitted distributions ==> this has only been implemented for Weibull and Exponential. Is is quite difficult and takes considerable time and testing. I will do Normal and Lognormal distributions next, then Gamma and Beta distributions. I hope to finish them all by September 2020.
-    Confidence intervals have been disabled in in ALT_probability_plotting and ALT_fitters to avoid cluttering on the plot.
-    The probability plot in Fit_Everything now uses the Exponential_probability_plot_Weibull_Scale instead of Exponential_probability_plot. It is much clearer to see the effectiveness of the fit using the Weibull scale.
-    Added an option to seed the random_samples functions within the Distributions module. This allows for repeatable results.
-    Improvements to rounding of all titles, labels, and stats in Distributions and Probability_plotting using a new function, round_to_decimals.
-    Added Other_functions.round_to_decimals which keeps the specified number of decimals after leading zeros. This is useful as round would make very small values appear as 0.
-    Minor improvements to color inheritance for probability_plotting.
-    Minor improvements to confidence interval color inheritance for Nonparametric.Kaplan_Meier and Nonparametric.Nelson_Aalen.
-    Within Stress_strength, the method of obtaining the solution has been changed from monte carlo to integration. Thanks to Thomas Enzinger for providing the formula for this method in response to an `Issue <https://github.com/MatthewReid854/reliability/issues/4>`_ that was raised. Using the integration method, accuracy is much higher (1e-11 error now vs 1e-3 error previously) and always consistent, and the speed is significantly improved over the monte carlo method. As noted below in API changes, there is no need to specify the number of monte_carlo_samples and no option to obtain the convergence plot.
-    Within Stress_strength, the colors used for shading have been changed to improve the style.
-    Probability_plotting.plot_points now includes the option to plot the points for the PDF and HF. These are not very useful as they appear messy due to the discontinuous nature of the function, but they are added for completeness.
-    Added Other_functions.transform_spaced. This is similar to np.linspace and np.logspace but it creates an array that is 'weibull spaced', 'normal spaced', 'exponential spaced', 'beta spaced', or 'gamma spaced'. It is used to get data points for the confidence intervals so they are as evenly spaced as possible, particularly on probability paper. This function is likely to be moved into utils.
-    Other_functions.make_right_censored_data has been added. This function accepts uncensored data and a threshold, and returns failures and right_censored arrays.
-    Added `mplcursors <https://mplcursors.readthedocs.io/en/stable/index.html>`_ to requirements in setup.py as it is needed for the crosshairs function.
-    Added crosshairs function to Other_functions. This is a very useful feature that provides interactive crosshairs to the plot using snap-to feature and also adds annotations on click events. Thanks to Antony Lee (the author of mplcursors) for help with getting this to work using his library.

**Bug fixes**

-    Within Stress_strength, there are improvements to the fill_between method as it had errors in some special cases.
-    Fixed an `Issue <https://github.com/MatthewReid854/reliability/issues/6>`_ in Lognormal_Probability_Plot that occurred for very large numbers (above 1e20)

**API Changes**

-    Within Stress_strength, the output format has changed from an object to a returned value of the probability of failure. This makes it much more simple to access the answer since the object had only one value.
-    Within Stress_strength, the method of obtaining the solution has been changed from monte carlo to integration. As a result, there is now no need to specify the number of monte_carlo_samples and no option to obtain the convergence plot.
-    Added the options initial_guess_method and optimizer to Fit_Weibull_2P and Fit_Weibull_3P. They were previously only in Fit_Weibull_2P_grouped. It is planned to add these options to all fitters.
-    There is now the option CI_type for the Weibull and Exponential fitters. This allows users to chose between confidence bounds on reliability and time. This option will be added to all fitters as the confidence intervals for the other distributions are completed.

**Other**

-    Added tests folder. This is planned to include automated tests.
-    Created utils module. I plan to move some utilities into here that are currently inside other modules where users can access them, but users should never need to access them so they just create clutter in the dropdown lists of your IDE.
-    Added Reliability_testing module. I plan to move everything related to reliability testing out of Other_functions as there is now enough functions to justify a new module dedicated to reliability testing.
-    Documentation updates to reflect the changes in Version 0.5.0

**Version: 0.4.9 --- Released: 27 April 2020**
''''''''''''''''''''''''''''''''''''''''''''''

**New features**

-    Updates to reliability_test_planner to include option for failure terminated test

**Other**

-    Addition of this Changelog to the documentation
