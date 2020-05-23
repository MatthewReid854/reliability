.. image:: images/logo.png

-------------------------------------

Changelog
'''''''''

**Version: 0.5.0 --- Currently in production and testing**

-    Confidence intervals on fitted distributions (this is quite difficult and is taking longer than expected)
-    Minor improvements to color inheritance for probability_plotting.
-    The probability plot in Fit_Everything now uses the Exponential_probability_plot_Weibull_Scale instead of Exponential_probability_plot. It is much clearer to see the effectiveness of the fit using the Weibull scale.
-    Added and option to seed to all random_samples modules in Distributions for repeatable results.
-    Improvements to rounding of all titles, labels, and stats in Distributions and Probability_plotting using a new function, round_to_decimals
-    Added Other_functions.round_to_decimals which is keeps the specified number of decimals after leading zeros. This is useful as round would make very small values appear as 0.
-    Minor improvements to confidence interval color inheritance for Nonparametric.Kaplan_Meier and Nonparametric.Nelson_Aalen.
-    Within Stress_strength, the output format has changed from an object to a returned value of the probability of failure. This makes it much more simple to access the answer since the object had only one value.
-    Within Stress_strength, the method of obtaining the solution has been changed from monte carlo to integration. Thanks to Thomas Enzinger for providing the formula for this method. Using the integration method, accuracy is much higher (1e-11 vs 1e-3) and always consistent, and the speed is significantly improved over the monte carlo method. There is also no need to specify the number of monte_carlo_samples or to obtain the convergence plot.
-    Within Stress_strength, there are improvements to the fill_between method as it has errors in some special cases, and the colors used for shading have also been changed to improve the style

**Version: 0.4.9 --- Released: 27 April 2020**

-    Updates to reliability_test_planner to include option for failure terminated test
-    Addition of this changelog to the documentation
