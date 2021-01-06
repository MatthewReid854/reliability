.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (alpha.reliability@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap. The current release schedule is approximately every 4 to 8 weeks.

**Planned for version 0.5.6 (around March 2021)**

-    Confidence intervals for Gamma and Beta Distributions. Currently the confidence intervals have been completed for all of the other standard distributions. These were going to be added in 0.5.5 however the large number of changes in 0.5.5 meant that this task was postponed.
-    Make CDF,SF,CHF have an argument return_CI=False. If True then return an object with out.x,out.y_lower, out.y,out.y_upper. This will enable users to extract the CI values for bounds on time or reliability.
-    Generate API docs for every function.
-    Investigate linting with `flake8 <https://flake8.pycqa.org/en/latest/>`_.
-    Within Fitters the following line can sometimes fail "covariance_matrix = np.linalg.inv(hessian_matrix)" with the error "numpy.linalg.LinAlgError: Singular matrix". This happenes because the optimiser's solution is a non-invertable matrix. The solution is to catch such errors throughout fitters and return the appropriate error stating that the confidence intervals are unavailable.
-    Improvements to ALT_Fitters:

    - capture failure to obtain the initial guess and return message rather than a RuntimeError from curvefit
    - capture failure to fit using MLE and return message rather than a RuntimeError from autograd
    - add goodness of fit dataframe to results. This is the same as was done for all Fitters.
    - perform input checking using a new Utils function. This is the same as was done for all Fitters.
    - place all repeated code in Utils. This will significantly reduce the lines of code and make fixing any future errors faster. This is the same as was done for all Fitters.

-    Add Convert_ALT_data module to convert between 3 ALT data formats (grouped into single stress and dual stress data formats):

     - Single stress data formats:
     
          - FSRS = failures, failure_stress, right_censored, right_censored_stress
          - FNSRNS = failures, num_failures, failure_stress, right_censored, num_right_censored, right_censored_stress
          - XCNS = unit, censoring_code, number_of_events, stress (where unit is time, km, cycles, etc.)

     - Dual stress data formats:
     
          - FSSRSS = failures, failure_stress_1, failure_stress_2, right_censored, right_censored_stress_1, right_censored_stress_2
          - FNSSRNSS = failures, num_failures, failure_stress_1, failure_stress_2, right_censored, num_right_censored, right_censored_stress_1, right_censored_stress_2
          - XCNSS = unit, censoring_code, number_of_events, stress_1, stress_2 (where unit is time, km, cycles, etc.)

**Planned for version 0.5.7 (around April 2021)**

-    DSZI Distribution along with its associated fitters and probability plots. DSZI is a combination of DS (`Defective Subpopulation Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF does not reach 1 due to a lot of right censored data) and ZI (`Zero Inflated Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF starts above 0 due to a lot of 'dead on arrival' products in the dataset). A DSZI distribution may include features of both the DS and ZI distributions.

-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Planned for version 0.5.8 (around June 2021)**

-    Within all fitters, use the FNRN format to give speed improvements in the same was as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P.
-    Fully deprecate the following functions: Fit_Expon_1P, Fit_Expon_2P, Convert_dataframe_to_grouped_lists
-    Fully deprecate the Stress_strength module

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
