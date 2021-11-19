.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library.
I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability.
This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice.
If you have something to add:

- new features - please send me an email (alpha.reliability@gmail.com) or fill out the `feedback form <https://form.jotform.com/203156856636058>`_.
- bugs - please send me an email (alpha.reliability@gmail.com) or raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github.

The current release schedule is approximately every 6 to 8 weeks for minor releases, and rapidly (within the day) for bug fixes.

**Planned for version 0.8.0 (around Dec 2021)**

-    Within all fitters, use the FNRN format to give speed improvements in the same way as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P. Need to confirm this method does not introduce too much cumulative error due to floating point precision limitations.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Planned for version 0.9.0 (around Mar 2022)**

-    Add confidence intervals for Weibull_Mixture, Weibull_CR, Weibull_DS, Weibull_ZI, and Weibull_DSZI
-    Enable the confidence intervals for CDF, SF, CHF to be extracted programatically from the distribution object.
-    Make tests for everything that doesn't have a test yet.
-    Add plotting to all things that can plot in order to increase test coverage.

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add `step-stress models <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_ to ALT models.
-    Add the `Kijima G-renewal process <http://www.soft4structures.com/WeibullGRP/JSPageGRP.jsp>`_ to repairable systems.