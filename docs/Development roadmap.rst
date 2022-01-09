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

**Planned for version 0.9.0 (by end of 2022)**

-    Within all fitters, use the FNRN format to give speed improvements in the same way as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P. Need to confirm this method does not introduce too much cumulative error due to floating point precision limitations.
-    Add confidence intervals for Weibull_Mixture, Weibull_CR, Weibull_DS, Weibull_ZI, and Weibull_DSZI

**Continuous improvement tasks (ongoing)**

-    Improvement to the online documentation for how these methods work, including the addition of more formulas, algorithms, and better referencing.
-    Make tests for everything that doesn't have a test yet.
-    Add plotting to all things that can plot in order to increase test coverage.

**Low priority (more of a wish list at this point)**

-    Warranty Module. A new module of tools for warranty calculations.
-    Proportional Hazards Models - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add `step-stress models <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_ to ALT models.
-    Add the `Kijima G-renewal process <http://www.soft4structures.com/WeibullGRP/JSPageGRP.jsp>`_ to repairable systems.