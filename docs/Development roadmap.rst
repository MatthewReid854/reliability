.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

Development of *reliability* has transitioned to primarily codebase maintenance (bug fixes).
I am turning my attention to learning Web Development in order to bring the power of this library to an interactive Web Application.
I hope this approach will help a wider audience who may not be familiar enough with Python to use this library in its current form.
If you're an experienced web developer and willing to help get this project off the ground, please get in touch via email (alpha.reliability@gmail.com).

The following items are residual "TO DO" tasks that I hope to some day return to. Listed in priority order they are:

-    Within all fitters, use the FNRN format to give speed improvements in the same way as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P. Need to confirm this method does not introduce too much cumulative error due to floating point precision limitations.
-    Add confidence intervals for Weibull_Mixture, Weibull_CR, Weibull_DS, Weibull_ZI, and Weibull_DSZI
-    Proportional Hazards Models - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Warranty Module. A new module of tools for warranty calculations.
-    Add `step-stress models <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_ to ALT models.
-    Add the `Kijima G-renewal process <http://www.soft4structures.com/WeibullGRP/JSPageGRP.jsp>`_ to repairable systems.
-    Improvement to the online documentation for how these methods work, including the addition of more formulas, algorithms, and better referencing.
-    Make tests for everything that doesn't have a test yet.
-    Add plotting to all things that can plot in order to increase test coverage.

I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability.
If you have something to add:

- new features - please send me an email (alpha.reliability@gmail.com) or fill out the `feedback form <https://form.jotform.com/203156856636058>`_.
- bugs - please send me an email (alpha.reliability@gmail.com) or raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github.

