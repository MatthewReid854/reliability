.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library. I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability. This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice. If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (alpha.reliability@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap. The current release schedule is approximately every 6 to 8 weeks.

**Planned for version 0.5.7 (around May 2021)**

-    Make tests for everything that doesn't have a test yet.
-    Add plotting to all things that can plot in order to increase test coverage.
-    Provide ax argument so that plots which normally make a new figure (such as in Fit_Everything and ALT_Fitters) will instead plot on the axes they are given. This will enable subplots of things that normally occur in their own figure.

**Planned for version 0.5.8 (around July 2021)**

-    Correct the formatting in the API docs for every function.
-    DSZI Distribution along with its associated fitters and probability plots. DSZI is a combination of DS (`Defective Subpopulation Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF does not reach 1 due to a lot of right censored data) and ZI (`Zero Inflated Distribution <https://www.jmp.com/support/help/14-2/distributions-2.shtml>`_. This is for when the CDF starts above 0 due to a lot of 'dead on arrival' products in the dataset). A DSZI distribution may include features of both the DS and ZI distributions.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.

**Planned for version 0.5.9 (around Sept 2021)**

-    Within all fitters, use the FNRN format to give speed improvements in the same way as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P. Need to confirm this method does not introduce too much cumulative error due to floating point precision limitations.
-    Convert_ALT_data module needed. Similar combinations to Convert_data for the formats FSRS, XCNS, FNSRNS (Single stress data) and FSSRSS, XCNSS, FNSSRNSS (Dual stress data). Note that single stress and dual stress data cannot be converted to eachother so they will each form a set of 6 interchangable formats plus 3 xlsx conversion functions. This will require 18 new functions for the Convert_ALT_data module.

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add `step-stress models <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_ to ALT models.
