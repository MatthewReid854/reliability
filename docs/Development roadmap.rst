.. image:: images/logo.png

-------------------------------------

Development roadmap
'''''''''''''''''''

The following development roadmap is the current task list and implementation plan for the Python reliability library.
I welcome the addition of new suggestions, both large and small, as well as help with writing the code if you feel that you have the ability.
This roadmap is regularly changing and you may see some things remain on here for a while without progressing, while others may be prioritized at short notice.
If you have a suggested feature or you find a bug, please raise an `Issue <https://github.com/MatthewReid854/reliability/issues>`_ on Github or email me (alpha.reliability@gmail.com) and I will endeavour to either add it rapidly (for simple tasks and bug fixes) or add it to the roadmap.
The current release schedule is approximately every 6 to 8 weeks.

**Planned for version 0.6.0 (around August 2021)**

-    Fit_Weibull_DSZI
-    Correct the formatting in the API docs for every function - still need to do ALT_Fitters, Convert_data, Datasets, PoF, and Utils

**Planned for version 0.7.0 (around December 2021)**

-    Within all fitters, use the FNRN format to give speed improvements in the same way as Fit_Weibull_2P_grouped works internally. This will subsequently result in the deprecation of Fit_Weibull_2P_grouped once its advantage is integrated in Fit_Weibull_2P. Need to confirm this method does not introduce too much cumulative error due to floating point precision limitations.
-    Improvement to the online documentation for how some of these methods work, including the addition of more formulas, algorithms, and better referencing.
-    Make tests for everything that doesn't have a test yet.
-    Add plotting to all things that can plot in order to increase test coverage.

**Low priority (more of a wish list at this point)**

-    Warranty Module. This will be a new module of many tools for warranty calculation.
-    New reliability growth models. Currently there is only the Duane model. It is planned to include the Crow Extended and AMSAA PM2 models.
-    Cox Proportional Hazards Model - This is available in `Lifelines <https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model>`_.
-    Add `step-stress models <http://reliawiki.com/index.php/Time-Varying_Stress_Models>`_ to ALT models.
