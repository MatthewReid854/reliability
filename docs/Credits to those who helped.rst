.. image:: images/logo.png

-------------------------------------

Credits to those who helped
'''''''''''''''''''''''''''

During the process of writing *reliability* there have been many problems that I was unable to solve alone. I would like to thank the following people who provided help and feedback on problems with the code and with the reliability concepts:

- `Cameron Davidson-Pilon <https://github.com/CamDavidsonPilon>`_ for help with getting autograd to work to fit censored data and for writing `autograd-gamma <https://github.com/CamDavidsonPilon/autograd-gamma>`_ which makes it possible to fit the gamma and beta distributions. Also for providing help with obtaining the Fisher Information Matrix so that the confidence intervals for parameters could be estimated.
- `Dr. Vasiliy Krivtsov <http://www.krivtsov.net/>`_ for providing feedback on PP and QQ plots, for further explaining optimal replacement time equations, and for guidance in developing the Competing risk model. Dr. Krivtsov teaches "Collection and analysis of Reliability Data (ENRE640)" at the University of Maryland.
- `Dr. Mohammad Modarres <https://enme.umd.edu/clark/faculty/568/Mohammad-Modarres>`_ for help with PoF, ALT_fitters, and ALT_probability_plotting. Dr. Modarres teaches several reliability engineering subjects at the University of Maryland and has authored several of the textbooks listed under `recommended resources <https://reliability.readthedocs.io/en/latest/Recommended%20resources.html>`_.
- The Stack Overflow user ImportanceOfBeingErnest for `this answer <https://stackoverflow.com/questions/57777621/matplotlib-custom-scaling-of-subplots-using-global-variables-does-not-work-if-th>`_ that was necessary to get the probability plotting functions working correctly for Gamma and Beta distributions.
- Antony Lee for help in adapting parts of his `mplcursors <https://mplcursors.readthedocs.io/en/stable/index.html>`_ library into the crosshairs function in reliability.Other_functions.crosshairs 
- `Thomas Enzinger <https://github.com/TEFEdotCC>`_ for help in improving the method of finding the area in stress-strength interference between any two distributions. Previously this was done using a monte-carlo method, but Thomas' method is much more accurate and always consistent. This is incorporated in Version 0.5.0.
- `Karthick Mani <https://www.linkedin.com/in/manikarthick/>`_ for help implementing the Loglogistic and Gumbel Distributions including implementation of these distributions in Fitters and Probability_plotting.
- Jake Sadie for identifying an error in the formula used for stress-strength interference of any two distributions. This error has been corrected in version 0.5.7.