.. image:: images/logo.png

-------------------------------------

Credits to those who helped
'''''''''''''''''''''''''''

During the process of writing ``reliability`` there have been many problems that I was unable to solve alone. I would like to thank the following people who provided help and feedback on problems with the code and with the reliability concepts:

- Cameron Davidson-Pilon for help with getting autograd to work to fit censored data and for writing autograd-gamma which makes it possible to fit the gamma and beta distributions. Also for providing help with obtaining the Fisher Information Matrix so that the confidence intervals for parameters could be estimated.
- Dr. Vasiliy Krivtsov for providing feedback on PP and QQ plots, and further explaining optimal replacement time equations. Dr. Krivtsov teaches "Collection and analysis of Reliability Data (ENRE640)" at the University of Maryland.
- Dr. Mohammad Modarres for help with the PoF, ALT_fitters and ALT_probability_plotting sections. Dr. Modarres teaches several reliability engineering subjects at the University of Maryland and has authored several of the textbooks under `recommended resources <https://reliability.readthedocs.io/en/latest/Recommended%20resources.html>`_.
- The Stack Overflow user ImportanceOfBeingErnest for `this answer <https://stackoverflow.com/questions/57777621/matplotlib-custom-scaling-of-subplots-using-global-variables-does-not-work-if-th>`_ that was necessary to get the probability plotting functions working correctly for Gamma and Beta distributions.
