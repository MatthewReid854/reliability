.. image:: images/logo.png

-------------------------------------

Credits to those who helped
'''''''''''''''''''''''''''

During the process of writing ``reliability`` there have been many problems that I was unable to solve alone. I would like to thank the following people who provided help and feedback on problems with the code and with the reliability concepts:

- Cameron Davidson-Pilon for help with getting autograd to work to fit censored data and for writing autograd-gamma which makes it possible to fit the gamma and beta distributions. Also for providing help with obtaining the variance matrix using the hessian matrix (this is similar to the Fisher Information Matrix) so that the confidence intervals for parameters could be estimated.
- Dr Vasiliy Krivtsov for providing feedback on PP and QQ plots, and further explaining optimal replacement time equations. Dr Krivtsov teaches 'Collection and analysis of Reliability Data (ENRE640)' at the University of Maryland.
- The stackoverflow user ImportanceOfBeingErnest for `this <https://stackoverflow.com/questions/57777621/matplotlib-custom-scaling-of-subplots-using-global-variables-does-not-work-if-th>`_ answer that was necessary to get the probability plotting functions working correctly for Gamma and Beta distributions.
