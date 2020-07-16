.. image:: images/logo.png

-------------------------------------

Competing risk models
'''''''''''''''''''''

What are competing risks models?
================================

Competing risks models are a combination of two or more distributions that represent failure modes which are "competing" to end the life of the system being modelled. This model is similar to mixture models in the sense that it uses multiple distributions to create a new model that has a shape with more flexibility than a single distribution. However, unlike mixture models, we are not adding proportions of the PDF or CDF, but are instead multiplying the survival functions. The resultant distribution will have a CDF that is equal to or above all other CDFs in the model, rather than lying between CDFs as is the case in the mixture model. The formula for the competing risks model is typically written in terms of the survival function (SF). Since we may consider the system's reliability to depend on the reliability of all the parts of the system (each with its own failure modes), the equation is written as if the system was in series, using the product of the survival functions for each failure mode. For a competing risks model with 2 distributions, the equations are shown below:

:math:`{SF}_{Competing\,Risks} = {SF}_1 \times {SF}_2`

:math:`{CDF}_{Competing\,Risks} = 1-{SF}_{Competing\,Risks}`

Since :math:`{SF} = exp(-CHF)` we may equivalently write the competing risks model in terms of the hazard or cumulative hazard function as:

:math:`{HF}_{Competing\,Risks} = {HF}_1 + {HF}_2`

:math:`{CHF}_{Competing\,Risks} = {CHF}_1 + {CHF}_2`

:math:`{PDF}_{Competing\,Risks} = {HF}_{Competing\,Risks} \times {SF}_{Competing\,Risks}`

Another option to obtain the PDF, is to find the derivative of the CDF. This is easiest to do numerically since the formula for the SF of the competing risks model can become quite complex as more risks are added. Note that using the PDF = HF x SF method requires the conversion of nan to 0 in the PDF for high xvals. This is because the HF of the component distributions is obtained using PDF/SF and for the region where the SF and PDF of the component distributions is 0 the resulting HF will be nan.

The image below illustrates the difference between the competing risks model and the mixture model, each made up of the same two component distributions. Note that the PDF of the competing risks model is always equal to or to the left of the component distributions, and the CDF is equal to or higher than the component distributions. This shows how a failure mode that occurs earlier in time can end the lives of units under observation before the second failure mode has the chance to. This behaviour is characteristic of real systems which experience multiple failure modes, each of which could cause system failure.

.. image:: images/CRvsMM.png

Competing risks models are useful when there is more than one failure mode that is generating the failure data. This can be recognised by the shape of the PDF and CDF being outside of what any single distribution can accurately model. On a probability plot, a combination of failure modes can be identified by bends in the data that you might otherwise expect to be linear. An example of this is shown in the image below. You should not use a competing risks model just because it fits your data better than a single distribution, but you should use a competing risks model if you suspect that there are multiple failure modes contributing to the failure data you are observing. To judge whether a competing risks model is justified, look at the goodness of fit criterion (AICc or BIC) which penalises the score based on the number of parameters in the model. The closer the goodness of fit criterion is to zero, the better the fit.

See also `mixture models <https://reliability.readthedocs.io/en/latest/Mixture%20models.html>`_ for another method of combining distributions using the sum of the CDF rather than the product of the SF.

.. image:: images/CRprobplot.png

Creating a competing risks model
================================

This section will be written soon


Fitting a competing risks model
===============================

This section will be written soon

