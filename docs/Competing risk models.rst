.. image:: images/logo.png

-------------------------------------

Competing risk models
'''''''''''''''''''''

What are competing risks Models?
================================

Competing risks models are a combination of two or more distributions that represent failure modes which are "competing" to end the life of the system being modelled. This model is similar to mixture models in the sense that it uses multiple distributions to create a new model that has a shape with more flexibility than a single distribution. However, unlike mixture models, we are not adding proportions of the PDF or CDF, but are instead multiplying the survival functions. The resultant distribution will have a CDF that is equal to or above all other CDFs in the model, rather than lying between CDFs as is the case in the mixture model. The formula for the competing risks model is typically written in terms of the survival function (SF). Since we may consider the system's reliability to depend on the reliability of all the parts of the system (each with its own failure modes), the equation is written as if the system was in series, using the product of the survival functions for each failure mode. For a competing risks model with 2 distributions, the equations are shown below:

:math:`{SF}_{Competing\,Risks} = {SF}_1 \times {SF}_2`

:math:`{CDF}_{Competing\,Risks} = 1-{SF}_{Competing\,Risks}`

Since :math:`{SF} = exp(-CHF)` we may equivalently write the competing risks model in terms of the hazard or cumulative hazard function as:

:math:`{HF}_{Competing\,Risks} = {HF}_1 + {HF}_2`

:math:`{CHF}_{Competing\,Risks} = {CHF}_1 + {CHF}_2`

:math:`{PDF}_{Competing\,Risks} = {HF}_{Competing\,Risks} \times {SF}_{Competing\,Risks}`

Another option to obtain the PDF, is to find the derivative of the CDF. This is easiest to do numerically since the formula for the SF of the competing risks model can become quite complex as more risks are added. Note that using the PDF = HF x SF method requires the conversion of nan to 0 in the PDF for high xvals. This is because the HF of the component distributions is obtained using PDF/SF and for the region where the SF and PDF of the component distributions is 0 the resulting HF will be nan.

The image below illustrates...





Creating a competing risks model
================================

This section will be written soon


Fitting a competing risks model
===============================

This section will be written soon

