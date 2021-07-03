.. image:: images/logo.png

-------------------------------------

DSZI models
'''''''''''

What are DSZI models?
=====================

DSZI is an acronym for "Defective Subpopulation Zero Inflated". It is a combination of the Defective Subpopulation (DS) model and the Zero Inflated (ZI) model.

A defective subpopulation model is where the CDF does not reach 1 during the period of observation. This is caused when a portion of the population fails (known as the defective subpopulation) but the remainder of the population does not fail (and is right censored) by the end of the observation period.

A zero inflated model is where the CDF starts above 0 at the start of the observation period. This is caused by many "dead-on-arrival" items from the population, represented by failure times of 0. This is not the same as left censored data since left censored is when the failures occurred between 0 and the observation time. In the zero inflated model, the observation time is considered to start at 0 so the failure times are 0.

In a DSZI model, the CDF (which normally goes from 0 to 1) goes from above 0 to below 1, as shown in the image below. In this image the scale of the PDF and CDF are normalized so they can both be viewed together. In reality the CDF is much larger than the PDF.

.. image:: images/DSZI_explained.png

A DSZI model may be applied to any distribution (Weibull, Normal, Lognormal, etc.) using the transformations explained in the next section. The plot below shows how a Weibull distribution can become a DS_Weibull, ZI_Weibull and DSZI_Weibull. Note that the PDF of the DS, ZI, and DSZI models appears smaller than that of the original Weibull model since the area under the PDF is no longer 1. This is because the CDF does not range from 0 to 1.

.. image:: images/DSZI_combined.png

Equations of DSZI models
========================

This section will be written soon

Creating a DSZI model
=====================

This section will be written soon

Fitting a DSZI model
====================

This section will be written soon
