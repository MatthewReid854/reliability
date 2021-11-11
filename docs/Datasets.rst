.. image:: images/logo.png

-------------------------------------

Datasets
''''''''

.. admonition:: API Reference

   For inputs and outputs see the `API reference <https://reliability.readthedocs.io/en/latest/API/Datasets.html>`_.

There are a few datasets that have been included with reliability that users may find useful for testing and experimenting. Within `reliability.Datasets` the following datasets are available:

**Standard datasets**

- automotive - 10 failures, 21 right censored. It is used in `this example <https://reliability.readthedocs.io/en/latest/Kaplan-Meier%20estimate%20of%20reliability.html>`_
- mileage - 100 failures with no right censoring. It is used in the examples for `KStest <https://reliability.readthedocs.io/en/latest/Kolmogorov-Smirnov%20test.html>`_ and `chi2test <https://reliability.readthedocs.io/en/latest/Chi-squared%20test.html>`_.
- defective_sample - 1350 failures, 12296 right censored. It exhibits the behavior of a defective sample (also known as Limited failure population or Defective subpopulation).
- mixture - 71 failures, 3320 right censored. This is best modelled using a mixture model.
- electronics - 10 failures, 4072 right censored. It is used in `this example <https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html#using-fit-weibull-2p-grouped-for-large-data-sets>`_.

**ALT Datasets**

- ALT_temperature - conducted at 3 temperatures. 35 failures, 102 right censored. For example usage of many of the ALT Datasets see the `examples here <https://reliability.readthedocs.io/en/latest/Fitting%20a%20model%20to%20ALT%20data.html>`_.
- ALT_temperature2 - conducted at 4 temperatures. 40 failures, 20 right censored.
- ALT_temperature3 - conducted at 3 temperatures. 30 failures, 0 right censored.
- ALT_temperature4 - conducted at 3 temperatures. 20 failures, 0 right censored.
- ALT_load - conducted at 3 loads. 20 failures, 0 censored.
- ALT_load2 - conducted at 3 loads. 13 failures, 5 right censored.
- ALT_temperature_voltage - conducted at 2 different temperatures and 2 different voltages. 12 failures, 0 right censored.
- ALT_temperature_voltage2 - conducted at 3 different temperatures and 2 different voltages. 18 failures, 8 right censored.
- ALT_temperature_humidity - conducted at 2 different temperatures and 2 different humidities. 12 failures, 0 right censored.

**MCF Datasets**

- MCF_1 - this dataset contains failure and retirement times for 5 repairable systems. Exhibits a worsening repair rate.
- MCF_2 - this dataset contains failure and retirement times for 56 repairable systems. Exhibits a worsening then improving repair rate. Difficult to fit this dataset.

All datasets are functions which create objects and every dataset object has several attributes.

For the standard datasets, these attributes are:

- info - a dataframe of statistics about the dataset
- failures - a list of the failure data
- right_censored - a list of the right_censored data
- right_censored_stress - a list of the right_censored stresses (ALT datasets only)

For the ALT datasets, these attributes are similar to the above standard attributes, just with some variation for the specific dataset. These include things like:

- failure_stress_humidity
- right_censored_stress_voltage
- failure_stress_temp
- other similarly named attributes based on the dataset

For the MCF datasets these attributes are:

- times
- number_of_systems

If you would like more information on a dataset, you can type the name of the dataset in the help function (after importing it).

.. code:: python

    from reliability.Datasets import automotive
    print(help(automotive))

If you would like the statistics about a dataset you can access the info dataframe as shown below.

.. code:: python

    from reliability.Datasets import defective_sample
    print(defective_sample().info)

    '''
                Stat             Value
                Name  defective_sample
        Total Values             13645
            Failures      1350 (9.89%)
      Right Censored    12295 (90.11%)
    '''

The following example shows how to import a dataset and use it. Note that we must use brackets () to call the dataset (since it is a class) before accessing the failures and right_censored values.

.. code:: python

    from reliability.Datasets import automotive
    from reliability.Fitters import Fit_Weibull_2P
    Fit_Weibull_2P(failures=automotive().failures,right_censored=automotive().right_censored,show_probability_plot=False)
    
    '''
    Results from Fit_Weibull_2P (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Failures / Right censored: 10/21 (67.74194% right censored) 

    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
        Alpha          134243         42371.1   72314.7    249204
         Beta         1.15586        0.295842  0.699905   1.90884 

    Goodness of fit    Value
     Log-likelihood -128.974
               AICc  262.376
                BIC  264.816
                 AD  35.6075 
    '''

If you have an interesting dataset, please email me (alpha.reliability@gmail.com) and I may include it in this database.

If you would like to use any of these datasets in you own work, you are permitted to do so under the `LGPLv3 <https://www.gnu.org/licenses/lgpl-3.0.txt>`_ license. Under this license you must `acknowledge the source <https://reliability.readthedocs.io/en/latest/Citing%20reliability%20in%20your%20work.html>`_ of the datasets.
