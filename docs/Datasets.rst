.. image:: images/logo.png

-------------------------------------

Datasets
''''''''

There are a few datasets that have been included with reliability that users may find useful for testing and experimenting.
The following datasets are available in ``reliability.Datasets``:

- automotive - This dataset is relatively small and a challenging task to fit with any distribution due to its size and shape. It also includes right censored data which makes fitting more difficult.
- defective_sample - This dataset is heavily right censored with intermixed censoring (not all censored values are greater than the largest failure). It exhibits the behavior of a defective sample (aka. Limited fraction defective).

All datasets are functions which create objects and every dataset object has three values. These are:

- info - a dataframe of statistics about the dataset
- failures - a list of the failure data
- right_censored - a list of the right_censored data

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

The following example shows how to import a dataset and use it. Note that we must use () before accessing the failures and righta_censored values.

.. code:: python

    from reliability.Datasets import automotive
    from reliability.Fitters import Fit_Weibull_2P
    Fit_Weibull_2P(failures=automotive().failures,right_censored=automotive().right_censored,show_probability_plot=False)
    
    '''
    Results from Fit_Weibull_2P (95% CI):
               Point Estimate  Standard Error      Lower CI       Upper CI
    Parameter                                                             
    Alpha        70800.334465    13513.304148  48704.685923  102920.022281
    Beta             1.143379        0.202691      0.807783       1.618400
    '''

If you have in interesting dataset, please email me (m.reid854@gmail.com) and I may include it in this database.
