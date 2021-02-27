.. image:: images/logo.png

-------------------------------------

Getting your ALT data in the right format
'''''''''''''''''''''''''''''''''''''''''

Because the ALT probability models failures and right censored data from many stress levels, it was not practical to make an input for each stress level. Instead, the failure times are combined in a single input and the failure_stress input provides a list of the corresponding stresses at which each failure occurred. The same is true of the right_censored and right_censored_stress inputs.

To get your data in the correct format, ensure you have combined all your failure times into a single list or numpy array and there is a corresponding list or array of the same length that provides all of the stresses. The following example illustrates one method to do this if you do not have the list already imported from Excel or another source. This is done for failures only but if you have right_censored data then you would do the same thing, but keep it separate to the failure data. There is no need to sort the data in any particular order as this is all done automatically. The only requirement is that the length of failures matches the length of the failure_stress, and that there are no new stresses in right_censored_stress that are not present in failure_stress.

.. code:: python

    import numpy as np
    #create the data
    failure_times_at_stress_1 = [800,850,910,940]
    failure_stress_1 = [40,40,40,40]
    failure_times_at_stress_2 = [650,670,715,740]
    failure_stress_2 = [50,50,50,50]
    failure_times_at_stress_3 = [300,320,350,380]
    failure_stress_3 = [60,60,60,60]
    # combine the data
    failures = np.hstack([failure_times_at_stress_1,failure_times_at_stress_2,failure_times_at_stress_3])
    failure_stresses = np.hstack([failure_stress_1,failure_stress_2,failure_stress_3])
    # print for inspection
    print(failures)
    print(failure_stresses)
    
    '''
    [800 850 910 940 650 670 715 740 300 320 350 380]
    [40 40 40 40 50 50 50 50 60 60 60 60]
    '''
