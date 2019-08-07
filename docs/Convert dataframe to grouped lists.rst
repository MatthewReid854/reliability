.. image:: images/logo.png

-------------------------------------

Convert dataframe to grouped lists
''''''''''''''''''''''''''''''''''

This function was written because many of our datasets are in the form of columns with one column indicating the label (eg. success, failure) and the other column providing the value. Since many functions in ``reliability`` are written to accept grouped lists (lists where all the values belong to the same group such as right_censored failure times), it is beneficial to have a fast way to perform this conversion. This function will split the dataframe into as many grouped lists as there are unique values in the labels (left) column.

Inputs:

-   input_dataframe. This must be a dataframe containing 2 columns, where the label column is the left column and the value column is the right column. The column titles are not important.

Outputs:

-   lists , names - lists is a list of the grouped lists, and names is the identifying labels used to group the lists from the first column.
    
In the example below, we will create some data in a pandas dataframe, print it to see what it looks like, and then split it up using this function.

.. code:: python

    from reliability.Other_functions import convert_dataframe_to_grouped_lists
    import pandas as pd
    #create some data in a dataframe
    data = {'outcome': ['Failed', 'Censored', 'Failed', 'Failed', 'Censored'],
            'cycles': [1253, 1500, 1342, 1489, 1500]}
    df = pd.DataFrame(data, columns=['outcome', 'cycles'])
    print(df,'\n')
    # usage of the function
    lists, names = convert_dataframe_to_grouped_lists(df)
    print(names[0],lists[0])
    print(names[1],lists[1])
    
    '''
        outcome  cycles
    0    Failed    1253
    1  Censored    1500
    2    Failed    1342
    3    Failed    1489
    4  Censored    1500 

    Censored [1500, 1500]
    Failed [1253, 1342, 1489]
    '''
