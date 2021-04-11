.. image:: images/logo.png

-------------------------------------

Crosshairs
''''''''''

This function provides interactive crosshairs on matplotlib plots. The crosshairs will follow the users' mouse cursor when they are near lines or points and will snap to these lines and points. Upon a mouse click the crosshairs will add an annotation to the plot. This annotation can be dragged to a new position. To delete the annotation, right click on it. To temporarily hide all annotations, toggle 'h' on your keyboard.

Note that crosshairs should be called after everything is added to the plot (but before plt.show()) so that the objects in the plot are identified for the 'snap to' feature. If something is added to the plot after calling crosshairs then you will not be able to move the crosshairs onto it.

If your interactive development environment does not generate the plot in its own window then your plot is not interactive and this will not work. For iPython notebook users, the interactive window should be available by typing "%matplotlib qt" after importing matplotlib as described `here <https://stackoverflow.com/questions/14261903/how-can-i-open-the-interactive-matplotlib-window-in-ipython-notebook>`_.

There are some customisable attributes of the crosshairs and annotations using the following inputs:

-   xlabel - the x-label for the annotation. Default is x.
-   ylabel - the y-label for the annotation. Default is y.
-   decimals - the number of decimals to use when rounding values in the crosshairs and in the annotation. Default is 2.
-   dateformat - the datetime format. If specified the x crosshair and label will be formatted as a date using the format provided. Default is None which results in no date format being used on x. For a list of acceptable dateformat strings see `strftime <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
-   plotting kwargs are also accepted. eg. color, linestyle, etc.

In the following example, we see the crosshairs being used to display the value of the Weibull CDF. The dynamic nature of this feature is shown in the video at the bottom of this page.

.. code:: python

    from reliability.Other_functions import crosshairs
    from reliability.Distributions import Weibull_Distribution
    import matplotlib.pyplot as plt

    Weibull_Distribution(alpha=50,beta=2).CDF()
    crosshairs(xlabel='t',ylabel='F') #it is important to call this last
    plt.show()

.. image:: images/crosshairs.png

.. raw:: html

    <div style="position: relative; padding-bottom: 2%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/xHRS0jHVN1w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

A special thanks goes to Antony Lee, the author of `mplcursors <https://mplcursors.readthedocs.io/en/stable/index.html>`_. The crosshairs function works using mplcursors to enable the 'snap to' feature and the annotations. Antony was very helpful in getting this to work.
