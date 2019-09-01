.. image:: images/logo.png

-------------------------------------

Probability plots
'''''''''''''''''

Proabability plots are a general term for several different plotting techniques. One of these techniques is a graphical method for comparing two data sets and includes `probability-probability <https://reliability.readthedocs.io/en/latest/Probability-Probability%20plots.html>`_ (PP) plots and `quantile-quantile <https://reliability.readthedocs.io/en/latest/Quantile-Quantile%20plots.html>`_ (QQ) plots. The second plotting technique is used for assessing the goodness of fit of a distribution by plotting the empirical CDF of the failures against their failure time and scaling the axes in such as way that the distribution appears linear. This method allows the reliability analyst to fit the distribution parameters using a simple "least squares" fitting method for a straight line and was popular before computers were capable of calculating the MLE estimates of the parameters. While we do not typically favour the use of least squares as a fitting method, we can still use probability plots to assess the goodness of fit.
The module ``reliability.Probability_plotting`` contains functions for each of the six distributions supported in ``reliability``. These functions are:

- Weibull_probability_plot
- Normal_probability_plot
- Lognormal_probability_plot
- Gamma_probability_plot
- Beta_probability_plot
- Exponential_probability_plot

There is also a function to obtain the plotting positions as well as the functions for custom axes scaling. These are explained more in the help file and will not be discussed further here.

Within each of the above probability plotting functions you may enter failure data as well as left or right censored data (either but not both). For those distributions that have a function in ``reliability.Fitters`` for fitting location shifted distributions (Weibull_3P, Gamma_3P, Exponential_2P), you can explicitly tell the probability plotting function to fit the gamma parameter using fit_gamma=True. By default the gamma parameter is not fitted. Fitting the gamma parameter will also change the x-axis to time-gamma such that everything will appear linear. An example of this is shown in the second example below.

Inputs:

- failures - the array or list of failure times
- right_censored - the array or list of right censored failure times
- left_censored - the array or list of left censored failure times
- fit_gamma - this is only included for Weibull, Gamma, and Exponential probability plots. Specify fit_gamma=True to fit the location shifted distribution.

Outputs:

- The plot is the only output. Use plt.show() to show it.

In the example below we generate some samples from a Normal Distribution and provide these to the probability plotting function. It is also possible to overlay other plots of the CDF as is shown by the dashed line.

.. code:: python

    from reliability.Distributions import Normal_Distribution
    from reliability.Probability_plotting import Normal_probability_plot
    import matplotlib.pyplot as plt
    dist = Normal_Distribution(mu=50,sigma=10)
    dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
    failures = dist.random_samples(100)
    Normal_probability_plot(failures=failures) #generates the probability plot
    plt.show()
    
.. image:: images/Normal_probability_plot.png

In this second example, we will fit an Exponential distribution to some right censored data. To create this data, we will draw it from an exponentail distribution that has a location shift of 12. Once again, the true CDF has also been plotted to provide the comparison. Note that the x-axis is time-gamma as it is necessary to subtract gamma from the x-plotting positions if we want the plot to appear linear.

.. code:: python

    from reliability.Distributions import Exponential_Distribution
    from reliability.Probability_plotting import Exponential_probability_plot
    import matplotlib.pyplot as plt
    dist = Exponential_Distribution(Lambda=0.25,gamma=12)
    raw_data = dist.random_samples(100) #draw some random data from an exponential distribution
    #right censor the data at 17
    failures = []
    censored = []
    for item in raw_data:
        if item > 17:
            censored.append(17)
        else:
            failures.append(item)
    Exponential_Distribution(Lambda=0.25).CDF(linestyle='--',label='True CDF') #we can't plot dist because it will be location shifted
    Exponential_probability_plot(failures=failures,right_censored=censored,fit_gamma=True) #do the probability plot. Note that we have specified to fit gamma
    plt.show()

.. image:: images/Exponential_probability_plot.png

In this third example, we will see how probability plotting can be used to highlight the importance of getting as much data as possible. This code performs a loop in which increasing numbers of samples are used for fitting a Weibull distribution and the accuracy of the results (shown both in the legend and by comparison with the True CDF) increases with the number of samples.

.. code:: python

    from reliability.Distributions import Weibull_Distribution
    from reliability.Probability_plotting import Weibull_probability_plot
    import matplotlib.pyplot as plt
    dist = Weibull_Distribution(alpha=250,beta=3)
    for i,x in enumerate([10,100,1000]):
        plt.subplot(131+i)
        dist.CDF(linestyle='--',label='True CDF')
        failures = dist.random_samples(x) #take 10,100,1000 samples
        Weibull_probability_plot(failures=failures) #this is the probability plot
        plt.title(str(str(x)+' samples'))
    plt.gcf().set_size_inches(15,7) #adjust the figuresize after creation. Necessary to do it after as it it automatically ajdusted within probability_plot
    plt.subplots_adjust(left=0.08,right=0.98,top=0.92,wspace=0.35) #formatting for the figure layout
    plt.show()

.. image:: images/Weibull_probability_plot_multi.png

In this final example, we take a look at how a probability plot can show us that there's something wrong with our assumption of a single distribution. To generate the data, the random samples are drawn from two different distributions which are shown in the left image. In the right image, the scatterplot of failure times is clearly non-linear. The red line is the attempt to fit a single Weibull_2P distribution and this will do a poor job of modelling the data. Also note that the points of the scatterplot do not fall on the True CDF of each distribution. This is because the median rank method of obtaining the plotting positions does not work well if the failure times come from more than one distribution. If you see a pattern like this, try a `mixture model <https://reliability.readthedocs.io/en/latest/Weibull%20mixture%20models.html>`_. Always remember that cusps, corners, and doglegs indicate a mixture of failure modes.

.. code:: python

    from reliability.Distributions import Weibull_Distribution
    from reliability.Probability_plotting import Weibull_probability_plot
    import matplotlib.pyplot as plt
    import numpy as np
    dist_1 = Weibull_Distribution(alpha=200,beta=3)
    dist_2 = Weibull_Distribution(alpha=900,beta=4)
    plt.subplot(121) #this is for the PDFs of the 2 individual distributions
    dist_1.PDF(label=dist_1.param_title_long)
    dist_2.PDF(label=dist_2.param_title_long)
    plt.legend()
    plt.title('PDF of two different distributions\nthat are contributing the failure data')
    plt.subplot(122) #this will be the probability plot
    dist_1_data = dist_1.random_samples(50)
    dist_2_data = dist_2.random_samples(50)
    all_data = np.hstack([dist_1_data,dist_2_data]) #combine the failure data into one array
    dist_1.CDF(label=dist_1.param_title_long) #plot each individual distribution for comparison
    dist_2.CDF(label=dist_2.param_title_long)
    Weibull_probability_plot(failures=all_data) #do the probability plot
    plt.gcf().set_size_inches(13,7) #adjust the figuresize after creation. Necessary to do it after as it it automatically ajdusted within probability_plot
    plt.subplots_adjust(left=0.08,right=0.96) #formatting the layout
    plt.legend(loc='lower right')
    plt.show()

.. image:: images/probability_plot_mixture.png
