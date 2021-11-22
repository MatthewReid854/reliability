.. image:: images/logo.png

-------------------------------------

How does Maximum Likelihood Estimation work
'''''''''''''''''''''''''''''''''''''''''''

Maximum Likelihood Estimation (MLE) is a method of estimating the parameters of a model using a set of data.
While MLE can be applied to many different types of models, this article will explain how MLE is used to fit the parameters of a probability distribution for a given set of failure and right censored data.

MLE works by calculating the probability of occurrence for each data point (we call this the likelihood) for a model with a given set of parameters.
These probabilities are summed for all the data points.
We then use an optimizer to change the parameters of the model in order to maximise the sum of the probabilities.
This is easiest to understand with an example which is provided below.

There are two major challenges with MLE. These are the need to use an optimizer (making hand calculations almost impossible for distributions with more than one parameter), and the need for a relatively accurate initial guess for the optimizer.
The initial guess for MLE is typically provided using `Least Squares Estimation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.
A variety of `optimizers <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_ are suitable for MLE, though some may perform better than others so trying a few is sometimes the best approach.

There are several advantages of MLE which make it the standard method for fitting probability distributions in most software.
These are that MLE does not need the equation to be linearizable (which is needed in Least Squares Estimation) so any equation can be modeled.
The other advantage of MLE is that unlike Least Squares Estimation which uses the plotting positions and does not directly use the right censored data, MLE uses the failure data and right censored data directly, making it more suitable for heavily censored datasets.

The MLE algorithm
"""""""""""""""""

The MLE algorithm is as follows:

1. Obtain an initial guess for the model parameters (typically done using least squares estimation).
2. Calculate the probability of occurrence of each data point (f(t) for failures, R(t) for right censored, F(t) for left censored).
3. Multiply the probabilities (or sum their logarithms which is much more computationally efficient).
4. Use an optimizer to change the model parameters and repeat steps 2 and 3 until the total probability is maximized.

As mentioned in step 2, different types of data need to be handled differently:

+------------------------+-----------------------------------------------------------------+
| Type of observation    | Likelihood function                                             |
+========================+=================================================================+
| Failure data           | :math:`L_i(\theta|t_i)=f(t_i|\theta)`                           |
+------------------------+-----------------------------------------------------------------+
| Right censored data    | :math:`L_i(\theta|t_i)=R(t_i|\theta)`                           |
+------------------------+-----------------------------------------------------------------+
| Left censored data     | :math:`L_i(\theta|t_i)=F(t_i|\theta)`                           |
+------------------------+-----------------------------------------------------------------+
| Interval censored data | :math:`L_i(\theta|t_i)=F(t_i^{RI}|\theta) - F(t_i^{LI}|\theta)` |
+------------------------+-----------------------------------------------------------------+

In words, the first equation above means "the likelihood of the parameters (:math:`\theta`) given the data (:math:`t_i`) is equal to the probability of failure (:math:`f(t)`) evaluated at each time :math:`t_i` with that given set of parameters (:math:`\theta`)".
The equations for the PDF (:math:`f(t)`), CDF (:math:`F(t)`), and SF (:math:`R(t)`) for each distribution is provided `here <https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html>`_. 

Once we have the likelihood (:math:`L_i` ) for each data point, we need to combine them. This is done by multiplying them together (think of this as an AND condition).
If we just had failures and right censored data then the equation would be:

:math:`L(\theta|D) = \prod_{i=1}^{n} f_i(t_i^{\textrm{failures}}|\theta) \times R_i(t_i^{\textrm{right censored}}|\theta)`

In words this means that "the likelihood of the parameters of the model (:math:`\theta`) given the data (D) is equal to the product of the values of the PDF (:math:`f(t)`) with the given set of parameters (:math:`\theta`) evaluated at each failure (:math:`t_i^{\textrm{failures}}`), multiplied by the product of the values of the SF (:math:`R(t)`) with the given set of parameters (:math:`\theta`) evaluated at each right censored value (:math:`t_i^{\textrm{right censored}}`)".

Since probabilities are between 0 and 1, multiplying many of these results in a very small number.
A loss precision occurs because computers can only store so many decimals. Multiplication is also slower than addition for computers.
To overcome this problem, we can use a logarithm rule to add the log-likelihoods rather than multiply the likelihoods.
We just need to take the log of the likelihood function (the PDF for failure data and the SF for right censored data), evaluate the probability, and sum the values.
The parameters that will maximize the log-likelihood function are the same parameters that will maximize the likelihood function.

An example using the Exponential Distribution
"""""""""""""""""""""""""""""""""""""""""""""

Let's say we have some failure times: t = [27, 64, 3, 18, 8]

We need an initial estimate for time model parameter (:math:`\lambda`) which we would typically get using Least Squares Estimation. For this example, lets start with 0.1 as our first guess for :math:`\lambda`.

For each of these values, we need to calculate the value of the PDF (with the given value of :math:`\lambda`).

Exponential PDF:     :math:`f(t) = \lambda {\rm e}^{-\lambda t}`

Exponential Log-PDF: :math:`ln(f(t)) = ln(\lambda)-\lambda t`

Now we substitute in :math:`\lambda=0.1` and :math:`t = [27, 64, 3, 18, 8]`

.. math::

    \begin{align}
    & L(\lambda=0.1|t=[27, 64, 3, 18, 8]) = \\
    & \qquad (ln(0.1)-0.1 \times 27) + (ln(0.1)-0.1 \times 64) + (ln(0.1)-0.1 \times 3)\\
    & \qquad + (ln(0.1)-0.1 \times 18) + (ln(0.1)-0.1 \times 8)\\
    & = -23.512925
    \end{align}

Here's where the optimization part comes in. We need to vary :math:`\lambda` until we maximize the log-likelihood.
The following graph shows how the log-likelihood varies as :math:`\lambda` varies.

.. image:: images/LL_range.png

This was produced using the following Python code:

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    data = np.array([27, 64, 3, 18, 8])
    
    lambda_array = np.geomspace(0.01, 0.1, 100)
    LL = []
    for L in lambda_array:
        loglik = np.log(L)-L*data
        LL.append(loglik.sum())
    
    plt.plot(lambda_array, LL)
    plt.xlabel('$\lambda$')
    plt.ylabel('Log-likelihood')
    plt.title('Log likelihood over a range of $\lambda$ values')
    plt.show()

The optimization process can be done in Python (using `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_) or in Excel (using `Solver <https://www.wikihow.com/Use-Solver-in-Microsoft-Excel>`_), or a variety of other software packages.
It could even be done by hand, though this is not only tedious, but also limited in practicality to single parameter distributions. 
The optimization process becomes much harder when there are 2 or more parameters that need to be optimized simultaneously.

So, using the above method, we see that the maximum for the log-likelihood occurred when :math:`\lambda` was around 0.041 at a log-likelihood of -20.89.
We can check the value using `reliability` as shown below which achieves an answer of :math:`\lambda = 0.0416667` at a log-likelihood of -20.8903:

.. code:: python

    from reliability.Fitters import Fit_Exponential_1P

    data = [27, 64, 3, 18, 8]
    Fit_Exponential_1P(failures=data,show_probability_plot=False)

    '''
    Results from Fit_Exponential_1P (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Optimizer: TNC
    Failures / Right censored: 5/0 (0% right censored) 
    
    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
       Lambda       0.0416667       0.0186339 0.0173428  0.100105
     1/Lambda              24         10.7331   9.98947   57.6607 
    
    Goodness of fit    Value
     Log-likelihood -20.8903
               AICc  45.1139
                BIC    43.39
                 AD  2.43793 
    '''

Another example using the Exponential Distribution with censored data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Lets use a new dataset that includes both failures and right censored values.

failures = [17, 5, 12] and right_censored = [20, 25]

Once again, we need an initial estimate for the model parameters, and for that we would typically use Least Squares Estimation.
For the purposes of this example, we will again use an initial guess of :math:`\lambda = 0.1`.

For each of the failures, we need to calculate the value of the PDF, and for each of the right censored values, we need to calculate the value of the SF (with the given value of :math:`\lambda`).

Exponential PDF:     :math:`f(t) = \lambda {\rm e}^{-\lambda t}`

Exponential Log-PDF: :math:`ln(f(t)) = ln(\lambda)-\lambda t`

Exponential SF:     :math:`R(t) = {\rm e}^{-\lambda t}`

Exponential Log-SF: :math:`ln(R(t)) = -\lambda t`

Now we substitute in :math:`\lambda=0.1`, :math:`t_{\textrm{failures}} = [17, 5, 12]`, and :math:`t_{\textrm{right censored}} = [20, 25]`.

.. math::

    \begin{align}
    & L(\lambda=0.1|t_{\textrm{failures}}=[17,5,12] {\textrm{ and }}t_{\textrm{right censored}}=[20, 25]) = \\
    & \qquad (ln(0.1)-0.1 \times 17) + (ln(0.1)-0.1 \times 5) + (ln(0.1)-0.1 \times 12)\\
    & \qquad + (-0.1 \times 20) + (-0.1 \times 25)\\
    & = -14.8077528
    \end{align}

Note that the last two terms are the right censored values. Their contribution to the log-likelihood is added in the same way that the contribution from each of the failures is added, except that right censored values use the the log-SF.

As with the previous example, we again need to use optimization to vary :math:`\lambda` until we maximize the log-likelihood.
The following graph shows how the log-likelihood varies as :math:`\lambda` varies.

.. image:: images/LL_range2.png

This was produced using the following Python code:

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    failures = np.array([17,5,12])
    right_censored = np.array([20, 25])
    
    lambda_array = np.geomspace(0.01, 0.1, 100)
    LL = []
    for L in lambda_array:
        loglik_failures = np.log(L)-L*failures
        loglik_right_censored =  -L*right_censored
        LL.append(loglik_failures.sum() + loglik_right_censored.sum())
    
    plt.plot(lambda_array, LL)
    plt.xlabel('$\lambda$')
    plt.ylabel('Log-likelihood')
    plt.title('Log likelihood over a range of $\lambda$ values')
    plt.show()

So, using the above method, we see that the maximum for the log-likelihood occurred when :math:`\lambda` was around 0.038 at a log-likelihood of -12.81.
We can check the value using `reliability` as shown below which achieves an answer of :math:`\lambda = 0.0379747` at a log-likelihood of -12.8125:

.. code:: python

    from reliability.Fitters import Fit_Exponential_1P

    failures = [17,5,12]
    right_censored = [20, 25]
    Fit_Exponential_1P(failures=failures, right_censored=right_censored, show_probability_plot=False)

    '''
    Results from Fit_Exponential_1P (95% CI):
    Analysis method: Maximum Likelihood Estimation (MLE)
    Optimizer: TNC
    Failures / Right censored: 3/2 (40% right censored) 
    
    Parameter  Point Estimate  Standard Error  Lower CI  Upper CI
       Lambda       0.0379747       0.0219247 0.0122476  0.117743
     1/Lambda         26.3333         15.2036   8.49306   81.6483 
    
    Goodness of fit    Value
     Log-likelihood -12.8125
               AICc  28.9583
                BIC  27.2345
                 AD  19.3533 
    '''

An example using the Weibull Distribution
"""""""""""""""""""""""""""""""""""""""""

Because it requires optimization, MLE is only practical using software if there is more than one parameter in the distribution.
The rest of the process is the same, but instead of the likelihood plot (the curves shown above) being a line, for 2 parameters it would be a surface, as shown in the example below.

We'll use the same dataset as in the previous example with failures = [17,5,12] and right_censored = [20, 25].

We also need an estimate for the parameters of the Weibull Distribution. We will use :math:`\alpha=15` and :math:`\beta=2`

For each of the failures, we need to calculate the value of the PDF, and for each of the right censored values, we need to calculate the value of the SF (with the given value of :math:`\lambda`).

Weibull PDF:     :math:`f(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{(\beta-1)}{\rm e}^{-(\frac{t}{\alpha })^ \beta }`

Weibull Log-PDF: :math:`ln(f(t)) = ln\left(\frac{\beta}{\alpha}\right)+(\beta-1).ln\left(\frac{t}{\alpha}\right)-(\frac{t}{\alpha })^ \beta`

Weibull SF:     :math:`R(t) = {\rm e}^{-(\frac{t}{\alpha })^ \beta }`

Weibull Log-SF: :math:`ln(R(t)) = -(\frac{t}{\alpha })^ \beta`

Now we substitute in :math:`\alpha=15`, :math:`\beta=2`, :math:`t_{\textrm{failures}} = [17, 5, 12]`, and :math:`t_{\textrm{right censored}} = [20, 25]` to the log-PDF and log-SF.





How does the optimization routine work in Python
""""""""""""""""""""""""""""""""""""""""""""""""

This will be writted soon.



