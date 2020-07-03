from reliability.Distributions import *
import pytest

weibull_dist = Weibull_Distribution(alpha=5,beta=2)
x5 = weibull_dist.PDF(5,shor_plot=False)

#test 1
#create a Weibull_Distribution and extract the parameters
#this will be finished later
