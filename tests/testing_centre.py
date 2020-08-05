from pytest import main
from reliability.Distributions import Normal_Distribution, Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P

class Test_Distributions:
    def test_Normal_Distribution(self):
        dist = Normal_Distribution(mu=5,sigma=2)
        assert dist.mean == 5
        assert dist.standard_deviation == 2
        assert dist.variance == 4

    def test_Weibull_Distribution(self):
        dist = Weibull_Distribution(alpha=5,beta=2)
        assert dist.mean == 4.4311346272637895
        assert dist.standard_deviation == 2.316256875880522
        assert dist.variance == 5.365045915063796

class Test_Fitters:
    def test_Fit_Weibull_2P(self):
        data = Weibull_Distribution(alpha=50,beta=2).random_samples(20,seed=5)
        fit = Fit_Weibull_2P(failures=data,show_probability_plot=False,print_results=False)
        assert fit.alpha == 47.507980313141516
        assert fit.beta == 2.492960611854891


main(['{}::{}'.format(__file__, Test_Fitters.__name__)])

main(['{}::{}'.format(__file__, Test_Distributions.__name__)])
