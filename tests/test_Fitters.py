from reliability.Fitters import Fit_Weibull_2P
from reliability.Distributions import Weibull_Distribution


def test_Fit_Weibull_2P():
    data = Weibull_Distribution(alpha=50, beta=2).random_samples(20, seed=5)
    fit = Fit_Weibull_2P(failures=data, show_probability_plot=False, print_results=False)
    assert fit.alpha == 47.507980313141516
    assert fit.beta == 2.492960611854891

#need to add more tests.