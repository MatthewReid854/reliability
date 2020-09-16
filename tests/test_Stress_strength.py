from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Normal_Distribution
from reliability.Stress_strength import Probability_of_failure, Probability_of_failure_normdist
from numpy.testing import assert_allclose
atol = 1e-8
rtol = 1e-7

def test_Probability_of_failure():
    stress = Weibull_Distribution(alpha=40, beta=4)
    strength = Lognormal_Distribution(mu=1.8, sigma=0.25, gamma=50)
    result = Probability_of_failure(stress=stress, strength=strength, print_results=False, show_distribution_plot=False)
    assert_allclose(result,0.02155359226336879,rtol=rtol,atol=atol)

def test_Probability_of_failure_normdist():
    stress = Normal_Distribution(mu=50, sigma=5)
    strength = Normal_Distribution(mu=80, sigma=7)
    result = Probability_of_failure_normdist(stress=stress, strength=strength, print_results=False, show_distribution_plot=False)
    assert_allclose(result,0.00024384404803800858,rtol=rtol,atol=atol)
