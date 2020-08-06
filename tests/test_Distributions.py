from reliability.Distributions import Normal_Distribution, Weibull_Distribution


def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 5
    assert dist.standard_deviation == 2
    assert dist.variance == 4


def test_Weibull_Distribution():
    dist = Weibull_Distribution(alpha=5, beta=2)
    assert dist.mean == 4.4311346272637895
    assert dist.standard_deviation == 2.316256875880522
    assert dist.variance == 5.365045915063796

#need to add more tests.
