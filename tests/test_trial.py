import pytest
from reliability.Distributions import Normal_Distribution

def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 5
    assert dist.standard_deviation == 2
    assert dist.variance == 4

def test_Normal_Distribution2():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 5
    assert dist.standard_deviation == 2
    assert dist.variance == 4

#need to add more tests.
