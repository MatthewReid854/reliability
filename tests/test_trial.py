#this is my first go at a test
#trying to get this working with travis-ci

import pytest
from reliability.Distributions import Normal_Distribution

def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 5
    assert dist.standard_deviation == 2
    assert dist.variance == 4


def test_Normal_Distribution_willfail():
    dist = Normal_Distribution(mu=5, sigma=2)
    assert dist.mean == 0
    assert dist.standard_deviation == 0
    assert dist.variance == 0