from numpy.testing import assert_allclose
import numpy as np

atol = 1e-8
rtol = 1e-7

def test_primary_seed():
    np.random.seed(1)
    num = np.random.random(size=1)[0]
    assert_allclose(num, 0.417022004702574, rtol=rtol, atol=atol)

def test_secondary_seed():
    np.random.seed(1)
    num1 = np.random.randint(low=0, high=1000000, size=1)[0]
    np.random.seed(num1)
    num2 = np.random.random(size=1)[0]
    assert_allclose(num2, 0.584546819599865, rtol=rtol, atol=atol)
