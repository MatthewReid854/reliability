from reliability.Probability_plotting import plotting_positions
import random
import numpy as np
from numpy.testing import assert_allclose

f = [5248, 7454, 16890, 17200, 38700, 45000, 49390, 69040, 72280, 131900]
rc = [3961, 4007, 4734, 6054, 7298, 10190, 23060, 27160, 28690, 37100, 40060, 45670, 53000, 67000, 69630, 77350, 78470, 91680, 105700, 106300, 150400]
f_ycheck = [0.0673076923076923, 0.16346153846153846, 0.25961538461538464, 0.3557692307692308, 0.4519230769230769, 0.5480769230769231, 0.6442307692307693, 0.7403846153846154,
            0.8365384615384615, 0.9326923076923076]
rc_ycheck = [0.025587524708983088, 0.0634323945327679, 0.10285413393254378, 0.14227587333231964, 0.19045799926537904, 0.24165150806925462, 0.29650169607340704, 0.3613246455328599,
             0.43335014493225193, 0.6254181433306308]


def test_SortedWithoutRightCentered():
    x, y = plotting_positions(f, preserve_order=True)
    assert_allclose(f, x)
    assert_allclose(y, f_ycheck)


def test_SortedWithRightCentered():
    x, y = plotting_positions(f, rc, preserve_order=True)
    assert_allclose(f, x)
    assert_allclose(y, rc_ycheck)


def test_ReversedWithoutRightCentered():
    f_rev = list(reversed(f))
    f_ycheck_rev = list(reversed(f_ycheck))
    x, y = plotting_positions(f_rev, preserve_order=True)
    assert_allclose(f_rev, x)
    assert_allclose(y, f_ycheck_rev)


def test_ReversedWithRightCentered():
    f_rev = list(reversed(f))
    rc_rev = list(reversed(rc))
    rc_ycheck_rev = list(reversed(rc_ycheck))
    x, y = plotting_positions(f_rev, rc_rev, preserve_order=True)
    assert_allclose(f_rev, x)
    assert_allclose(y, rc_ycheck_rev)


def test_ShuffleWithoutRightCentered():
    f_shuffled = f.copy()
    random.shuffle(f_shuffled)
    ishuffled_f = [f.index(x) for x in f_shuffled]
    f_ycheck_shuffled = np.array(f_ycheck)[ishuffled_f].tolist()
    x, y = plotting_positions(f_shuffled, preserve_order=True)
    assert_allclose(f_shuffled, x)
    assert_allclose(y, f_ycheck_shuffled)


def test_ShuffleWithRightCentered():
    f_shuffled = f.copy()
    random.shuffle(f_shuffled)
    ishuffled_f = [f.index(x) for x in f_shuffled]
    rc_ycheck_shuffled = np.array(rc_ycheck)[ishuffled_f].tolist()
    rc_shuffled = rc.copy()
    random.shuffle(rc_shuffled)

    x, y = plotting_positions(f_shuffled, rc_shuffled, preserve_order=True)
    assert_allclose(f_shuffled, x)
    assert_allclose(y, rc_ycheck_shuffled)
