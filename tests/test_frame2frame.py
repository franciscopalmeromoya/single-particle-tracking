import pytest 
import numpy as np

from spt import frame2frame

def test_frame2frame():
    locs0 = np.array([[0., 0.],
                     [5., 5.]])
    locs1 = np.array([[1., 0.],])
    max_dist = 2
    lap = frame2frame(locs0, locs1, max_dist)

    assert  == lap.all()