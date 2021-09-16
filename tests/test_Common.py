import sys
import pytest
import numpy as np

# set path
sys.path.append('../VideoClassificationWithAttention')

import Common


@pytest.mark.parametrize("L", [1, 5, 10])
def test_calcStackedOpticalFlow(L):
    dummyvideo = np.zeros((40, 128, 128, 3), np.uint8)
    optflowstack = Common.calcStackedOpticalFlow(dummyvideo, L)
    assert(optflowstack.shape == (40-L, 128, 128, 2*L))
