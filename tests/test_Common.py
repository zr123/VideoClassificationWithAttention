import pytest
import numpy as np
from VCWA import Common


@pytest.mark.parametrize("original_frames,downsampling_frames", [(100, 40), (69, 25)])
def test_downsample_video(original_frames, downsampling_frames):
    dummyvideo = np.zeros((original_frames, 128, 128, 3), np.uint8)
    downsampled_video = Common.downsample_video(dummyvideo, downsampling_frames)
    assert(downsampled_video.shape == (downsampling_frames, 128, 128, 3))


@pytest.mark.parametrize("L", [1, 5, 10])
def test_calcStackedOpticalFlow(L):
    dummyvideo = np.zeros((40, 128, 128, 3), np.uint8)
    optflowstack = Common.calcStackedOpticalFlow(dummyvideo, L)
    assert(optflowstack.shape == (40-L, 128, 128, 2*L))
