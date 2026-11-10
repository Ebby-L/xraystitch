import os

import numpy as np
import pytest

from xraystitch.matchers import SIFTMatcher
from xraystitch.phantom import long_leg_phantom
from xraystitch.stitch import chain_error, pair_errors, stitch
from xraystitch.synth import make_segments


@pytest.fixture(scope="module")
def full():
    return long_leg_phantom(height=1800, width=600)


def test_segments_overlap_and_cover(full):
    segs = make_segments(full, n=3, overlap=0.25, seed=0)
    assert len(segs) == 3
    for s in segs:
        assert s.image.shape == s.valid.shape and s.valid.mean() > 0.9


def test_ground_truth_is_consistent(full):
    """A pixel in a segment, mapped by its true transform, lands on the same full-image content."""
    segs = make_segments(full, n=3, noise=0, max_rot=0, max_scale=0, seed=0)
    s = segs[1]
    x, y = 300, 200
    fx, fy, _ = s.true_T @ [x, y, 1]
    assert abs(int(s.image[y, x]) - int(full[int(round(fy)), int(round(fx))])) < 40


def test_sift_stitch_is_accurate(full):
    segs = make_segments(full, n=3, overlap=0.25, seed=1)
    r = stitch(segs, SIFTMatcher())
    assert r.ok and r.mosaic is not None
    assert max(pair_errors(segs, r)) < 2.0
    assert chain_error(segs, r) < 5.0


def test_failure_is_reported_not_hidden(full):
    blank = make_segments(np.zeros_like(full), n=3, seed=0)
    r = stitch(blank, SIFTMatcher())
    assert not r.ok and r.mosaic is None
    assert np.isnan(pair_errors(blank, r)).all()


@pytest.mark.skipif(not os.getenv("RUN_DEEP"), reason="set RUN_DEEP=1 (downloads model weights)")
def test_deep_stitch_is_accurate(full):
    from xraystitch.matchers import DeepMatcher
    segs = make_segments(full, n=3, overlap=0.25, seed=1)
    r = stitch(segs, DeepMatcher())
    assert r.ok and max(pair_errors(segs, r)) < 2.0
