"""Pairwise registration, chaining, blending and accuracy evaluation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .synth import Segment


@dataclass
class PairResult:
    M: np.ndarray | None          # 3x3, segment i+1 coords -> segment i coords
    matches: int
    inliers: int
    seconds: float


@dataclass
class StitchResult:
    method: str
    pairs: list[PairResult]
    global_T: list[np.ndarray] = field(default_factory=list)  # segment -> segment 0 frame
    mosaic: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return all(p.M is not None for p in self.pairs)


def register_pair(matcher, a: Segment, b: Segment, ransac_px: float = 3.0) -> PairResult:
    """Estimate the rotation + scale + translation that maps segment b onto segment a."""
    t0 = time.perf_counter()
    pa, pb = matcher(a.image, b.image, a.valid, b.valid)
    M, inl = (None, None)
    if len(pa) >= 3:
        M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_px, maxIters=5000,
                                             confidence=0.999)
    n_in = int(inl.sum()) if inl is not None else 0
    if M is None or n_in < 8:      # too little support to trust the estimate
        M3 = None
    else:
        M3 = np.vstack([M, [0, 0, 1]])
    return PairResult(M3, len(pa), n_in, time.perf_counter() - t0)


def stitch(segments: list[Segment], matcher, blend: bool = True) -> StitchResult:
    pairs = [register_pair(matcher, segments[i], segments[i + 1]) for i in range(len(segments) - 1)]
    res = StitchResult(matcher.name, pairs)
    if not res.ok:
        return res
    G = [np.eye(3)]
    for p in pairs:
        G.append(G[-1] @ p.M)
    res.global_T = G
    if blend:
        res.mosaic = compose(segments, G)
    return res


def compose(segments: list[Segment], G: list[np.ndarray]) -> np.ndarray:
    """Warp all segments into one canvas and feather-blend the overlaps, so exposure
    differences fade across the seam instead of showing as a hard line."""
    corners = []
    for s, T in zip(segments, G):
        h, w = s.image.shape
        c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
        corners.append((T @ c)[:2].T)
    corners = np.vstack(corners)
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    off = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    size = (x1 - x0, y1 - y0)

    acc = np.zeros(size[::-1], np.float32)
    wsum = np.zeros(size[::-1], np.float32)
    for s, T in zip(segments, G):
        A = (off @ T)[:2]
        img = cv2.warpAffine(s.image.astype(np.float32), A, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(s.valid.astype(np.uint8), A, size, flags=cv2.INTER_NEAREST)
        w = cv2.distanceTransform(mask, cv2.DIST_L2, 5)   # weight grows away from the edge
        acc += img * w
        wsum += w
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


def _grid(shape, n: int = 10) -> np.ndarray:
    h, w = shape
    xs, ys = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n))
    return np.vstack([xs.ravel(), ys.ravel(), np.ones(n * n)])


def pair_errors(segments: list[Segment], result: StitchResult) -> list[float]:
    """Mean pixel error of each estimated pairwise transform against ground truth,
    measured over a grid of points spanning the segment."""
    errs = []
    for i, p in enumerate(result.pairs):
        a, b = segments[i], segments[i + 1]
        if p.M is None:
            errs.append(float("nan"))
            continue
        true_rel = np.linalg.inv(a.true_T) @ b.true_T
        g = _grid(b.image.shape)
        errs.append(float(np.linalg.norm((p.M @ g - true_rel @ g)[:2], axis=0).mean()))
    return errs


def chain_error(segments: list[Segment], result: StitchResult) -> float:
    """Mean error of the LAST segment's placement relative to the first: errors
    accumulate along the chain, so this is what a radiologist would see at the ankle."""
    if not result.ok:
        return float("nan")
    true = np.linalg.inv(segments[0].true_T) @ segments[-1].true_T
    g = _grid(segments[-1].image.shape)
    return float(np.linalg.norm((result.global_T[-1] @ g - true @ g)[:2], axis=0).mean())
