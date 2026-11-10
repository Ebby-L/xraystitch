"""Cut a full radiograph into overlapping 'exposures' with known misalignment.

Because we create the misalignment ourselves, the true transform of every
segment is known, so stitching accuracy can be measured in pixels instead of
judged by eye.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segment:
    image: np.ndarray          # uint8, what the "detector" saw
    valid: np.ndarray          # bool mask: pixels that came from the patient, not padding
    true_T: np.ndarray         # 3x3 transform, segment pixel coords -> full-image coords


def similarity(angle_deg: float, scale: float, tx: float, ty: float, center=(0.0, 0.0)) -> np.ndarray:
    """3x3 similarity transform (rotation and scale about `center`, then translation)."""
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    T = np.vstack([M, [0, 0, 1]])
    T[0, 2] += tx
    T[1, 2] += ty
    return T


def make_segments(full: np.ndarray, n: int = 3, overlap: float = 0.25, max_rot: float = 2.0,
                  max_shift: float = 15.0, max_scale: float = 0.02, noise: float = 4.0,
                  seed: int = 0) -> list[Segment]:
    """Split `full` vertically into `n` overlapping segments, each with a random pose,
    exposure (gain and gamma) and detector noise."""
    rng = np.random.default_rng(seed)
    H, W = full.shape
    seg_h = int(H / (n - (n - 1) * overlap))
    step = int(seg_h * (1 - overlap))
    segs = []
    for i in range(n):
        y0 = min(i * step, H - seg_h)
        pose = similarity(rng.uniform(-max_rot, max_rot), 1 + rng.uniform(-max_scale, max_scale),
                          rng.uniform(-max_shift, max_shift), rng.uniform(-max_shift, max_shift),
                          center=(W / 2, seg_h / 2))
        T = np.array([[1, 0, 0], [0, 1, y0], [0, 0, 1]], float) @ pose   # segment -> full
        inv = np.linalg.inv(T)[:2]
        img = cv2.warpAffine(full, inv, (W, seg_h), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpAffine(np.ones_like(full), inv, (W, seg_h), flags=cv2.INTER_NEAREST) > 0

        # Different exposure per shot
        gain, gamma = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.2)
        f = (img.astype(np.float32) / 255) ** gamma * gain
        f += rng.normal(0, noise / 255, f.shape)
        img = np.clip(f * 255, 0, 255).astype(np.uint8)
        img[~valid] = 0
        segs.append(Segment(img, valid, T))
    return segs
