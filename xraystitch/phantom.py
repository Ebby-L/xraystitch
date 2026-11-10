"""Procedural long-leg radiograph phantom, so the pipeline runs without patient data.

It is not anatomically exact. It reproduces what matters for stitching: large
low-texture regions, smooth soft-tissue gradients, bright cortical edges and
fine trabecular texture inside bone.
"""
from __future__ import annotations

import cv2
import numpy as np


def _bone(mask: np.ndarray, rng: np.random.Generator, shape) -> np.ndarray:
    """Bone with a bright cortex and textured trabecular interior."""
    h, w = shape
    cortex = mask.astype(np.float32) - cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(np.float32)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.2)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-6)
    return mask * (0.30 + 0.30 * texture) + cortex * 0.22


def long_leg_phantom(height: int = 2400, width: int = 700, seed: int = 0) -> np.ndarray:
    """Return a uint8 grayscale image of a hip-to-ankle radiograph phantom."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), np.float32)
    cx = width // 2

    # Soft tissue: a tapering limb outline with a smooth intensity gradient
    tissue = np.zeros((height, width), np.uint8)
    pts = []
    for y in range(0, height, 20):
        half = int(width * (0.40 - 0.18 * y / height) + 10 * np.sin(y / 180))
        pts.append((cx - half, y))
    pts += [(2 * cx - x, y) for x, y in reversed(pts)]
    cv2.fillPoly(tissue, [np.array(pts, np.int32)], 1)
    tissue = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), 25)
    img += 0.28 * tissue

    bones = np.zeros((height, width), np.uint8)
    # Pelvis fragment and femoral head
    cv2.ellipse(bones, (cx - 40, int(0.06 * height)), (110, 70), 15, 0, 360, 1, -1)
    cv2.circle(bones, (cx + 30, int(0.10 * height)), 55, 1, -1)
    # Femur shaft, slightly oblique
    cv2.line(bones, (cx + 30, int(0.12 * height)), (cx + 5, int(0.47 * height)), 1, 46)
    # Knee: femoral condyles, patella, tibial plateau
    cv2.ellipse(bones, (cx, int(0.49 * height)), (85, 45), 0, 0, 360, 1, -1)
    cv2.ellipse(bones, (cx - 5, int(0.455 * height)), (30, 40), 0, 0, 360, 1, -1)
    cv2.ellipse(bones, (cx, int(0.535 * height)), (80, 35), 0, 0, 360, 1, -1)
    # Tibia and fibula
    cv2.line(bones, (cx - 5, int(0.54 * height)), (cx - 15, int(0.92 * height)), 1, 40)
    cv2.line(bones, (cx + 60, int(0.55 * height)), (cx + 45, int(0.91 * height)), 1, 16)
    # Ankle
    cv2.ellipse(bones, (cx - 10, int(0.95 * height)), (70, 40), 0, 0, 360, 1, -1)
    img += _bone(bones, rng, (height, width))

    # Detector noise and a calibration ruler, common on stitched long-leg films
    img += rng.normal(0, 0.025, img.shape).astype(np.float32)
    for y in range(40, height - 40, 50):
        cv2.line(img, (18, y), (40 if y % 250 == 40 else 30, y), 0.95, 2)
    return np.clip(img * 255, 0, 255).astype(np.uint8)
