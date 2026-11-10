"""Feature matchers: classical SIFT and learned DISK + LightGlue.

Both return matched point pairs (N x 2 arrays, x/y) between two grayscale images,
ignoring keypoints near padding borders, which would otherwise produce false matches.
"""
from __future__ import annotations

import cv2
import numpy as np

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _keep(pts: np.ndarray, valid: np.ndarray | None, margin: int = 8) -> np.ndarray:
    if valid is None or len(pts) == 0:
        return np.ones(len(pts), bool)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    x = np.clip(pts[:, 0].round().astype(int), 0, valid.shape[1] - 1)
    y = np.clip(pts[:, 1].round().astype(int), 0, valid.shape[0] - 1)
    return inner[y, x]


class SIFTMatcher:
    """Classical baseline: CLAHE contrast enhancement, SIFT and Lowe's ratio test."""
    name = "SIFT"

    def __init__(self, ratio: float = 0.75, nfeatures: int = 4000):
        self.ratio = ratio
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def __call__(self, a, b, valid_a=None, valid_b=None):
        ka, da = self.sift.detectAndCompute(CLAHE.apply(a), None)
        kb, db = self.sift.detectAndCompute(CLAHE.apply(b), None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        good = [m for m, n in self.bf.knnMatch(da, db, k=2) if m.distance < self.ratio * n.distance]
        pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
        keep = _keep(pa, valid_a) & _keep(pb, valid_b)
        return pa[keep], pb[keep]


class DeepMatcher:
    """Learned features (DISK) matched with a learned matcher (LightGlue), via Kornia."""
    name = "DISK + LightGlue"

    def __init__(self, num_features: int = 2048, device: str | None = None):
        import kornia.feature as KF
        import torch
        self.torch, self.KF = torch, KF
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.lg = KF.LightGlueMatcher("disk").to(self.device).eval()
        self.n = num_features

    def _features(self, img):
        t = self.torch.from_numpy(CLAHE.apply(img)).float().div(255)
        t = t[None, None].repeat(1, 3, 1, 1).to(self.device)  # DISK expects RGB
        f = self.disk(t, self.n, pad_if_not_divisible=True)[0]
        return f.keypoints, f.descriptors

    def __call__(self, a, b, valid_a=None, valid_b=None):
        torch, KF = self.torch, self.KF
        with torch.inference_mode():
            ka, da = self._features(a)
            kb, db = self._features(b)
            lafs_a = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=self.device))
            lafs_b = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=self.device))
            _, idx = self.lg(da, db, lafs_a, lafs_b,
                             hw1=torch.tensor(a.shape, device=self.device),
                             hw2=torch.tensor(b.shape, device=self.device))
        idx = idx.cpu().numpy()
        pa = ka.cpu().numpy()[idx[:, 0]].reshape(-1, 2)
        pb = kb.cpu().numpy()[idx[:, 1]].reshape(-1, 2)
        keep = _keep(pa, valid_a) & _keep(pb, valid_b)
        return pa[keep], pb[keep]
