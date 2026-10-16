import numpy as np


class GroundEstimator:
    """Blind ground estimate from footholds only (no camera, no LiDAR).

    Raw plane: least-squares fit z = a0 + a1*x + a2*y through the latest confirmed
    touchdown point of each foot (Bledt et al. 2018; Focchi et al. 2017).
    Filtered plane: stored as a height h at an anchor point (the body x, y) plus a slope,
    low-passed toward the raw plane so one odd foothold does not jerk the body.
    """

    def __init__(self, tau=0.2):
        self.tau = tau                  # filter time constant (s)
        self.pts = None                 # 4 x 3, latest touchdown point of each foot (ground level)
        self.raw = np.zeros(3)          # raw plane coefficients a0, a1, a2
        self.anchor = np.zeros(2)       # x, y where the filtered height is stored
        self.h = 0.0                    # filtered ground height at the anchor
        self.slope = np.zeros(2)        # filtered slope: dz/dx, dz/dy

    def reset(self, pts, xy):
        # start (or restart) with the filter already settled on these points
        self.pts = np.array(pts, dtype=float)
        self._fit()
        self.anchor = np.array(xy, dtype=float)
        self.h = self._raw_height(self.anchor)
        self.slope = self.raw[1:3].copy()

    def touchdown(self, i, point):
        # a foot just confirmed contact: its point replaces the old one, then refit
        self.pts[i] = point # type: ignore
        self._fit()

    def update(self, xy, dt):
        # 1) slide the anchor to the body along the filtered plane itself (the plane does not change)
        xy = np.asarray(xy, dtype=float)
        self.h += float(self.slope @ (xy - self.anchor))
        self.anchor = xy.copy()
        # 2) first-order low-pass toward the raw plane, compared at the same point
        k = dt / (self.tau + dt)
        self.h += k * (self._raw_height(xy) - self.h)
        self.slope += k * (self.raw[1:3] - self.slope)

    def height(self, xy):
        # filtered ground height at any x, y
        return self.h + float(self.slope @ (np.asarray(xy, dtype=float) - self.anchor))

    def normal(self):
        # unit normal of the filtered plane, pointing up (used in Piece 3)
        n = np.array([-self.slope[0], -self.slope[1], 1.0])
        return n / np.linalg.norm(n)

    def _fit(self):
        A = np.column_stack([np.ones(len(self.pts)), self.pts[:, 0:2]]) # type: ignore
        self.raw = np.linalg.lstsq(A, self.pts[:, 2], rcond=None)[0] # type: ignore

    def _raw_height(self, xy):
        return float(self.raw[0] + self.raw[1:3] @ xy)
