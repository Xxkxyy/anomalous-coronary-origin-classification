"""Generate synthetic NIfTI volumes for a quick smoke test.

Creates a few small 3D volumes with simple shapes (ellipsoid / ring),
saved as .nii.gz files under sample_data/. They are NOT medical images;
they exist only to verify the inference entry point runs end-to-end.

Usage:
    python tools/make_synthetic_data.py
"""
import os

import nibabel as nib
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "sample_data")
SHAPE = (160, 160, 8)


def _ellipsoid(shape, center, radii, sigma=2.0):
    """Binary (with soft edges) ellipsoid inside a volume."""
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    d2 = ((x - center[0]) / radii[0]) ** 2 + \
         ((y - center[1]) / radii[1]) ** 2 + \
         ((z - center[2]) / radii[2]) ** 2
    mask = d2 <= 1.0
    vol = np.zeros(shape, dtype=np.float32)
    vol[mask] = 1.0
    # simple smoothing to mimic soft tissue gradients
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol, sigma=sigma)


def _ring(shape, center, r_outer, r_inner, sigma=2.0):
    """Annular (ring) shape - distinct from the solid ellipsoid."""
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    d2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    mask = (d2 <= r_outer ** 2) & (d2 >= r_inner ** 2)
    vol = np.zeros(shape, dtype=np.float32)
    vol[mask] = 1.0
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol, sigma=sigma)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vol0 = _ellipsoid(SHAPE, center=(80, 80, 4), radii=(25, 35, 4))
    vol1 = _ring(SHAPE, center=(80, 80, 4), r_outer=40, r_inner=30)

    # Add mild per-slice noise for realism.
    rng = np.random.default_rng(0)
    for name, vol in [("synthetic_case_000.nii.gz", vol0),
                      ("synthetic_case_001.nii.gz", vol1)]:
        noisy = vol + 0.05 * rng.standard_normal(SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(noisy, np.eye(4)), os.path.join(OUT_DIR, name))
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
