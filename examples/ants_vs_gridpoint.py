"""ANTs and gridpoint on one image pair, as a notebook.

Cells are marked with ``# %%``, so this opens as a notebook in VS Code or PyCharm and
converts with ``jupytext --to ipynb examples/ants_vs_gridpoint.py`` -- while still
running top to bottom as an ordinary script::

    python examples/ants_vs_gridpoint.py [fixed.nii.gz moving.nii.gz]

This is the hand-rolled version of what ``commandants.benchmark.compare`` automates. It
is longer but there is nothing hidden in it: build, inspect, run, score.

Needs ANTs (``commandants install-ants``) and ``gridpoint`` on ``$STPTPATH``. With no
image pair on the command line it generates one, so it runs as-is.
"""

# %% Setup
import os
import sys

import numpy as np
import SimpleITK as sitk

from commandants import presets
from commandants.stpt import presets as stpt_presets

OUT = os.path.abspath("ants_vs_gridpoint_out")
ANTS_DIR = os.path.join(OUT, "ants")
GRID_DIR = os.path.join(OUT, "gridpoint")
for d in (OUT, ANTS_DIR, GRID_DIR):
    os.makedirs(d, exist_ok=True)
print("writing to", OUT)


# %% The image pair
# Texture matters here. gridpoint's deformable metric is an *uncentered* cross-
# correlation, which is nearly flat inside large uniform regions, so a phantom of
# constant-valued blocks would understate it. Real anatomy is textured everywhere.
def make_pair(n=96, seed=0):
    """A textured ball, plus a copy bent by a known local (non-affine) warp."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    texture = (
        60 * np.sin(xx / 7) * np.sin(yy / 9) * np.sin(zz / 8)
        + 40 * np.sin(xx / 3 + yy / 4)
        + 30 * np.cos(yy / 5 + zz / 6)
        + 10 * rng.normal(size=(n, n, n))
        + 120
    )
    ball = ((xx - n / 2) / (n * 0.33)) ** 2 + ((yy - n / 2) / (n * 0.30)) ** 2 + ((zz - n / 2) / (n * 0.28)) ** 2
    fixed = sitk.GetImageFromArray(np.ascontiguousarray(np.where(ball < 1, texture, 0).astype(np.float32)))

    # Two cycles across the volume: too local for an affine to absorb, which is exactly
    # the regime a deformable stage exists for.
    dx = 8 * np.sin(4 * np.pi * yy / n) * np.sin(np.pi * zz / n)
    dy = 8 * np.sin(4 * np.pi * xx / n) * np.sin(np.pi * zz / n)
    field = sitk.GetImageFromArray(np.stack([dx, dy, np.zeros_like(dx)], -1).astype(np.float64), isVector=True)
    field.CopyInformation(fixed)
    moving = sitk.Resample(fixed, fixed, sitk.DisplacementFieldTransform(field), sitk.sitkLinear, 0.0)
    return fixed, moving


def pair_from_argv():
    """The two image paths from the command line, or None if there aren't any.

    Deliberately not a ``len(sys.argv)`` check: a Jupyter kernel puts its own
    ``-f /path/to/kernel-<id>.json`` in ``sys.argv``, so counting arguments silently
    hands the flag to the registration as the fixed image.
    """
    candidates = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if len(candidates) >= 2 and all(os.path.isfile(path) for path in candidates[:2]):
        return candidates[0], candidates[1]
    return None


given = pair_from_argv()
if given is not None:
    fixed_path, moving_path = given
else:
    print("no image pair given -- generating one")
    fixed_image, moving_image = make_pair()
    fixed_path = os.path.join(OUT, "fixed.nii.gz")
    moving_path = os.path.join(OUT, "moving.nii.gz")
    sitk.WriteImage(fixed_image, fixed_path)
    sitk.WriteImage(moving_image, moving_path)

print("fixed :", fixed_path)
print("moving:", moving_path)


# %% Build both registrations -- nothing has run yet
# Same schedule on both sides: rigid -> affine -> deformable.
#
# The one API difference worth remembering: ANTs takes an output *prefix* (it prepends it
# to every file it writes), gridpoint takes an output *directory* (it names its own files
# inside). Everything else about the two objects behaves identically.
ants_reg = presets.ccf_stpt(
    fixed_path,
    moving_path,
    os.path.join(ANTS_DIR, "ants_"),
    warped_output=os.path.join(ANTS_DIR, "ants_warped.nii.gz"),
    verbose=True,
)
grid_reg = stpt_presets.ccf_stpt(fixed_path, moving_path, GRID_DIR, verbose=True)

# Always worth a look before a long run -- this is the literal command line.
print("\n--- ANTs ---\n", ants_reg.to_shell(), sep="")
print("\n--- gridpoint ---\n", grid_reg.to_shell(), sep="")


# %% Run ANTs
ants_result = ants_reg.run(stream=True, log_file=os.path.join(ANTS_DIR, "ants.log"))
print("\nANTs exited %d in %.1fs" % (ants_result.returncode, ants_result.duration_seconds))
print("outputs:", ants_result.outputs)


# %% Run gridpoint
grid_result = grid_reg.run(stream=True)
print("\ngridpoint exited %d in %.1fs" % (grid_result.returncode, grid_result.duration_seconds))
print("outputs:", grid_result.outputs)


# %% Score them
# Timing alone cannot tell you a backend was fast because it did nothing, so measure
# agreement with the fixed image too. Normalized cross-correlation: 1.0 is identical.
def ncc(a, b):
    """Normalized cross-correlation between two arrays."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


reference = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path))
print("\n%-12s %10s %10s" % ("", "NCC", "seconds"))
print("%-12s %10.4f %10s" % ("before", ncc(sitk.GetArrayFromImage(sitk.ReadImage(moving_path)), reference), "-"))

warped = {}
for name, result in (("ANTs", ants_result), ("gridpoint", grid_result)):
    # Both backends declare their warped image under the same key, so this loop does not
    # care which one it is holding.
    warped[name] = sitk.GetArrayFromImage(result.load("warped"))
    print("%-12s %10.4f %10.1f" % (name, ncc(warped[name], reference), result.duration_seconds))


# %% Optional: look at a mid-slice
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed -- skipping the figure")
else:
    moving_array = sitk.GetArrayFromImage(sitk.ReadImage(moving_path))
    mid = reference.shape[0] // 2
    panels = [("fixed", reference), ("moving", moving_array)] + list(warped.items())
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, volume) in zip(axes, panels):
        axis.imshow(volume[mid], cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "midslice.png"), dpi=120)
    print("\nwrote", os.path.join(OUT, "midslice.png"))
    plt.show()
