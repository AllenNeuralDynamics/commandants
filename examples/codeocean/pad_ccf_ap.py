"""Pad the CCF template along anterior-posterior only, instead of on every axis.

    python pad_ccf_ap.py                                   # 50 voxels at each AP end
    python pad_ccf_ap.py --anterior 40 --posterior 80      # uneven AP padding
    python pad_ccf_ap.py --other 10                        # plus 10 voxels on each DV/ML face
    python pad_ccf_ap.py --also /data/.../annotation_25.nii.gz   # pad other volumes on the same grid

``ants.pad_image(ccf, pad_width=[100, 100, 100])`` adds 50 voxels to EACH end of EVERY axis,
taking the 25 um CCF from 528 x 320 x 456 (77.0 M voxels) to 628 x 420 x 556 (146.7 M), so every
registration step costs ~1.9x. Padding only AP, 50 per end, gives 628 x 320 x 456 (91.6 M, 1.19x).

The AP padding is identical to the old padding, and so are the physical positions of the CCF's
voxels; only the DV and ML margins go away. Points and transforms expressed in physical space
therefore line up with the old padded grid wherever both grids have voxels.

The AP axis and which end is anterior are read from the image header (the Allen 25 um NIfTI is
"PIR": index 0 runs anterior -> posterior). The script checks this against the image shape
(AP is the CCF's longest axis) and stops if they disagree; --ap-axis overrides.

Needs SimpleITK and numpy only. ANTsPy equivalent for the standard PIR file (axes AP, DV, ML):

    ants.pad_image(ccf, pad_width=[(50, 50), (0, 0), (0, 0)])
"""

import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk

DEFAULT_CCF = "/data/allen_mouse_ccf/average_template/average_template_25.nii.gz"
AP_LETTERS = ("A", "P")


def orientation(img):
    """Per index axis, the anatomical direction the index increases toward (ITK/LPS letters)."""
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(img.GetDirection())


def find_ap_axis(img, forced=None):
    letters = orientation(img)
    from_header = [i for i, c in enumerate(letters) if c in AP_LETTERS]
    extent = [n * s for n, s in zip(img.GetSize(), img.GetSpacing())]
    longest = int(np.argmax(extent))
    if forced is not None:
        if from_header and from_header[0] != forced:
            print("note: header says AP is axis %d (%s), using --ap-axis %d" % (from_header[0], letters, forced))
        return forced, letters
    if len(from_header) != 1 or from_header[0] != longest:
        sys.exit("Header orientation %s puts AP on axis %s, but the longest axis is %d (extents %s mm).\n"
                 "The header may be wrong; check the image and rerun with --ap-axis N."
                 % (letters, from_header, longest, ", ".join("%.2f" % e for e in extent)))
    return longest, letters


def margin_report(img, ap_axis, letters):
    """Empty margin between each face of the input and the first slice holding tissue (Otsu threshold)."""
    a = sitk.GetArrayViewFromImage(img)  # numpy order is (z, y, x): reverse of the index axes
    thr = sitk.OtsuThresholdImageFilter()
    thr.Execute(img)
    tissue = a > thr.GetThreshold()
    spacing = img.GetSpacing()
    print("Empty margin from each face to the tissue (Otsu threshold %.0f):" % thr.GetThreshold())
    for axis in range(3):
        others = tuple(i for i in range(3) if i != 2 - axis)
        rows = np.flatnonzero(tissue.any(axis=others))
        low, high = int(rows[0]), int(len(tissue.any(axis=others)) - 1 - rows[-1])
        print("  axis %d %s: %s end %3d voxels (%.2f mm), %s end %3d voxels (%.2f mm)"
              % (axis, "(AP)" if axis == ap_axis else "    ", opposite(letters[axis]), low, low * spacing[axis],
                 letters[axis], high, high * spacing[axis]))


def opposite(letter):
    return {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}[letter]


def pad(img, lower, upper):
    out = sitk.ConstantPad(img, [int(v) for v in lower], [int(v) for v in upper], 0)
    # The original voxels must keep their values and physical positions.
    core = out[tuple(slice(lo, lo + n) for lo, n in zip(lower, img.GetSize()))]
    assert np.array_equal(sitk.GetArrayViewFromImage(core), sitk.GetArrayViewFromImage(img)), "voxel values moved"
    p_in = img.TransformIndexToPhysicalPoint((0, 0, 0))
    p_out = out.TransformIndexToPhysicalPoint(tuple(int(v) for v in lower))
    assert np.allclose(p_in, p_out, atol=1e-6), "physical positions moved"
    assert out.GetSpacing() == img.GetSpacing() and out.GetDirection() == img.GetDirection()
    return out


def same_grid(a, b):
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing())
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6))


def out_name(path, out_dir, tag):
    base = os.path.basename(path)
    stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]
    return os.path.join(out_dir, "%s_%s.nii.gz" % (stem, tag))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ccf", nargs="?", default=DEFAULT_CCF, help="CCF template (default: %s)" % DEFAULT_CCF)
    ap.add_argument("--anterior", type=int, default=50, help="voxels added at the anterior end (default 50)")
    ap.add_argument("--posterior", type=int, default=50, help="voxels added at the posterior end (default 50)")
    ap.add_argument("--other", type=int, default=0, help="voxels added at each DV and ML face (default 0)")
    ap.add_argument("--also", action="append", default=[], metavar="PATH",
                    help="more volumes on the CCF grid (annotation, masks) to pad identically; repeatable")
    ap.add_argument("--ap-axis", type=int, choices=(0, 1, 2), help="override the AP index axis")
    ap.add_argument("--out-dir", help="where to write (default: /scratch/tmp if it exists, else the current folder)")
    args = ap.parse_args()

    img = sitk.ReadImage(args.ccf)
    ap_axis, letters = find_ap_axis(img, args.ap_axis)
    # Orientation letter = where the index increases toward. "P" means index 0 is the anterior end.
    anterior_low = letters[ap_axis] == "P"
    lower, upper = [args.other] * 3, [args.other] * 3
    lower[ap_axis], upper[ap_axis] = ((args.anterior, args.posterior) if anterior_low
                                      else (args.posterior, args.anterior))

    print("Input  %s" % args.ccf)
    print("  size %s, spacing %s mm, orientation %s; AP is index axis %d (index 0 = %s end)"
          % (img.GetSize(), ", ".join("%.4g" % s for s in img.GetSpacing()), letters, ap_axis,
             "anterior" if anterior_low else "posterior"))
    margin_report(img, ap_axis, letters)

    out = pad(img, lower, upper)
    n_in, n_out = np.prod(img.GetSize()), np.prod(out.GetSize())
    old = [n + 100 for n in img.GetSize()]  # ants.pad_image(pad_width=[100]*3): 50 per end, every axis
    n_old = np.prod(old)
    print("\nPadding (index axes 0, 1, 2): low end %s, high end %s" % (lower, upper))
    print("  %-34s %-18s %7.1f M voxels  %.2fx" % ("unpadded", "x".join(map(str, img.GetSize())), n_in / 1e6, 1.0))
    print("  %-34s %-18s %7.1f M voxels  %.2fx" % ("ants.pad_image(pad_width=[100]*3)", "x".join(map(str, old)),
                                                   n_old / 1e6, n_old / n_in))
    label = "this padding" if args.other else "AP-only padding"
    print("  %-34s %-18s %7.1f M voxels  %.2fx  <- this" % (label, "x".join(map(str, out.GetSize())),
                                                            n_out / 1e6, n_out / n_in))
    print("  Registration time and memory scale roughly with voxels: %.0f%% of the old padded grid."
          % (100.0 * n_out / n_old))

    out_dir = args.out_dir or ("/scratch/tmp" if os.path.isdir("/scratch") else os.getcwd())
    os.makedirs(out_dir, exist_ok=True)
    tag = "padAP%d-%d" % (args.anterior, args.posterior) + ("_other%d" % args.other if args.other else "")
    written = []
    dst = out_name(args.ccf, out_dir, tag)
    sitk.WriteImage(out, dst, useCompression=True)
    written.append(dst)
    for path in args.also:
        extra = sitk.ReadImage(path)
        if not same_grid(extra, img):
            sys.exit("%s is not on the CCF grid (size %s, origin %s); resample it first."
                     % (path, extra.GetSize(), extra.GetOrigin()))
        dst = out_name(path, out_dir, tag)
        sitk.WriteImage(pad(extra, lower, upper), dst, useCompression=True)
        written.append(dst)
    print("\nChecked: original voxels unchanged and at the same physical positions.")
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
