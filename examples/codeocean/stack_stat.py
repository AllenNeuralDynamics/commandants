"""Voxelwise median (or mean) across more images than fit in RAM.

    python stack_stat.py -o template.nii /results/*/[0-9]*_warped.nii
    python stack_stat.py -o mean.nii --stat mean --max-ram-gb 8 inputs.txt

An exact voxelwise median needs every image's value at a voxel at the same time, so the
whole stack cannot be held: N images x 147 M voxels x 4 bytes is 59 GB at N=100 and
985 GB at N=1675. But a *slab* of voxels can be. This reads the same slab out of every
image, reduces it, and moves on, so peak memory is set by the slab, not by N.

That only works on uncompressed images. gzip is a sequential stream, so reading voxel
range [a, b) from a .nii.gz means inflating everything before it -- every slab, every
image. Register with --warped-ext .nii (the default in register_one_gridpoint.py) and
this reads by seeking.

The mean does not need any of this: it is a running sum in one pass with O(1) memory,
and --stat mean takes that path.
"""

import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def collect_inputs(items):
    """Accept paths directly, or a text file with one path per line."""
    if len(items) == 1 and not items[0].endswith((".nii", ".nii.gz")):
        with open(items[0]) as fh:
            return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    return list(items)


def check_geometry(paths):
    """Header-only pass. A sample on the wrong grid is a silent disaster otherwise."""
    ref = nib.load(paths[0])
    shape, affine = ref.shape, ref.affine
    if len(shape) != 3:
        sys.exit("expected 3D images, %s is %dD" % (paths[0], len(shape)))
    bad = []
    compressed = []
    for p in paths:
        img = nib.load(p)
        if img.shape != shape or not np.allclose(img.affine, affine, atol=1e-4):
            bad.append(p)
        if p.endswith(".gz"):
            compressed.append(p)
    if bad:
        sys.exit("these do not share the reference grid %s:\n  %s"
                 % (str(shape), "\n  ".join(bad[:10])))
    return ref, shape, affine, compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image paths, or one text file of paths")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--stat", default="median", choices=["median", "mean"])
    ap.add_argument("--max-ram-gb", default=8.0, type=float,
                    help="budget for the working slab (default 8); the output image is "
                         "held separately")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    paths = collect_inputs(args.inputs)
    if not paths:
        sys.exit("no inputs")
    ref, shape, affine, compressed = check_geometry(paths)
    n = len(paths)
    dtype = np.dtype(args.dtype)

    if compressed and args.stat == "median":
        print("WARNING: %d of %d inputs are gzipped. Slab reads will inflate each file "
              "from the start every time; this will be very slow. Re-run the "
              "registrations with --warped-ext .nii." % (len(compressed), n),
              file=sys.stderr)

    nx, ny, nz = shape
    plane = nx * ny * dtype.itemsize            # bytes per z-slice, per image
    print("%d images, %s voxels each (%s per image as %s)"
          % (n, "x".join(map(str, shape)), human(nx * ny * nz * dtype.itemsize), dtype))

    if args.stat == "mean":
        # One pass, O(1) memory: no slabbing needed at all.
        acc = np.zeros(shape, np.float64)
        t0 = time.time()
        for i, p in enumerate(paths, 1):
            acc += np.asanyarray(nib.load(p).dataobj, dtype=np.float64)
            print("\r  %d/%d" % (i, n), end="", file=sys.stderr)
        print(file=sys.stderr)
        out = (acc / n).astype(dtype)
        print("mean over %d images in %.1fs" % (n, time.time() - t0))
    else:
        budget = max(1, int(args.max_ram_gb * 1024 ** 3))
        slab = max(1, min(nz, budget // (n * plane)))
        print("slab = %d of %d slices -> %s resident per block (budget %.1f GB)"
              % (slab, nz, human(n * plane * slab), args.max_ram_gb))
        if slab == 1 and n * plane > budget:
            print("WARNING: even a single slice needs %s, above the budget."
                  % human(n * plane), file=sys.stderr)

        out = np.empty(shape, dtype)
        block = np.empty((n, nx, ny, slab), dtype)
        k = n // 2
        kth = (k - 1, k) if n % 2 == 0 else k
        t0 = time.time()
        for z0 in range(0, nz, slab):
            z1 = min(z0 + slab, nz)
            view = block[:, :, :, : z1 - z0]
            for i, p in enumerate(paths):
                # Uncompressed: this seeks to the slab instead of reading the whole image.
                view[i] = np.asanyarray(nib.load(p).dataobj[..., z0:z1], dtype=dtype)
            part = np.partition(view, kth, axis=0)
            if n % 2:
                out[..., z0:z1] = part[k]
            else:
                out[..., z0:z1] = 0.5 * (part[k - 1] + part[k])
            done = z1 / nz
            print("\r  %5.1f%%  %.0fs elapsed" % (100 * done, time.time() - t0),
                  end="", file=sys.stderr)
        print(file=sys.stderr)
        print("median over %d images in %.1fs" % (n, time.time() - t0))

    img = nib.Nifti1Image(out, affine, ref.header)
    img.set_data_dtype(dtype)
    nib.save(img, args.output)
    print("wrote %s (%s)" % (args.output, human(os.path.getsize(args.output))))


if __name__ == "__main__":
    main()
