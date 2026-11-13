"""Register one sample to CCF with gridpoint, for the template-building pipeline.

Replaces the three-call ANTs chain (Rigid on the mask -> Affine -> SyNOnly) with a
single gridpoint invocation. gridpoint runs MI rigid -> MI affine -> discrete symmetric
MRF deformable internally and hands the transform between its own stages, so there is no
``initial_transform`` to pass and nothing to chain by hand.

Outputs land in /results/<sample>/ under the SAME names the ANTs version produced, so
downstream template code needs no changes:

    0GenericAffine.mat        collapsed rigid+affine, maps fixed -> moving
    1Warp.nii.gz              forward field, fixed -> moving
    1InverseWarp.nii.gz       backward field, moving -> fixed
    <sample>_warped.nii       moving resampled onto the fixed grid (uncompressed by
                              default, so the template step can memory-map slabs of it)
    stpt.log                  gridpoint's settings, version and outcome (extra)

Transform lists, in ANTs order (antsApplyTransforms applies -t right to left):

    fwdtransforms = [1Warp.nii.gz, 0GenericAffine.mat]                   moving -> fixed
    invtransforms = [0GenericAffine.mat, 1InverseWarp.nii.gz]            fixed -> moving
                    with whichtoinvert=[True, False]
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import psutil
import SimpleITK as sitk

from commandants.stpt import presets as stpt_presets

logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
                    level=logging.INFO)

# Code Ocean temp hygiene: keep scratch off the small root filesystem.
SCRATCH = "/scratch/tmp"
os.makedirs(SCRATCH, exist_ok=True)
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = SCRATCH
tempfile.tempdir = SCRATCH


def print_memory():
    process = psutil.Process(os.getpid())
    mem = psutil.virtual_memory()
    logging.info("Process: %.1f GB | Available: %.1f GB | Total: %.1f GB",
                 process.memory_info().rss / 1024 ** 3,
                 mem.available / 1024 ** 3,
                 mem.total / 1024 ** 3)


# --------------------------------------------------------------------- options
# The ANTs --reg_iter* knobs have no gridpoint equivalent: the deformable stage is not
# iterative. Its shape knobs are the control-grid schedule and the pyramid depth.
parser = argparse.ArgumentParser()
parser.add_argument("--grid", default="30,15,8,4",
                    help="deformable control-grid spacings in VOXELS of the fixed grid, "
                         "coarsest first. The default is the Allen schedule and is "
                         "correct at 25 um (750/375/200/100 um). Scale it if the fixed "
                         "image is at another resolution.")
parser.add_argument("--levels", default=None, type=int,
                    help="linear-stage multi-resolution levels (driver default 4)")
parser.add_argument("--linear-iterations", default=None, type=int,
                    help="MI optimizer iterations per level (driver default 100)")
parser.add_argument("--pad", default=100, type=int,
                    help="total CCF padding per axis, in voxels, matching "
                         "ants.pad_image(pad_width=[N,N,N]) -- half is added to each side")
parser.add_argument("--threads", default=psutil.cpu_count(logical=False) or 1, type=int)
parser.add_argument("--warped-ext", default=".nii", choices=[".nii", ".nii.gz"],
                    help="extension for the warped image. .nii (the default) is "
                         "uncompressed and so can be memory-mapped a slab at a time by "
                         "the template step; .nii.gz is ~2x smaller but only readable "
                         "front to back.")
parser.add_argument("--no-fixed-mask", action="store_true",
                    help="skip building a CCF brain mask from the annotation volume")
args = parser.parse_args()

print_memory()
data_dir = Path("/data")


# ------------------------------------------------------------------ discovery
logging.info("Finding moving data")
image_file_loc = list(data_dir.rglob("*image.nii.gz"))[0]
mask_file_loc = list(data_dir.rglob("*mask.nii.gz"))[0]

sample_name = str(image_file_loc).split("/")[-1].split("_image")[0]
out_dir = f"/results/{sample_name}/"
os.makedirs(out_dir, exist_ok=True)
logging.info("Sample name: %s", sample_name)
logging.info("Moving image: %s", image_file_loc)
logging.info("Moving mask:  %s", mask_file_loc)


# ---------------------------------------------------------------- fixed image
# gridpoint needs a path, and the fixed grid defines the output space for every sample
# in the template, so the padded CCF is written once and reused. ants.pad_image is kept
# here deliberately: it reproduces the grid the existing template iterations live on,
# byte for byte. (sitk.ConstantPad(img, (50,)*3, (50,)*3, 0.0) is the dependency-free
# equivalent -- verify origin and direction match before switching a running pipeline.)
def resolve_fixed() -> str:
    ccf_file = Path("/data/allen_mouse_ccf/average_template/average_template_25.nii.gz")
    if ccf_file.exists():
        import ants

        ccf = ants.image_read(str(ccf_file))
        padded = ants.pad_image(ccf, pad_width=[args.pad] * 3)
        dst = os.path.join(SCRATCH, f"ccf25_pad{args.pad}.nii.gz")
        if not os.path.exists(dst):
            ants.image_write(padded, dst)
        logging.info("Used padded CCF as fixed: %s -> %s", ccf.shape, padded.shape)
        return dst

    for candidate in (Path("/data/tissuecyte_upsample_to_10.nii.gz"),
                      Path("/data/tissuecyte_upsample_to_10/tissuecyte_upsample_to_10.nii.gz")):
        if candidate.exists():
            logging.info("Used %s as fixed", candidate)
            return str(candidate)

    all_nii = [Path(root) / name
               for root, _, names in os.walk(data_dir, followlinks=True)
               for name in names if name.endswith(".nii.gz")]
    stages = [p for p in all_nii if p.name.startswith("template_stage_")]
    if not stages:
        raise FileNotFoundError(
            "No file matching template_stage_*.nii.gz was found.\n"
            "Accessible NIfTI files were:\n" + "\n".join(str(p) for p in all_nii))
    logging.info("Used %s as fixed", stages[0])
    return str(stages[0])


fixed_path = resolve_fixed()
print_memory()


# ---------------------------------------------------------------------- masks
# gridpoint reads masks as unsigned char, non-zero meaning inside. Threshold BEFORE any
# cast: casting a uint32 annotation volume straight to uint8 wraps every label that is a
# multiple of 256 to zero. Masks are interpreted in physical space, so they need correct
# origin/spacing/direction but not a matching grid -- the resample below is belt and
# braces, and only fires when the geometry actually differs.
# Deliberately NOT under /results: the discovery step above globs *mask.nii.gz across
# /data, and a results directory mounted as data in a later capsule would then find
# these intermediates instead of the sample's own mask.
MASK_DIR = os.path.join(SCRATCH, f"{sample_name}_masks")
os.makedirs(MASK_DIR, exist_ok=True)


def _geometry(path):
    """Header only -- avoids pulling a whole label volume in just to read its grid."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection()


def binarize_mask(src, reference, dst, label):
    img = sitk.ReadImage(str(src))
    mask = sitk.Cast(sitk.NotEqual(img, 0), sitk.sitkUInt8)
    del img

    size, spacing, origin, direction = _geometry(reference)
    same = (tuple(mask.GetSize()) == tuple(size)
            and all(abs(a - b) < 1e-6 for a, b in zip(mask.GetSpacing(), spacing))
            and all(abs(a - b) < 1e-4 for a, b in zip(mask.GetOrigin(), origin))
            and all(abs(a - b) < 1e-6 for a, b in zip(mask.GetDirection(), direction)))
    if not same:
        logging.info("  %s mask grid differs from the reference -- resampling (nearest)", label)
        ref = sitk.Image(size, sitk.sitkUInt8)
        ref.SetSpacing(spacing)
        ref.SetOrigin(origin)
        ref.SetDirection(direction)
        mask = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                             sitk.sitkUInt8)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(mask)
    logging.info("  %-6s mask: %s uint8, covers %.1f%% of the volume",
                 label, mask.GetSize(), 100.0 * stats.GetMean())
    sitk.WriteImage(mask, dst, useCompression=True)
    return dst


logging.info("Building masks in %s", MASK_DIR)
# The moving mask stays in the MOVING image's own space -- gridpoint carries it through
# the linear transform itself and re-displaces it at every pyramid level. Do not pre-warp.
moving_mask_path = binarize_mask(mask_file_loc, image_file_loc,
                                 os.path.join(MASK_DIR, "moving_mask.nii.gz"), "moving")

fixed_mask_path = None
if not args.no_fixed_mask:
    annotation = next(iter(sorted(data_dir.rglob("annotation_25.nii.gz"))), None)
    if annotation is not None:
        fixed_mask_path = binarize_mask(annotation, fixed_path,
                                        os.path.join(MASK_DIR, "ccf_brain_mask.nii.gz"),
                                        "fixed")
    else:
        logging.info("  no annotation_25.nii.gz found; running without a fixed mask")
print_memory()


# --------------------------------------------------------------- registration
grid_reg = stpt_presets.ccf_stpt(
    fixed_path,
    str(image_file_loc),
    out_dir,
    fixed_mask=fixed_mask_path,
    moving_mask=moving_mask_path,
    grid=args.grid,
    levels=args.levels,
    linear_iterations=args.linear_iterations,
    threads=args.threads,
    # Uncompressed, so the template step can memory-map a slab out of each warped image
    # instead of inflating the whole thing. gzip is a sequential stream: reading voxel
    # range [a, b) from a .nii.gz means decompressing everything before it, every time,
    # which turns an out-of-core median into an O(images x slabs x filesize) read.
    # It also removes a couple of minutes of compression per sample.
    warped_ext=args.warped_ext,
    write_deformation_field=True,
    verbose=True,
)

# Always worth a look before a long run -- this is the literal command line.
logging.info("gridpoint command:\n%s", grid_reg.to_shell())

result = grid_reg.run(stream=True)
logging.info("gridpoint exited %d in %.1fs", result.returncode, result.duration_seconds)
print_memory()

if result.returncode != 0:
    logging.error("Registration failed; see %s", result.outputs.get("log"))
    sys.exit(result.returncode)

for key, path in sorted(result.outputs.items()):
    logging.info("  %-13s %s", key, path)


# ----------------------------------------------------- ANTs-compatible layout
# Rename gridpoint's outputs to the names the ANTs version wrote, so every downstream
# consumer of /results/<sample>/ keeps working.
#
# The two displacement fields are renamed as-is: gridpoint already writes them as 3-
# component vector NIfTI on the fixed grid, which is exactly what antsApplyTransforms
# consumes (verified -- that chain reproduces gridpoint's own warped image to NCC 0.9990).
#
# The affine is CONVERTED, not renamed. stpt_Affine.txt is ITK's text transform format;
# a .mat extension dispatches to a different ITK transform reader, so renaming it would
# produce a file that silently fails to load. Reading and rewriting it round-trips the
# parameters and the centre exactly, and ANTsPy reads the result.
def publish_ants_layout():
    affine_mat = os.path.join(out_dir, "0GenericAffine.mat")
    sitk.WriteTransform(sitk.ReadTransform(result.output_path("affine")), affine_mat)
    logging.info("  0GenericAffine.mat   <- converted from stpt_Affine.txt")
    os.remove(result.output_path("affine"))

    renames = [
        ("warp", "1Warp.nii.gz"),
        ("inverse_warp", "1InverseWarp.nii.gz"),
        ("warped", f"{sample_name}_warped{args.warped_ext}"),
    ]
    for key, name in renames:
        if key not in result.outputs:
            continue  # e.g. the fields, when write_deformation_field is off
        src = result.output_path(key)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, name)
        shutil.move(src, dst)
        logging.info("  %-20s <- %s", name, os.path.basename(src))

    return {
        "fwdtransforms": [os.path.join(out_dir, "1Warp.nii.gz"), affine_mat],
        "invtransforms": [affine_mat, os.path.join(out_dir, "1InverseWarp.nii.gz")],
    }


logging.info("Publishing ANTs-compatible names in %s", out_dir)
transforms = publish_ants_layout()
logging.info("fwdtransforms (moving -> fixed): %s", transforms["fwdtransforms"])
logging.info("invtransforms (fixed -> moving): %s  whichtoinvert=[True, False]",
             transforms["invtransforms"])
