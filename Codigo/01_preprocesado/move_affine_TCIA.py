from pathlib import Path
import shutil
import re
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Empareja las imagenes NIfTI de TCIA (convertidas con dicom_to_nifti_TCIA.py)
# con sus mascaras de ganglios, corrige el volteo de eje Y de la mascara y deja
# ambos en una carpeta por paciente con nombre comun (image.nii.gz / node_segmentation.nii.gz).

img_dir     = Path(r"C:\Users\diego\Desktop\TFM\LymphNodes_NII_TCIA")
mask_root   = Path(r"C:\Users\diego\Desktop\TFM\MED_ABD_LYMPH_MASKS\MED_ABD_LYMPH_MASKS")
output_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_TCIA")

if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True, exist_ok=True)

img_by_id = {}
for im in img_dir.glob("*.nii.gz"):
    stem = im.name.replace(".nii.gz", "")
    mo = re.match(r"(ABD_LYMPH_\d+|MED_LYMPH_\d+)", stem)
    if not mo:
        continue
    pid = mo.group(1)
    if pid not in img_by_id or im.stat().st_size > img_by_id[pid].stat().st_size:
        img_by_id[pid] = im

# mascaras
mask_by_id = {}
for carpeta in mask_root.iterdir():
    if not carpeta.is_dir():
        continue
    pid = carpeta.name  # "MED_LYMPH_002"
    cand = carpeta / f"{pid}_mask.nii.gz"
    if cand.exists():
        mask_by_id[pid] = cand

# match
ids = sorted(set(img_by_id) & set(mask_by_id))

print(f"\nImagenes: {len(img_by_id)} | Mascaras: {len(mask_by_id)} | Emparejados: {len(ids)}")

# id como nombre de carpeta y transformacion affine
for pid in ids:
    pout = output_root / pid
    pout.mkdir(parents=True, exist_ok=True)

    ct_path = img_by_id[pid]
    mask_path = mask_by_id[pid]

    ct = nib.load(str(ct_path))
    mask = nib.load(str(mask_path))

    shutil.copy2(ct_path,  pout / "image.nii.gz")

    mask_data = mask.get_fdata()

    # flip en eje Y respecto al CT (las mascaras de TCIA vienen volteadas en ese eje)
    mask_fixed = np.flip(mask_data, axis=1)

    if ct.shape != mask.shape:
        mask_fixed_nii = nib.Nifti1Image(mask_fixed, ct.affine)
        mask_resampled = resample_from_to(mask_fixed_nii, ct, order=0)
        mask_final = (mask_resampled.get_fdata() > 0).astype(np.uint8)
    else:
        mask_final = (mask_fixed > 0).astype(np.uint8)

    nib.save(
        nib.Nifti1Image(mask_final, ct.affine),
        pout / "node_segmentation.nii.gz"
    )

    print(f"  {pid} OK")

print(f"\nListo: {len(ids)} pacientes en {output_root}")
