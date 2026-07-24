from pathlib import Path
import shutil
import numpy as np
import nibabel as nib

# Genera el VOI (bounding box 3D) de ground truth de cada paciente a partir de
# la mascara preprocesada. El VOI se guarda con un margen de 8 vox en cada eje,
# y es el que usan luego los scripts de metricas de YOLO (07_metricas_YOLO) para
# comprobar si un VOI predicho contiene al ganglio real.

input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Preprocesados_TFM")
gt_output_root = Path(r"C:\Users\diego\Desktop\TFM\GT_VOIs")

excluidos = ['017']
margen = 8

if gt_output_root.exists():
    shutil.rmtree(gt_output_root)
gt_output_root.mkdir(exist_ok=True)

patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

for patient_dir in patient_dirs:
    if patient_dir.name in excluidos:
        continue

    print(f"paciente {patient_dir.name}")

    mask = nib.load(str(patient_dir / "node_segmentation_resampled_process.nii.gz")).get_fdata() > 0
    img = nib.load(str(patient_dir / "image_resampled_process.nii.gz")).get_fdata()

    N0, N1, N2 = img.shape  # eje0=x (sagital), eje1=y (coronal), eje2=z (axial)

    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0)

    # caja CON margen en los tres ejes
    x_min, x_max = max(0, x0 - margen), min(N0 - 1, x1 + margen)
    y_min, y_max = max(0, y0 - margen), min(N1 - 1, y1 + margen)
    z_min, z_max = max(0, z0 - margen), min(N2 - 1, z1 + margen)

    voi_dir = gt_output_root / patient_dir.name
    voi_dir.mkdir(parents=True, exist_ok=True)

    voi_path = voi_dir / f"gt_voi_{patient_dir.name}.txt"
    with open(voi_path, "w") as f:
        f.write(f"{x_min} {x_max} {y_min} {y_max} {z_min} {z_max}\n")

print("\nGT VOIs generados en:", gt_output_root)
