from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output, resample_from_to
import shutil

# Preprocesado comun a las dos cohortes (hospital Ramon y Cajal y TCIA):
# resamplea a 1x1x1 mm y aplica ventana HU de interes anatomico [-150, 250].
# Se ejecuta DOS VECES, una por cohorte, cambiando las rutas de abajo.

# ---- Cohorte TCIA  ----
input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_TCIA")
output_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_TCIA_Preprocesados_TFM")

# ---- Cohorte hospital (Ramon y Cajal): descomentar estas dos y comentar las de arriba ----
# input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Yolo")
# output_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Preprocesados_TFM")

image_names = ["image.nii.gz"]
mask_names = ["node_segmentation.nii.gz"]

# Paciente TCIA excluido por tener muy pocos cortes
exclude_patients = {"MED_LYMPH_021"}

if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True, exist_ok=True)


patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

for patient_dir in patient_dirs:
    if patient_dir.name in exclude_patients:
        continue
    print(f"Transformando paciente {patient_dir.name}")
    image_path = patient_dir / image_names[0]
    mask_path = patient_dir / mask_names[0]

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    image_nii = nib.as_closest_canonical(image_nii)
    mask_nii = nib.as_closest_canonical(mask_nii)

    # Muestreamos CT 1x1x1 mm (order 1, lineal)
    image_resampled = resample_to_output(image_nii, voxel_sizes=(1.0, 1.0, 1.0), order=1)

    # Muestreamos mascara, con relacion a CT
    mask_resampled = resample_from_to(mask_nii, image_resampled, order=0)

    if image_resampled.shape[:3] != mask_resampled.shape[:3]:
        print(f"Paciente {patient_dir.name} con distintas dimensiones.")
        break

    mask_data = np.asanyarray(mask_resampled.dataobj)  # Obtener imagen
    mask_bin = (mask_data > 0).astype(np.uint8)  # Blanco 1, matriz 3d binaria

    print("Valores mascara binaria:", np.unique(mask_bin))

    if np.sum(mask_bin) == 0:
        print(f"Paciente {patient_dir.name} con mascara vacia")
        break

    # enventanado interes anatomico
    image_data = np.asanyarray(image_resampled.dataobj)
    image_final = np.clip(image_data, -150, 250)

    # Save
    patient_out = output_root / patient_dir.name
    patient_out.mkdir(exist_ok=True)

    # Affine
    affine = image_resampled.affine

    nib.save(nib.Nifti1Image(image_final, affine), patient_out / "image_resampled_process.nii.gz")
    nib.save(nib.Nifti1Image(mask_bin.astype(np.uint8), affine),
             patient_out / "node_segmentation_resampled_process.nii.gz")

    print(f"Paciente {patient_dir.name} guardado")
    print()
