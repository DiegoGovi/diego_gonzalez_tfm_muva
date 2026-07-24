from pathlib import Path
import nibabel as nib
from PIL import Image
import numpy as np
import shutil

# Construye el dataset de imagenes 2.5D + etiquetas YOLO para la cohorte
# hospital (Ramon y Cajal), a partir de los volumenes preprocesados por
# pre_process.py. Sirve de entrada para el split 5-fold (5_folds_YOLO.py).

input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Preprocesados_TFM")
output_root = Path(r"C:\Users\diego\Desktop\TFM\Task_Images_YOLO")

excluidos = ['017']

if output_root.exists():
    shutil.rmtree(output_root)

output_root.mkdir(exist_ok=True)


patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

margin_box = 8        # px de margen en el bounding box
margin_slices = 1     # cortes extra por arriba y por abajo
gap = 1
planes = {0: "Sagital", 1: "Coronal", 2: "Axial"}

def get_slice(vol, axis, i):
    if axis == 0:      # Sagital
        sl = vol[i, :, :]
    elif axis == 1:    # Coronal
        sl = vol[:, i, :]
    else:              # Axial
        sl = vol[:, :, i]

    sl = np.rot90(sl)
    sl = np.fliplr(sl)

    return sl

def get_slice_25d(vol, axis, i, gap):
    n = vol.shape[axis]
    i_prev = max(0, i - gap)
    i_next = min(n - 1, i + gap)

    s_prev = get_slice(vol, axis, i_prev)
    s_current = get_slice(vol, axis, i)
    s_next = get_slice(vol, axis, i_next)

    return s_prev, s_current, s_next

def norm_hu(img_in):
    """
    Imagen clipeada en [-150, 250] a uint8 escala de grises.
    """

    hu_min, hu_max = -150, 250
    norm = ((img_in - hu_min) / (hu_max - hu_min) * 255).clip(0, 255).astype(np.uint8)

    return norm


def to_png(img_slice):
    """
    Imagen clipeada en [-150, 250] a uint8 escala de grises.
    """

    hu_min, hu_max = -150, 250
    norm = ((img_slice - hu_min) / (hu_max - hu_min) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(norm, mode="L")

def to_png_25d(s_prev, s_current, s_next):
    r = norm_hu(s_prev)
    g = norm_hu(s_current)
    b = norm_hu(s_next)

    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)
    return Image.fromarray(rgb, mode="RGB")


def yolo_label(mask_slice):
    """
    Bounding box YOLO normalizado.
    Devuelve (cls, xc, yc, w, h) o None si no hay nodulo.
    """
    ys, xs = np.where(mask_slice > 0)
    if xs.size == 0:
        return None

    H, W = mask_slice.shape

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin_box)
    x2 = min(W - 1, x2 + margin_box)

    y1 = max(0, y1 - margin_box)
    y2 = min(H - 1, y2 + margin_box)

    w = (x2 - x1 + 1) / W
    h = (y2 - y1 + 1) / H

    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H

    return 0, xc, yc, w, h

def process_plane(img_vol, mask_vol, patient_out, axis):

    plane = planes[axis]
    name = patient_out.name

    img_dir = patient_out / plane / "Images"
    lbl_dir = patient_out / plane / "Labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Cortes donde hay nodulo a lo largo de este eje
    if axis == 0:
        present = np.where(mask_vol.any(axis=(1, 2)))[0]
    elif axis == 1:
        present = np.where(mask_vol.any(axis=(0, 2)))[0]
    else:  # axis == 2
        present = np.where(mask_vol.any(axis=(0, 1)))[0]

    if present.size == 0:
        return

    z0 = max(0, present.min() - margin_slices)
    z1 = min(mask_vol.shape[axis] - 1, present.max() + margin_slices)

    # Genera y guarda imagen + etiqueta de cada corte del rango
    for i in range(z0, z1 + 1):
        stem = f"{name}_{plane}_{i}"

        # Imagen 2.5D, tres cortes (i-gap, i, i+gap) en RGB
        s_prev, s_curr, s_next = get_slice_25d(img_vol, axis, i, gap)
        to_png_25d(s_prev, s_curr, s_next).save(img_dir / f"{stem}.png")

        label = yolo_label(get_slice(mask_vol, axis, i))
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            if label is not None:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}")

for patient_dir in patient_dirs:
    if patient_dir.name in excluidos:
        continue
    print(f"Procesando {patient_dir.name}")

    img_vol = nib.load(str(patient_dir / "image_resampled_process.nii.gz")).get_fdata()
    mask_vol = nib.load(str(patient_dir / "node_segmentation_resampled_process.nii.gz")).get_fdata()
    mask_vol = (mask_vol > 0).astype(np.uint8)

    patient_out = output_root / patient_dir.name

    for axis in (0, 1, 2):
        process_plane(img_vol, mask_vol, patient_out, axis)

    print(f"  {patient_dir.name} terminado")
