from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
from scipy.ndimage import label as cc_label
from PIL import Image, ImageDraw

# Construye el dataset de imagenes 2.5D + etiquetas YOLO para la cohorte TCIA,
# a partir de los volumenes preprocesados por pre_process.py.
# Genera, por paciente y por plano, las imagenes (Images), las etiquetas
# YOLO (Labels) y una visualizacion de la caja dibujada (GT).

input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_TCIA_Preprocesados_TFM")
output_root = Path(r"C:\Users\diego\Desktop\TFM\Task_Images_TCIA_YOLO")

if output_root.exists():
    shutil.rmtree(output_root)

output_root.mkdir(exist_ok=True)

patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])


margin_box = 8          # px de margen en el bounding box
margin_slices = 1       # cortes extra por arriba y por abajo
gap = 1                 # separacion entre cortes para 2.5D
min_area = 12           # area minima para ignorar ruido

merge_iou_thr = 0.3
merge_edge_dist_thr = 2

planes = {
    0: "Sagital",
    1: "Coronal",
    2: "Axial"
}


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
    Imagen clipeada en [-150, 250] a uint8.
    """
    hu_min, hu_max = -150, 250

    norm = ((img_in - hu_min) / (hu_max - hu_min) * 255)
    norm = norm.clip(0, 255).astype(np.uint8)

    return norm


def to_png(img_slice):
    """
    Convierte un corte 2D en PNG escala de grises.
    """
    norm = norm_hu(img_slice)
    return Image.fromarray(norm, mode="L")


def to_png_25d(s_prev, s_current, s_next):
    """
    Convierte tres cortes en una imagen RGB 2.5D.
    """
    r = norm_hu(s_prev)
    g = norm_hu(s_current)
    b = norm_hu(s_next)

    rgb = np.stack([r, g, b], axis=-1)

    return Image.fromarray(rgb, mode="RGB")

def box_iou_px(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)

    union = area_a + area_b - inter

    if union == 0:
        return 0.0

    return inter / union

def box_edge_distance_px(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)

    return (dx ** 2 + dy ** 2) ** 0.5


def merge_two_boxes_px(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)

    return x1, y1, x2, y2


def merge_close_boxes_px(boxes, iou_thr, edge_dist_thr):
    """
    Fusiona cajas si su IoU supera el umbral indicado.
    """
    if len(boxes) <= 1:
        return boxes

    boxes = boxes.copy()
    changed = True

    while changed:
        changed = False
        merged_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current = boxes[i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                iou = box_iou_px(current, boxes[j])

                if iou > iou_thr:
                    current = merge_two_boxes_px(current, boxes[j])
                    used[j] = True
                    changed = True

            merged_boxes.append(current)

        boxes = merged_boxes

    return boxes


def box_px_to_yolo(box, H, W):
    x1, y1, x2, y2 = box

    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))

    w = (x2 - x1 + 1) / W
    h = (y2 - y1 + 1) / H

    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H

    return 0, xc, yc, w, h


def draw_yolo_boxes_from_labels(img_path, labels, out_path):
    """
    Dibuja cajas YOLO sobre una imagen PNG.
    labels: [(cls, xc, yc, w, h), ...]
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    W, H = img.size

    for label in labels:
        cls, xc, yc, w, h = label

        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)

        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    img.save(out_path)


def yolo_label_TCIA(mask_slice, min_area, merge_iou_thr, merge_edge_dist_thr):
    """
    Devuelve una lista de bounding boxes YOLO:
    [(cls, xc, yc, w, h), ...]
    """
    H, W = mask_slice.shape
    mask_bin = mask_slice > 0

    labeled, num = cc_label(mask_bin)

    boxes_px = []

    for node in range(1, num + 1):
        node_id = labeled == node
        ys, xs = np.where(node_id)

        if xs.size == 0:
            continue

        area = xs.size

        if area < min_area:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        x1 = max(0, x1 - margin_box)
        x2 = min(W - 1, x2 + margin_box)

        y1 = max(0, y1 - margin_box)
        y2 = min(H - 1, y2 + margin_box)

        boxes_px.append((x1, y1, x2, y2))

    boxes_px = merge_close_boxes_px(
        boxes_px,
        iou_thr=merge_iou_thr,
        edge_dist_thr=merge_edge_dist_thr
    )

    labels = [box_px_to_yolo(box, H, W) for box in boxes_px]

    return labels

def process_plane_TCIA(img_vol, mask_vol, patient_out, axis):
    plane = planes[axis]
    name = patient_out.name

    img_dir = patient_out / plane / "Images"
    lbl_dir = patient_out / plane / "Labels"
    gt_dir = patient_out / plane / "GT"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Cortes donde hay nodulo a lo largo de este eje
    if axis == 0:
        present = np.where(mask_vol.any(axis=(1, 2)))[0]
    elif axis == 1:
        present = np.where(mask_vol.any(axis=(0, 2)))[0]
    else:
        present = np.where(mask_vol.any(axis=(0, 1)))[0]

    if present.size == 0:
        return

    slice_indices = set()

    for idx in present:
        start = max(0, idx - margin_slices)
        end = min(mask_vol.shape[axis] - 1, idx + margin_slices)

        for j in range(start, end + 1):
            slice_indices.add(j)

    slice_indices = sorted(slice_indices)

    for i in slice_indices:
        stem = f"{name}_{plane}_{i}"

        s_prev, s_curr, s_next = get_slice_25d(img_vol, axis, i, gap)

        img_path = img_dir / f"{stem}.png"
        lbl_path = lbl_dir / f"{stem}.txt"
        gt_path = gt_dir / f"{stem}.png"

        to_png_25d(s_prev, s_curr, s_next).save(img_path)

        mask_slice = get_slice(mask_vol, axis, i)

        labels = yolo_label_TCIA(
            mask_slice,
            min_area,
            merge_iou_thr,
            merge_edge_dist_thr
        )

        with open(lbl_path, "w") as f:
            for label in labels:
                f.write(
                    f"{label[0]} {label[1]:.6f} {label[2]:.6f} "
                    f"{label[3]:.6f} {label[4]:.6f}\n"
                )

        draw_yolo_boxes_from_labels(img_path, labels, gt_path)


for patient_dir in patient_dirs:
    print(f"Procesando {patient_dir.name}")

    img_path = patient_dir / "image_resampled_process.nii.gz"
    mask_path = patient_dir / "node_segmentation_resampled_process.nii.gz"

    img_vol = nib.load(str(img_path)).get_fdata()
    mask_vol = nib.load(str(mask_path)).get_fdata()
    mask_vol = (mask_vol > 0).astype(np.uint8)

    patient_out = output_root / patient_dir.name

    for axis in (0, 1, 2):
        process_plane_TCIA(img_vol, mask_vol, patient_out, axis)

    print(f"{patient_dir.name} terminado")
