"""
slicer_nndet.py
----------------
Genera, a partir de las predicciones de nnDetection (fold0) en un umbral de
score fijo, los .mrk.json de ROIs para 3D Slicer y un CSV de evaluacion por
paciente (mismo criterio de contencion que YOLO, tol=4).
"""

from pathlib import Path
import json
import pickle
import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Esquinas de un VOI (x0,x1,y0,y1,z0,z1) en indices de array.
# ---------------------------------------------------------------------------
def voi_to_corners(voi):
    x0, x1, y0, y1, z0, z1 = voi
    return np.array([[x, y, z]
                     for x in (x0, x1)
                     for y in (y0, y1)
                     for z in (z0, z1)], dtype=float)


def voi_a_roi_ras(voi, affine):
    corners_idx = voi_to_corners(voi)
    corners_h = np.hstack([corners_idx, np.ones((8, 1))])
    corners_ras = (affine @ corners_h.T).T[:, :3]
    ras_min = corners_ras.min(axis=0)
    ras_max = corners_ras.max(axis=0)
    centro = (ras_min + ras_max) / 2
    size = ras_max - ras_min
    return centro, size


# ---------------------------------------------------------------------------
# Criterio de contencion (tol=4), mismo que YOLO y que metrics_nndet.py.
# ---------------------------------------------------------------------------
def gt_contenido_en_voi(voi_pred, gt_voi, tol=4):
    px0, px1, py0, py1, pz0, pz1 = voi_pred
    gx0, gx1, gy0, gy1, gz0, gz1 = gt_voi
    return (px0 - tol <= gx0 and gx1 <= px1 + tol and
            py0 - tol <= gy0 and gy1 <= py1 + tol and
            pz0 - tol <= gz0 and gz1 <= pz1 + tol)


# ---------------------------------------------------------------------------
# GT del caso en espacio Task, leido de labelsTs. VOI (x0,x1,y0,y1,z0,z1).
# ---------------------------------------------------------------------------
def cargar_gt_task(task_root, case):
    gt_path = Path(task_root) / "raw_splitted" / "labelsTs" / f"{case}.nii.gz"
    if not gt_path.exists():
        return None
    gd = nib.load(str(gt_path)).get_fdata() > 0
    idx = np.argwhere(gd)
    if idx.size == 0:
        return None
    return (int(idx[:, 0].min()), int(idx[:, 0].max()),
            int(idx[:, 1].min()), int(idx[:, 1].max()),
            int(idx[:, 2].min()), int(idx[:, 2].max()))


# ---------------------------------------------------------------------------
# Cajas del pkl. Formato nnDet: [z1, y1, z2, y2, x1, x2], espacio Task.
# Devuelve lista de (voi, score) por encima del umbral.
# ---------------------------------------------------------------------------
def cargar_vois_nndet(pkl_path, thr):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    boxes = d["pred_boxes"]
    scores = d["pred_scores"]
    labels = d["pred_labels"]

    preds = []
    for b, s, l in zip(boxes, scores, labels):
        if int(l) != 0 or s < thr:
            continue
        z1, y1, z2, y2, x1, x2 = b
        voi = (min(x1, x2), max(x1, x2),
               min(y1, y2), max(y1, y2),
               min(z1, z2), max(z1, z2))
        preds.append((voi, float(s)))
    return preds


# ---------------------------------------------------------------------------
# Genera el .mrk.json de ROIs para Slicer. color es [R,G,B] en 0..1.
# ---------------------------------------------------------------------------
def guardar_mrk_json(vois, affine, output_dir, paciente, color, sufijo):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{sufijo}_{paciente}.mrk.json"

    markups = []
    for n, voi in enumerate(vois):
        centro, size = voi_a_roi_ras(voi, affine)
        markups.append({
            "type": "ROI",
            "coordinateSystem": "RAS",
            "coordinateUnits": "mm",
            "locked": True,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": 1,
            "roiType": "Box",
            "center": [float(centro[0]), float(centro[1]), float(centro[2])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "size": [float(size[0]), float(size[1]), float(size[2])],
            "insideOut": False,
            "controlPoints": [{
                "id": str(n + 1),
                "label": f"nndet_{n + 1}",
                "description": "",
                "associatedNodeID": "",
                "position": [float(centro[0]), float(centro[1]), float(centro[2])],
                "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }],
            "measurements": [{
                "name": "volume",
                "enabled": False,
                "units": "cm3",
                "printFormat": "%-#4.4g%s",
            }],
            "display": {
                "visibility": True,
                "opacity": 0.8,
                "color": [float(color[0]), float(color[1]), float(color[2])],
                "selectedColor": [float(color[0]), float(color[1]), float(color[2])],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": False,
                "pointLabelsVisibility": False,
                "textScale": 0.0,
                "glyphType": "Sphere3D",
                "glyphScale": 0.0,
                "glyphSize": 1.0,
                "useGlyphScale": True,
                "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2200000000000004,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": False,
                "translationHandleVisibility": False,
                "rotationHandleVisibility": False,
                "scaleHandleVisibility": False,
                "interactionHandleScale": 3.0,
                "snapMode": "toVisibleSurface",
            },
        })

    contenido = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/"
                   "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups,
    }

    with open(json_path, "w") as f:
        json.dump(contenido, f, indent=2)


if __name__ == "__main__":

    thr = 0.85   # umbral de score (punto de operacion F1 max de metrics_nndet.py). Baja a 0.1 para ver todo.
    tol = 4

    color_nndet = [0.0, 0.0, 1.0]   # azul
    sufijo = "nndet_predict"

    task_root = Path(r"C:\Users\diego\Desktop\TFM\Task100_LymphNodes_KFOLD")
    pred_dir = Path(r"C:\Users\diego\Desktop\TFM\nnDet_fold0\fold0\test_predictions")
    out_dir = Path(r"C:\Users\diego\Desktop\TFM\nnDet_fold0\mrk_json")
    csv_out = Path(r"C:\Users\diego\Desktop\TFM\nndet_por_paciente.csv")

    imagesTs = task_root / "raw_splitted" / "imagesTs"

    with open(task_root / "id_nnd_mapping.json") as f:
        case_a_paciente = json.load(f)

    tp_total = fp_total = fn_total = 0
    filas = []   # para el csv y el resumen
    n_generados = 0

    print("pac    cajas  TP  FP  FN  score_acierto  contiene_GT")
    for pkl_path in sorted(pred_dir.glob("*_boxes.pkl")):
        case = pkl_path.stem.replace("_boxes", "")
        paciente = case_a_paciente.get(case, case)

        img_path = imagesTs / f"{case}_0000.nii.gz"
        if not img_path.exists():
            print(f"Sin volumen Task para {case}, se omite")
            continue
        affine = nib.load(str(img_path)).affine

        preds = cargar_vois_nndet(pkl_path, thr)
        vois = [voi for voi, s in preds]

        # generar el mrk.json
        guardar_mrk_json(vois, affine, out_dir, paciente, color_nndet, sufijo)
        n_generados += 1

        # evaluacion por paciente contra el GT
        gt_voi = cargar_gt_task(task_root, case)
        acertante_score = None
        if gt_voi is not None:
            for voi, s in preds:
                if gt_contenido_en_voi(voi, gt_voi, tol):
                    acertante_score = s
                    break

        if gt_voi is None:
            tp = fp = fn = 0
            estado = "sin_GT"
        elif acertante_score is not None:
            tp = 1
            fp = len(preds) - 1
            fn = 0
            estado = "SI"
        else:
            tp = 0
            fp = len(preds)
            fn = 1
            estado = "NO"

        tp_total += tp
        fp_total += fp
        fn_total += fn

        sc_str = f"{acertante_score:.3f}" if acertante_score is not None else "-"
        print(f"{paciente:>4}   {len(preds):4d}  {tp:2d}  {fp:3d}  {fn:2d}   "
              f"{sc_str:>8}        {estado}")

        filas.append((paciente, len(preds), tp, fp, fn,
                      sc_str if acertante_score is not None else "", estado))

    # resumen global
    n_pac = len(filas)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_por_pac = fp_total / n_pac if n_pac > 0 else 0.0

    print("\n=== RESUMEN GLOBAL (umbral %.2f) ===" % thr)
    print(f"Pacientes: {n_pac}")
    print(f"TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"FP/paciente: {fp_por_pac:.2f}")

    # csv por paciente (Excel ES: ; y coma decimal)
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("paciente;n_cajas;TP;FP;FN;score_acierto;contiene_GT\n")
        for fila in filas:
            linea = ";".join(str(x) for x in fila).replace(".", ",")
            f.write(linea + "\n")

    print(f"\nGenerados {n_generados} ficheros .mrk.json en {out_dir}")
    print(f"CSV por paciente guardado en {csv_out}")
