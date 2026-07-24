"""
metrics_nndet.py
----------------
Barrido de umbral de score sobre las predicciones de nnDetection (fold0) y
calculo de precision/recall/F1/FP-paciente/error para cada umbral, mas el
punto de operacion de F1 maximo.

Mismo criterio de contencion que YOLO (tol=4), para que la comparacion entre
enfoques sea justa. Limitacion declarada: nnDetection solo se evaluo en el
fold0 (entrenamiento en Vast.ai, instancia ya destruida), no en los 5 folds.
"""

from pathlib import Path
import math
import pickle
import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Criterio de contencion identico al de YOLO (tol=4).
# ---------------------------------------------------------------------------
def gt_contenido_en_voi(voi_pred, gt_voi, tol=4):
    px0, px1, py0, py1, pz0, pz1 = voi_pred
    gx0, gx1, gy0, gy1, gz0, gz1 = gt_voi

    dentro_x = px0 - tol <= gx0 and gx1 <= px1 + tol
    dentro_y = py0 - tol <= gy0 and gy1 <= py1 + tol
    dentro_z = pz0 - tol <= gz0 and gz1 <= pz1 + tol

    return dentro_x and dentro_y and dentro_z


def volumen_voi(voi):
    x0, x1, y0, y1, z0, z1 = voi
    return (x1 - x0) * (y1 - y0) * (z1 - z0)


# ---------------------------------------------------------------------------
# GT en el espacio de la Task, leido de labelsTs. Mismo espacio que el pkl.
# Devuelve VOI (x0,x1,y0,y1,z0,z1) en vox de la Task.
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
# Cajas del pkl. Formato nnDet: [z1, y1, z2, y2, x1, x2].
# Ya estan en el espacio de la Task, sin offset.
# Devuelve lista de (voi, score), voi = (x0,x1,y0,y1,z0,z1) ordenado.
# ---------------------------------------------------------------------------
def cargar_pred_nndet(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    boxes = d["pred_boxes"]
    scores = d["pred_scores"]
    labels = d["pred_labels"]

    preds = []
    for b, s, l in zip(boxes, scores, labels):
        if int(l) != 0:
            continue
        z1, y1, z2, y2, x1, x2 = b
        voi = (min(x1, x2), max(x1, x2),
               min(y1, y2), max(y1, y2),
               min(z1, z2), max(z1, z2))
        preds.append((voi, float(s)))
    return preds


# ---------------------------------------------------------------------------
# Evaluacion de un paciente a un umbral de score.
# ---------------------------------------------------------------------------
def evaluar_paciente_nndet(preds, gt_voi, thr, tol=4):
    vois = [voi for voi, s in preds if s >= thr]

    acertante = None
    for voi in vois:
        if gt_contenido_en_voi(voi, gt_voi, tol):
            acertante = voi
            break

    if acertante is not None:
        tp = 1
        fp = len(vois) - 1
        fn = 0
        error_vol = volumen_voi(acertante) - volumen_voi(gt_voi)
    else:
        tp = 0
        fp = len(vois)
        fn = 1
        error_vol = None

    return tp, fp, fn, error_vol


def metricas_a_umbral(preds_por_paciente, gt_por_paciente, thr, tol=4):
    tp_total = fp_total = fn_total = 0
    v_errors = []

    for case, preds in preds_por_paciente.items():
        gt_voi = gt_por_paciente[case]
        tp, fp, fn, error_vol = evaluar_paciente_nndet(preds, gt_voi, thr, tol)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if error_vol is not None:
            v_errors.append(error_vol)

    n_pac = len(preds_por_paciente)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_por_pac = fp_total / n_pac if n_pac > 0 else 0.0

    if len(v_errors) > 0:
        lin = [abs(v) ** (1 / 3) for v in v_errors]
        mean_lin = sum(lin) / len(lin)
        var_lin = sum((x - mean_lin) ** 2 for x in lin) / len(lin)
        std_lin = math.sqrt(var_lin)
    else:
        mean_lin = float("nan")
        std_lin = float("nan")

    return {"thr": thr, "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "precision": precision, "recall": recall, "f1": f1,
            "fp_por_pac": fp_por_pac, "err_lin_mean": mean_lin, "err_lin_std": std_lin}


if __name__ == "__main__":

    tol = 4

    task_root = Path(r"C:\Users\diego\Desktop\TFM\Task100_LymphNodes_KFOLD")
    pred_dir = Path(r"C:\Users\diego\Desktop\TFM\nnDet_fold0\fold0\test_predictions")
    csv_out = Path(r"C:\Users\diego\Desktop\TFM\nndet_barrido_umbral.csv")

    preds_por_paciente = {}
    gt_por_paciente = {}

    for pkl_path in sorted(pred_dir.glob("*_boxes.pkl")):
        case = pkl_path.stem.replace("_boxes", "")
        gt_voi = cargar_gt_task(task_root, case)
        if gt_voi is None:
            print(f"Sin GT en labelsTs para {case}, se omite")
            continue
        preds_por_paciente[case] = cargar_pred_nndet(pkl_path)
        gt_por_paciente[case] = gt_voi

    print(f"Casos evaluados: {len(preds_por_paciente)}")

    umbrales = [round(t, 2) for t in np.arange(0.0, 1.0, 0.05)]
    resultados = [metricas_a_umbral(preds_por_paciente, gt_por_paciente, thr, tol)
                  for thr in umbrales]
    mejor = max(resultados, key=lambda r: r["f1"])

    print("\nthr   TP  FP    FN  Prec   Rec    F1     FP/pac  ErrLin")
    for r in resultados:
        marca = "  <-- F1 max" if r is mejor else ""
        print(f"{r['thr']:.2f}  {r['tp']:2d}  {r['fp']:4d}  {r['fn']:2d}  "
              f"{r['precision']:.3f}  {r['recall']:.3f}  {r['f1']:.3f}  "
              f"{r['fp_por_pac']:6.2f}   {r['err_lin_mean']:.2f}{marca}")

    print("\n--- Punto de operacion (F1 maximo) ---")
    print(f"thr={mejor['thr']:.2f}  Prec={mejor['precision']:.3f}  "
          f"Rec={mejor['recall']:.3f}  F1={mejor['f1']:.3f}  "
          f"FP/pac={mejor['fp_por_pac']:.2f}  "
          f"ErrLin={mejor['err_lin_mean']:.2f} +/- {mejor['err_lin_std']:.2f} px")

    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("thr;TP;FP;FN;precision;recall;f1;fp_por_pac;err_lin_mean;err_lin_std\n")
        for r in resultados:
            fila = (f"{r['thr']:.2f};{r['tp']};{r['fp']};{r['fn']};"
                    f"{r['precision']:.4f};{r['recall']:.4f};{r['f1']:.4f};"
                    f"{r['fp_por_pac']:.4f};{r['err_lin_mean']:.4f};{r['err_lin_std']:.4f}")
            f.write(fila.replace(".", ",") + "\n")

    print(f"\nCSV guardado en {csv_out}")
