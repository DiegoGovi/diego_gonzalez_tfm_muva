"""
metricas_fusion_acotada_YOLO.py
--------------------------------
Evaluacion 5-fold del pipeline YOLO 2.5D multiplano en modo ACOTADO: la
inferencia de cada plano se restringe a los cortes donde hay GT (mas un
margen), en vez de recorrer el volumen entero. Esto es leakage deliberado
(se usa la mascara para elegir donde mirar) y se declara como limitacion:
la precision que da este modo es optimista frente al modo full-volume
(metricas_fusion_YOLO_folds.py), que es el despliegue real.

Mismos parametros bloqueados que el modo full-volume: interseccion TRIPLE,
confianza por plano (axial 0.10, coronal/sagital 0.05), t=4, t_voi=8,
iou_thr=0.3, tol=4.
"""

from pathlib import Path
import sys
import json
import csv
import math
import nibabel as nib
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent / "06_fusion_3D"))

from algoritmo_YOLO import (get_slice_25d, norm_hu, boxes_to_vois, find_vois,
    fusionar_por_iou, suprimir_vois, volumen_voi,
    guardar_vois, guardar_mrk_json, cargar_gt_voi, plot_vois_3d)

# ---- parametros del pipeline (bloqueados) ----
gap = 1
t = 4
t_voi = 8
iou_thr = 0.3
tol = 4
margin_slices = 1

# ---- confianza POR PLANO ----
# axial es el plano fuerte: se deja mas alto para no meter FP.
# coronal y sagital son los flojos: se bajan para arañar detecciones debiles.
conf_ax = 0.1    # axial
conf_co = 0.05   # coronal
conf_sa = 0.05   # sagital

# ---- flags de ploteo ----
plot_flag_3D = True
pacientes_plot = []

# ---- rutas ----
base = Path(r"C:\Users\diego\Desktop\TFM")
pacientes_root = base / "Pacientes_Preprocesados_TFM"
gt_root = base / "GT_VOIs"
ft_root = base / "FineTuning_YOLO11s_KFold_results"
resumen_path = base / "fold_resumen_kfold.json"
out_root = base / "VOIs_predict_KFold_acotado"
csv_path = out_root / "metricas_kfold_acotado.csv"


def predict_plane_test(img_vol, mask_vol, axis, model, conf, gap, margin_slices):
    """
    Infiere SOLO en los cortes donde hay GT (mas un margen). Usa la mascara para
    elegir esos cortes: esto es el leakage del modo acotado, declarado como
    limitacion. 'conf' es el umbral de confianza de ESE plano (puede ser distinto
    por plano). Devuelve {indice_corte: [(x1,y1,x2,y2), ...]}.
    """
    # cortes donde el GT esta presente, a lo largo del eje de este plano
    if axis == 0:
        present = np.where(mask_vol.any(axis=(1, 2)))[0]
    elif axis == 1:
        present = np.where(mask_vol.any(axis=(0, 2)))[0]
    else:
        present = np.where(mask_vol.any(axis=(0, 1)))[0]

    boxes_por_corte = {}
    if present.size == 0:
        return boxes_por_corte

    # expandir cada corte con GT por su margen (margin_slices a cada lado)
    indices = set()
    for idx in present:
        ini = max(0, idx - margin_slices)
        fin = min(mask_vol.shape[axis] - 1, idx + margin_slices)
        for j in range(ini, fin + 1):
            indices.add(j)

    # inferir en cada corte seleccionado
    for i in sorted(indices):
        s_prev, s_curr, s_next = get_slice_25d(img_vol, axis, i, gap)
        rgb = np.stack([norm_hu(s_prev), norm_hu(s_curr), norm_hu(s_next)], axis=-1)
        results = model(rgb, conf=conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            boxes_por_corte[i] = [tuple(map(float, b)) for b in boxes]

    return boxes_por_corte


def fusionar_paciente_test(img_vol, mask_vol, model_ax, model_co, model_sa,
                           t, conf_ax, conf_co, conf_sa, gap, t_voi, iou_thr, margin_slices):
    """
    Pipeline de un paciente con interseccion TRIPLE.
    Cada plano se infiere con SU propio umbral de confianza (conf_ax, conf_co,
    conf_sa). Luego se cruzan los tres (find_vois), se fusionan los que solapan
    mucho y se suprimen los contenidos.
    """
    N0, N1, N2 = img_vol.shape

    # cada plano usa su confianza propia
    boxes_ax = predict_plane_test(img_vol, mask_vol, 2, model_ax, conf_ax, gap, margin_slices)
    boxes_co = predict_plane_test(img_vol, mask_vol, 1, model_co, conf_co, gap, margin_slices)
    boxes_sa = predict_plane_test(img_vol, mask_vol, 0, model_sa, conf_sa, gap, margin_slices)

    # llevar las cajas 2D de cada plano al espacio 3D del volumen
    vois_ax = boxes_to_vois(boxes_ax, 2, N0, N1)
    vois_co = boxes_to_vois(boxes_co, 1, N0, N2)
    vois_sa = boxes_to_vois(boxes_sa, 0, N1, N2)

    # interseccion TRIPLE: un VOI necesita coincidir en los tres planos
    vois_fusion = find_vois(vois_ax, vois_co, vois_sa, t)
    vois_fusion = fusionar_por_iou(vois_fusion, iou_thr)   # une los que solapan mucho
    vois_fusion = suprimir_vois(vois_fusion, t_voi)         # quita los contenidos en otro

    return vois_fusion, (vois_ax, vois_co, vois_sa)


def gt_contenido_en_voi(voi_pred, gt_voi, tol):
    """
    True si el GT cae dentro del VOI predicho. Criterio de contencion: cada
    limite del GT debe quedar entre los limites del predicho, con holgura tol.
    """
    px0, px1, py0, py1, pz0, pz1 = voi_pred
    gx0, gx1, gy0, gy1, gz0, gz1 = gt_voi
    dentro_x = px0 - tol <= gx0 and gx1 <= px1 + tol
    dentro_y = py0 - tol <= gy0 and gy1 <= py1 + tol
    dentro_z = pz0 - tol <= gz0 and gz1 <= pz1 + tol
    return dentro_x and dentro_y and dentro_z


def evaluar_paciente(vois_pred, gt_voi, tol):
    """
    Evalua un paciente (tiene un unico ganglio anotado).
    Si algun VOI predicho contiene el GT: 1 TP, el resto de predichos son FP.
    Si ninguno lo contiene: 1 FN y todos los predichos son FP.
    El error de volumen se mide sobre el VOI acertante.
    """
    acertante = None
    for voi in vois_pred:
        if gt_contenido_en_voi(voi, gt_voi, tol):
            acertante = voi
            break
    if acertante is not None:
        tp = 1
        fp = len(vois_pred) - 1
        fn = 0
        error_vol = volumen_voi(acertante) - volumen_voi(gt_voi)
    else:
        tp = 0
        fp = len(vois_pred)
        fn = 1
        error_vol = None
    return tp, fp, fn, error_vol


def metricas_fold(tp, fp, fn, n_pac):
    """Precision, recall, F1 y FP por paciente de un fold."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_por_paciente = fp / n_pac if n_pac > 0 else 0.0
    return precision, recall, f1, fp_por_paciente


def error_lineal_px(v_errors):
    """Raiz cubica del |error de volumen| en px (1 voxel = 1 px a 1x1x1)."""
    if not v_errors:
        return None, None
    lin = [abs(v) ** (1 / 3) for v in v_errors]
    m = sum(lin) / len(lin)
    var = sum((x - m) ** 2 for x in lin) / len(lin)
    return m, math.sqrt(var)


def procesar_fold(k, test_pacientes):
    print(f"\n===== FOLD {k} (ACOTADO, TRIPLE, conf por plano) =====")

    # modelos del fold k, uno por plano
    model_ax = YOLO(str(ft_root / f"YOLO11s_finetune_Axial_fold{k}"   / "weights" / "best.pt"))
    model_co = YOLO(str(ft_root / f"YOLO11s_finetune_Coronal_fold{k}" / "weights" / "best.pt"))
    model_sa = YOLO(str(ft_root / f"YOLO11s_finetune_Sagital_fold{k}" / "weights" / "best.pt"))

    out_voi_dir = out_root / f"fold_{k}" / "VOIs"
    out_mrk_dir = out_root / f"fold_{k}" / "Slicer"
    out_html_dir = out_root / f"fold_{k}" / "plots_3D"

    tp_total = fp_total = fn_total = 0
    n_con_gt = 0
    v_errors = []

    for paciente in test_pacientes:
        patient_dir = pacientes_root / paciente
        img_nii = nib.load(str(patient_dir / "image_resampled_process.nii.gz"))
        img_vol = img_nii.get_fdata()
        affine = img_nii.affine
        mask_vol = nib.load(str(patient_dir / "node_segmentation_resampled_process.nii.gz")).get_fdata()
        mask_vol = (mask_vol > 0).astype(np.uint8)

        # pipeline del paciente con las tres confianzas
        vois_pred, vois_por_plano = fusionar_paciente_test(
            img_vol, mask_vol, model_ax, model_co, model_sa,
            t, conf_ax, conf_co, conf_sa, gap, t_voi, iou_thr, margin_slices)

        guardar_vois(vois_pred, out_voi_dir, paciente)
        guardar_mrk_json(vois_pred, affine, out_mrk_dir, paciente)

        gt_voi = cargar_gt_voi(gt_root, paciente)

        if plot_flag_3D and (not pacientes_plot or paciente in pacientes_plot):
            out_html_dir.mkdir(parents=True, exist_ok=True)
            out_html = out_html_dir / f"{paciente}_vois.html"
            plot_vois_3d(vois_por_plano, vois_pred, paciente, str(out_html), gt_voi)

        if gt_voi is None:
            print(f"  {paciente}: sin GT, se omite de metricas")
            continue

        tp, fp, fn, error_vol = evaluar_paciente(vois_pred, gt_voi, tol)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        n_con_gt += 1
        if error_vol is not None:
            v_errors.append(error_vol)
        print(f"  {paciente}: TP={tp} FP={fp} FN={fn} | VOIs={len(vois_pred)}")

    precision, recall, f1, fp_pac = metricas_fold(tp_total, fp_total, fn_total, n_con_gt)
    err_m, err_s = error_lineal_px(v_errors)
    err_str = f"{err_m:.2f}+/-{err_s:.2f}" if err_m is not None else "NA"

    print(f"  -> FOLD {k}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
          f"FP/pac={fp_pac:.2f} ErrLin={err_str} px | "
          f"TP={tp_total} FP={fp_total} FN={fn_total}")

    return {
        "fold": k, "n_pac": n_con_gt,
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "precision": precision, "recall": recall, "f1": f1,
        "fp_por_paciente": fp_pac,
        "err_lin_mean": err_m if err_m is not None else "",
        "err_lin_std": err_s if err_s is not None else "",
    }


def media_std(valores):
    """Media y desviacion tipica muestral (N-1) entre folds."""
    n = len(valores)
    m = sum(valores) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in valores) / (n - 1)
    return m, math.sqrt(var)


if __name__ == "__main__":
    with open(resumen_path, "r", encoding="utf-8") as f:
        resumen = json.load(f)

    out_root.mkdir(parents=True, exist_ok=True)

    filas = []
    for k in range(5):
        test_pacientes = resumen[f"fold_{k}"]["test"]
        filas.append(procesar_fold(k, test_pacientes))

    # medias y stds entre folds
    p_m, p_s = media_std([r["precision"] for r in filas])
    r_m, r_s = media_std([r["recall"] for r in filas])
    f_m, f_s = media_std([r["f1"] for r in filas])
    fp_m, fp_s = media_std([r["fp_por_paciente"] for r in filas])
    errs = [r["err_lin_mean"] for r in filas if r["err_lin_mean"] != ""]
    e_m, e_s = media_std(errs) if errs else (None, None)

    def fmt(x):
        # formato Excel ES: coma decimal
        return f"{x:.4f}".replace(".", ",")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["fold", "n_pac", "TP", "FP", "FN",
                         "precision", "recall", "f1", "fp_por_paciente",
                         "err_lin_mean_px", "err_lin_std_px"])
        for r in filas:
            em = fmt(r["err_lin_mean"]) if r["err_lin_mean"] != "" else ""
            es = fmt(r["err_lin_std"]) if r["err_lin_std"] != "" else ""
            writer.writerow([r["fold"], r["n_pac"], r["TP"], r["FP"], r["FN"],
                             fmt(r["precision"]), fmt(r["recall"]),
                             fmt(r["f1"]), fmt(r["fp_por_paciente"]), em, es])
        writer.writerow([])
        writer.writerow(["media", "", "", "", "",
                         fmt(p_m), fmt(r_m), fmt(f_m), fmt(fp_m),
                         fmt(e_m) if e_m is not None else "", ""])
        writer.writerow(["std", "", "", "", "",
                         fmt(p_s), fmt(r_s), fmt(f_s), fmt(fp_s),
                         fmt(e_s) if e_s is not None else "", ""])

    print("\n========== RESUMEN 5-FOLD (ACOTADO, TRIPLE, conf por plano) ==========")
    print(f"conf: ax={conf_ax} co={conf_co} sa={conf_sa}")
    print(f"Precision: {p_m:.3f} +/- {p_s:.3f}")
    print(f"Recall:    {r_m:.3f} +/- {r_s:.3f}")
    print(f"F1:        {f_m:.3f} +/- {f_s:.3f}")
    print(f"FP/pac:    {fp_m:.3f} +/- {fp_s:.3f}")
    if e_m is not None:
        print(f"ErrLin:    {e_m:.2f} +/- {e_s:.2f} px")
    print(f"\nCSV en: {csv_path}")
