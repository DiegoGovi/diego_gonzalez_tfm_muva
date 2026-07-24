"""
metricas_fusion_YOLO_folds.py
------------------------------
Evaluacion 5-fold del pipeline YOLO 2.5D multiplano sobre el VOLUMEN ENTERO
(full volume, SIN mascara GT). Mide el rendimiento real de despliegue: la
precision aqui es la honesta (mas baja que en el modo acotado, que se beneficia
del recorte por mascara).

Respeta los folds: para cada paciente se carga el modelo del fold donde ese
paciente esta en TEST, nunca un modelo que lo haya visto en entrenamiento.

Configuracion final del TFM (bloqueada, no cambiar sin repetir la validacion):
  - interseccion TRIPLE (find_vois, dentro de fusionar_paciente)
  - confianza POR PLANO: axial 0.10, coronal 0.05, sagital 0.05
  - t=4 (holgura del casado entre planos)
  - tol=4 (holgura de la verificacion de contencion en la evaluacion)

Genera, por fold: los txt de VOIs, los .mrk.json de Slicer y los plots 3D.
Y un CSV con precision/recall/F1/FP-pac/error por fold mas media y std.

Criterio de acierto: contencion del GT en algun VOI predicho (no IoU), porque
los VOIs son intencionadamente holgados. Un unico ganglio anotado por paciente.
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
from algoritmo_YOLO import (
    fusionar_paciente, guardar_vois, guardar_mrk_json,
    cargar_gt_voi, volumen_voi, plot_vois_3d,
)

# ---- parametros de fusion (configuracion final del TFM) ----
gap = 1
t = 4            # holgura del casado entre planos
t_voi = 8        # margen de supresion de VOIs contenidos
iou_thr = 0.3    # umbral de fusion de VOIs solapados
tol = 4          # holgura de la verificacion de contencion (evaluacion)

# ---- confianza POR PLANO ----
conf_ax = 0.10   # axial (plano fuerte)
conf_co = 0.05   # coronal (plano flojo)
conf_sa = 0.05   # sagital (plano flojo)

# ---- flags de ploteo ----
plot_flag_3D = True
pacientes_plot = []   # vacio = plotear todos; o ["030", "055"] para filtrar

# ---- rutas ----
base = Path(r"C:\Users\diego\Desktop\TFM")
pacientes_root = base / "Pacientes_Preprocesados_TFM"
gt_root = base / "GT_VOIs"
ft_root = base / "FineTuning_YOLO11s_KFold_results"
resumen_path = base / "fold_resumen_kfold.json"
out_root = base / "VOIs_predict_KFold"
csv_path = out_root / "metricas_kfold_full_volume.csv"


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
    Evalua un paciente (un unico ganglio anotado).
    Si algun VOI contiene el GT: 1 TP, el resto de predichos son FP.
    Si ninguno lo contiene: 1 FN y todos los predichos son FP.
    error_vol = volumen del VOI acertante menos volumen del GT.
    """
    acertante = None
    for voi in vois_pred:
        if gt_contenido_en_voi(voi, gt_voi, tol):
            acertante = voi
            break
    if acertante is not None:
        tp, fp, fn = 1, len(vois_pred) - 1, 0
        error_vol = volumen_voi(acertante) - volumen_voi(gt_voi)
    else:
        tp, fp, fn = 0, len(vois_pred), 1
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
    """
    Error lineal en px: raiz cubica del |error de volumen| (1 voxel = 1 px a
    1x1x1mm). Devuelve media y std poblacional (N) dentro del fold.
    """
    if not v_errors:
        return None, None
    lin = [abs(v) ** (1 / 3) for v in v_errors]
    m = sum(lin) / len(lin)
    var = sum((x - m) ** 2 for x in lin) / len(lin)
    return m, math.sqrt(var)


def procesar_fold(k, test_pacientes):
    print(f"\n===== FOLD {k} =====")

    # modelos del fold k (uno por plano). Cada paciente se evalua solo con el
    # modelo del fold donde esta en test -> sin leakage.
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

        # pipeline full volume con conf por plano (fusionar_paciente de algoritmo_YOLO)
        vois_pred, vois_por_plano = fusionar_paciente(
            img_vol, model_ax, model_co, model_sa,
            t, conf_ax, conf_co, conf_sa, gap, t_voi, iou_thr)

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
    """Media y desviacion tipica muestral (N-1) entre los folds."""
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

    # ---- agregados sobre los 5 folds (media y std muestral N-1) ----
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

    print("\n========== RESUMEN 5-FOLD (FULL VOLUME) ==========")
    print(f"conf: ax={conf_ax} co={conf_co} sa={conf_sa} | t={t} tol={tol} iou={iou_thr}")
    print(f"Precision: {p_m:.3f} +/- {p_s:.3f}")
    print(f"Recall:    {r_m:.3f} +/- {r_s:.3f}")
    print(f"F1:        {f_m:.3f} +/- {f_s:.3f}")
    print(f"FP/pac:    {fp_m:.3f} +/- {fp_s:.3f}")
    if e_m is not None:
        print(f"ErrLin:    {e_m:.2f} +/- {e_s:.2f} px")
    print(f"\nCSV en: {csv_path}")
