from pathlib import Path
import random
import shutil
import json

# Reparte los pacientes de la cohorte hospital en 5 folds y construye, para
# cada fold, la estructura de carpetas train/val/test (por plano) que espera
# ultralytics, con su data.yaml. Rotacion: el fold actual es test, el
# siguiente es val, el resto es train.

source_root = Path(r"C:\Users\diego\Desktop\TFM\Task_Images_YOLO")    # los 110 pacientes
dataset_root = Path(r"C:\Users\diego\Desktop\TFM\Yolo_Train_KFold")   # donde se genera la estructura k-fold
split_path = Path(r"C:\Users\diego\Desktop\TFM\fold_map_kfold.json")  # reparto paciente-fold

planes = ["Axial", "Coronal", "Sagital"]
k = 5
seed = 42

# Borro salida
if dataset_root.exists():
    shutil.rmtree(dataset_root)

# pacientes barajados
patients = sorted(p.name for p in source_root.iterdir() if p.is_dir())
random.seed(seed)
random.shuffle(patients)


# fold de cada paciente
fold_of = {pid: (i % k) for i, pid in enumerate(patients)}

# pacientes por fold
n = len(patients)
print(f"Total pacientes: {n} | K={k}")
for fold_id in range(k):
    cnt = sum(1 for v in fold_of.values() if v == fold_id)
    print(f"  Fold {fold_id}: {cnt} pacientes")


# reparto en json
with open(split_path, "w", encoding="utf-8") as f:
    json.dump(fold_of, f, indent=4)
print(f"\nReparto guardado en: {split_path}")


# rol de un paciente dentro de un fold concreto:
#   - si su fold es el actual: test
#   - si su fold es el siguiente: val
#   - en cualquier otro caso: train
def role_for(pid_fold, fold_actual, total_folds):
    if pid_fold == fold_actual:
        return "test"
    elif pid_fold == (fold_actual + 1) % total_folds:
        return "val"
    else:
        return "train"


# Estructura de carpetas:
# dataset_root / fold_i / plano / images|labels / train|val|test
for i in range(k):
    for plane in planes:
        for split in ("train", "val", "test"):
            (dataset_root / f"fold_{i}" / plane / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / f"fold_{i}" / plane / "labels" / split).mkdir(parents=True, exist_ok=True)


# Por cada paciente copio imagenes y etiquetas a su rol en cada fold
for pid in patients:
    pid_fold = fold_of[pid]
    patient_dir = source_root / pid

    for i in range(k):
        split = role_for(pid_fold, i, k)

        for plane in planes:
            img_src = patient_dir / plane / "Images"
            lbl_src = patient_dir / plane / "Labels"

            img_dst = dataset_root / f"fold_{i}" / plane / "images" / split
            lbl_dst = dataset_root / f"fold_{i}" / plane / "labels" / split

            for png in img_src.glob("*.png"):
                shutil.copy2(png, img_dst / png.name)
                txt = lbl_src / f"{png.stem}.txt"
                if txt.exists():
                    shutil.copy2(txt, lbl_dst / txt.name)


# data.yaml para cada fold y cada plano
for i in range(k):
    for plane in planes:
        base = dataset_root / f"fold_{i}" / plane
        yaml_path = base / f"data_fold{i}_{plane.lower()}.yaml"
        contenido = (
            f"path: {base.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"nc: 1\n"
            f"names: ['Ganglio']\n"
        )
        yaml_path.write_text(contenido)


print("\nEstructura k-fold creada. data.yaml por fold y plano listos.")

resumen = {}  # tambien lo guardo en json para la memoria

for i in range(k):
    train_pat = sorted(pid for pid in patients if role_for(fold_of[pid], i, k) == "train")
    val_pat = sorted(pid for pid in patients if role_for(fold_of[pid], i, k) == "val")
    test_pat = sorted(pid for pid in patients if role_for(fold_of[pid], i, k) == "test")

    print(f"\nFOLD {i}  (train={len(train_pat)}, val={len(val_pat)}, test={len(test_pat)})")
    print(f"  TEST : {test_pat}")
    print(f"  VAL  : {val_pat}")
    print(f"  TRAIN: {train_pat}")

    resumen[f"fold_{i}"] = {
        "train": train_pat,
        "val": val_pat,
        "test": test_pat,
    }

# Guardo el resumen completo en json (lo usan los scripts de metricas)
resumen_path = split_path.parent / "fold_resumen_kfold.json"
with open(resumen_path, "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=4)
print(f"\nResumen detallado guardado en: {resumen_path}")
