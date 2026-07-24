from pathlib import Path
import random
import shutil
import json

# Construye el dataset de entrenamiento de ultralytics para el pre-entrenamiento
# en TCIA: split 80/20 train/val por paciente (para no mezclar cortes del mismo
# paciente entre train y val) y un data.yaml por plano.

source_root = Path(r"C:\Users\diego\Desktop\TFM\Task_Images_TCIA_YOLO")
dataset_root = Path(r"C:\Users\diego\Desktop\TFM\Yolo_Train_TCIA")
split_path = Path(r"C:\Users\diego\Desktop\TFM\split_map_TCIA.json")

planes = ["Axial", "Coronal", "Sagital"]
splits = ["train", "val"]
seed = 42
val_ratio = 0.2

if dataset_root.exists():
    shutil.rmtree(dataset_root)

# Split 80/20 por paciente
patient_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())

random.seed(seed)
random.shuffle(patient_dirs)

n = len(patient_dirs)
n_val = int(round(n * val_ratio))

val_pat = patient_dirs[:n_val]
train_pat = patient_dirs[n_val:]

split_map = {}
split_map.update({p.name: "val" for p in val_pat})
split_map.update({p.name: "train" for p in train_pat})

print(f"Total: {n} | Train: {len(train_pat)} | Val: {len(val_pat)}")

for p in patient_dirs:
    print(f"  {p.name}: {split_map[p.name]}")

with open(split_path, "w", encoding="utf-8") as f:
    json.dump(split_map, f, indent=4)


# Dir
for plane in planes:
    for split in splits:
        (dataset_root / plane / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / plane / "labels" / split).mkdir(parents=True, exist_ok=True)

# Copiar imagenes y etiquetas
for patient_dir in patient_dirs:
    split = split_map[patient_dir.name]
    for plane in planes:
        img_src = patient_dir / plane / "Images"
        lbl_src = patient_dir / plane / "Labels"
        if not img_src.exists():
            continue

        img_dst = dataset_root / plane / "images" / split
        lbl_dst = dataset_root / plane / "labels" / split

        for png in img_src.glob("*.png"):
            shutil.copy2(png, img_dst / png.name)
            txt = lbl_src / f"{png.stem}.txt"
            if txt.exists():
                shutil.copy2(txt, lbl_dst / txt.name)

# Crea dataset.yaml para ultralytics, uno por plano
for plane in planes:
    yaml_path = dataset_root / plane / f"data_TCIA_{plane.lower()}.yaml"
    contenido = (
        f"path: {(dataset_root / plane).as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['Ganglio']\n"
    )
    yaml_path.write_text(contenido)
    print(f"data.yaml creado: {yaml_path}")
