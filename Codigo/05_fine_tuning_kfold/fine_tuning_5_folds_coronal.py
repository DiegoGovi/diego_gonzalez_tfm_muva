from ultralytics import YOLO
import torch
import gc

# Fine-tuning 5-fold del plano coronal: parte del peso pre-entrenado en TCIA y
# lo ajusta con cada fold de la cohorte hospital (Yolo_Train_KFold, generado
# por 5_folds_YOLO.py).

BASE = r"C:\Users\diego\Desktop\TFM"
KFOLD_ROOT = rf"{BASE}\Yolo_Train_KFold"
PROJECT_ROOT = rf"{BASE}\FineTuning_YOLO11s_KFold_results"
TCIA_WEIGHTS = rf"{BASE}\TCIA_YOLO11s_results\YOLO11s_scratch_Coronal_TCIA_more_aug\weights\best.pt"


def main():
    print("CUDA:", torch.cuda.is_available())

    for k in range(5):
        torch.cuda.empty_cache()
        gc.collect()

        data_yaml = rf"{KFOLD_ROOT}\fold_{k}\Coronal\data_fold{k}_coronal.yaml"

        model = YOLO(TCIA_WEIGHTS)

        model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=8,
            patience=30,

            optimizer="AdamW",
            lr0=1e-4,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=2,
            weight_decay=1e-3,

            seed=42,
            deterministic=True,
            device=0,

            project=PROJECT_ROOT,
            name=f"YOLO11s_finetune_Coronal_fold{k}",

            plots=True,
            save=True,
            val=True,
            exist_ok=False,
            verbose=True,
            workers=4,
            cache=False,

            # Augmentacion (sin flip: fliplr=0.0 bloqueado para el fine-tuning)
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            fliplr=0.0,
            flipud=0.0,
            translate=0.1,
            scale=0.5,
            degrees=10,
            shear=0.0,
            perspective=0.0,
            mosaic=0.0, close_mosaic=0,
            mixup=0.0, copy_paste=0.0,
            erasing=0.0,
            auto_augment=None,
        )

if __name__ == "__main__":
    main()
