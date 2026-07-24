from ultralytics import YOLO
import torch
import gc

# Entrenamiento desde cero (scratch) del plano sagital sobre TCIA, para
# pre-entrenar el YOLO que luego se hace fine-tuning con la cohorte hospital.

DATA_YAML = r"C:\Users\diego\Desktop\TFM\Yolo_Train_TCIA\Sagital\data_TCIA_sagital.yaml"
PROJECT_ROOT = r"C:\Users\diego\Desktop\TFM\TCIA_YOLO11s_results"
RUN_NAME = "YOLO11s_scratch_Sagital_TCIA_more_aug_flip"


def main():
    print("CUDA:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    gc.collect()

    model = YOLO("yolo11s.yaml")

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        patience=60,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=1e-3,
        dropout=0.2,
        pretrained=False,
        seed=42,
        deterministic=True,
        device=0,
        project=PROJECT_ROOT,
        name=RUN_NAME,
        plots=True, save=True, val=True, exist_ok=True, verbose=True,
        workers=4, cache=False,

        # Augmentacion
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        fliplr=0.5,
        flipud=0.0,
        translate=0.1,
        scale=0.5,
        degrees=10,
        shear=0.1,
        perspective=0.0,
        mosaic=0.5, close_mosaic=10,
        mixup=0.1, copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
    )

if __name__ == '__main__':
    main()
