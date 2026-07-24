# 📌 Trabajo de Fin de Máster - MODELO DE DETECCIÓN AUTOMÁTICA DE GANGLIOS EN IMÁGENES DE TOMOGRAFÍA COMPUTARIZADA DE PACIENTES CON CÁNCER COLORRECTAL MEDIANTE ENFOQUES 2.5D Y TRIDIMENSIONALES

Este repositorio contiene el código utilizado en mi **TFM de Visión Artificial**, donde se comparan dos enfoques de detección de ganglios linfáticos en TC toracoabdominopélvico: un pipeline **YOLO 2.5D multiplano con fusión 3D** (mejora de mi TFG de YOLO 2D) y **nnDetection** (detección 3D nativa). Agradecer al Laboratorio de Análisis de Imagen Médica y Biometría de la Universidad Rey Juan Carlos la confianza depositada.

## 📂 Estructura del Repositorio

```
```
/
├── Codigo/
│   ├── 01_preprocesado/             # Conversión de DICOM a NIfTI, resampleo y ventana de intensidad (HU)
│   ├── 02_dataset_TCIA/             # Construcción del dataset YOLO 2.5D de la cohorte pública TCIA
│   ├── 03_entrenamiento_TCIA/       # Pre-entrenamiento de YOLO desde cero sobre TCIA
│   ├── 04_dataset_kfold/            # Construcción del dataset YOLO 2.5D y split en 5 folds (cohorte hospitalaria)
│   ├── 05_fine_tuning_kfold/        # Fine-tuning de YOLO en 5 folds, un modelo por plano
│   ├── 06_fusion_3D/                # Fusión de las detecciones 2.5D de los 3 planos en volúmenes de interés (VOI) 3D
│   ├── 07_metricas_YOLO/            # Evaluación en 5 folds, en modo full-volume y en modo acotado
│   └── 08_nndetection/              # Preparación del Task, métricas y visualización de nnDetection
├── VOIs_predict_YOLO_KFold/         # VOIs predichos por YOLO, organizados por fold (.txt)
├── VOIs_predict_nnDetection_fold0/  # Marcadores de predicción de nnDetection para 3D Slicer (.mrk.json)
├── requirements.txt
└── README.md
```


## 🔗 Descarga de Pesos de los Modelos

Los pesos de YOLO (5 folds × 3 planos + pre-entrenamiento TCIA) y el checkpoint de nnDetection (fold 0) se encuentran en la sección **Releases** del repositorio:

👉 [Pesos de los modelos - GitHub Releases](https://github.com/DiegoGovi/diego_gonzalez_tfm_muva/releases)


---

Si tienes alguna duda o sugerencia, no dudes en contactarme. 🚀
