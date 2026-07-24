from pathlib import Path
import shutil
import json

# Convierte la cohorte hospital preprocesada al formato Task de nnDetection
# (imagesTr/labelsTr con nombres case0000, case0001, ...). El split en folds
# y el entrenamiento se hicieron fuera de este repo (instancia Vast.ai, ver
# README de 08_nndetection).

output_root = Path(r"C:\Users\diego\Desktop\TFM\Task100_LymphNodes")
input_root = Path(r"C:\Users\diego\Desktop\TFM\Pacientes_Preprocesados_TFM")

(output_root / "raw_splitted" / "imagesTr").mkdir(parents=True, exist_ok=True)
(output_root / "raw_splitted" / "labelsTr").mkdir(parents=True, exist_ok=True)

patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

mapping = {}
for num, patient_dir in enumerate(patient_dirs):
    id_ = f"case{num:04d}"

    mapping[id_] = patient_dir.name  # seguimiento de que caso es que paciente

    image_path = patient_dir / "image_resampled_process.nii.gz"
    dst_path = output_root / "raw_splitted" / "imagesTr" / f"{id_}_0000.nii.gz"
    shutil.copy(image_path, dst_path)

    node_path = patient_dir / "node_segmentation_resampled_process.nii.gz"
    node_dst = output_root / "raw_splitted" / "labelsTr" / f"{id_}.nii.gz"
    shutil.copy(node_path, node_dst)

    json_dict = {"instances": {"1": 0}}
    json_path = output_root / "raw_splitted" / "labelsTr" / f"{id_}.json"
    json_path.write_text(json.dumps(json_dict, indent=4))

    print(f"{patient_dir.name}  ->  {dst_path.name}")

# mapeo caso nnDet <-> paciente, necesario para volver a leer las metricas por paciente
mapping_path = output_root / "id_nnd_mapping.json"
mapping_path.write_text(json.dumps(mapping, indent=4))
