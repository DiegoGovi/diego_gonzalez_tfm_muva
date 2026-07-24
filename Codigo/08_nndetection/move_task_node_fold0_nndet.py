from pathlib import Path
import json
import shutil

# Extrae del Task de nnDetection los casos de test del fold0 (imagesTs/labelsTs),
# a partir de la lista de casos generada al hacer el split de fold0.
#
# NOTA: Task100_LymphNodes_KFOLD es una copia de Task100_LymphNodes (generado
# por format_nndetection.py) sobre la que se aplico el split de fold0 con las
# herramientas propias de nnDetection en la instancia Vast.ai ya destruida.
# test_cases_fold0.json tampoco lo genera ningun script de este repo: es la
# lista de casos de test de ese fold0, producida en ese mismo paso externo.

TASK_ROOT = Path(r"C:\Users\diego\Desktop\TFM\Task100_LymphNodes_KFOLD")
TEST_LIST = Path(r"C:\Users\diego\Desktop\TFM\test_cases_fold0.json")

raw = TASK_ROOT / "raw_splitted"
imagesTr = raw / "imagesTr"
labelsTr = raw / "labelsTr"
imagesTs = raw / "imagesTs"
labelsTs = raw / "labelsTs"

# Cargo la lista de casos de test
with open(TEST_LIST) as f:
    test_cases = json.load(f)["test_cases"]   # ["case0007", "case0013", ...]

print(f"Casos de test a mover: {len(test_cases)}")

imagesTs.mkdir(parents=True, exist_ok=True)
labelsTs.mkdir(parents=True, exist_ok=True)

movidos_img = 0
movidos_lbl = 0
movidos_json = 0
faltan = []

for case in test_cases:
    # Imagen: caseXXXX_0000.nii.gz (sufijo de modalidad)
    img_src = imagesTr / f"{case}_0000.nii.gz"
    # Label: caseXXXX.nii.gz + caseXXXX.json
    lbl_src = labelsTr / f"{case}.nii.gz"
    json_src = labelsTr / f"{case}.json"

    if img_src.exists():
        shutil.move(str(img_src), str(imagesTs / img_src.name))
        movidos_img += 1
    else:
        faltan.append(str(img_src))

    if lbl_src.exists():
        shutil.move(str(lbl_src), str(labelsTs / lbl_src.name))
        movidos_lbl += 1
    else:
        faltan.append(str(lbl_src))

    if json_src.exists():
        shutil.move(str(json_src), str(labelsTs / json_src.name))
        movidos_json += 1
    else:
        faltan.append(str(json_src))

print(f"Imagenes movidas a imagesTs: {movidos_img}")
print(f"Labels .nii.gz movidos a labelsTs: {movidos_lbl}")
print(f"JSON movidos a labelsTs: {movidos_json}")

if faltan:
    print("\nATENCION, no se encontraron estos archivos:")
    for x in faltan:
        print("  ", x)
else:
    print("\nTodos los archivos de test se movieron correctamente.")

# Recuento final
n_tr_img = len(list(imagesTr.glob("*.nii.gz")))
n_tr_lbl = len(list(labelsTr.glob("*.nii.gz")))
n_ts_img = len(list(imagesTs.glob("*.nii.gz")))
n_ts_lbl = len(list(labelsTs.glob("*.nii.gz")))

print("\n=== RECUENTO FINAL ===")
print(f"imagesTr: {n_tr_img}  (debe ser 88)")
print(f"labelsTr: {n_tr_lbl}  (debe ser 88)")
print(f"imagesTs: {n_ts_img}  (debe ser 22)")
print(f"labelsTs: {n_ts_lbl}  (debe ser 22)")
