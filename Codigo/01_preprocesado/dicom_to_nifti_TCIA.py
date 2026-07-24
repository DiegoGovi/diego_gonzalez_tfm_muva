import subprocess
from pathlib import Path

# Requiere dcm2niix instalado y accesible en el PATH del sistema.

dicom_root = Path(r"C:\Users\diego\Desktop\TFM\ct_lymph_nodes")
out_dir = Path(r"C:\Users\diego\Desktop\TFM\LymphNodes_NII_TCIA")
out_dir.mkdir(parents=True, exist_ok=True)

# Por cada paciente, carpeta con MAS archivos .dcm
# Evitar SEG
pacientes = sorted(p for p in dicom_root.iterdir() if p.is_dir())
print(f"Pacientes encontrados: {len(pacientes)}\n")

ok, fallos = 0, []
for pat in pacientes:
    # contienen .dcm
    series = {}
    for dcm in pat.rglob("*.dcm"):
        series[dcm.parent] = series.get(dcm.parent, 0) + 1

    if not series:
        print(f"  [SIN DICOM] {pat.name}")
        fallos.append(pat.name)
        continue

    # serie CT
    serie_ct = max(series, key=series.get)
    n_ct = series[serie_ct]
    print(f"{pat.name}: serie CT = {serie_ct.name} ({n_ct} cortes)  [otras series: {len(series)-1}]")

    # convertir SOLO esa carpeta, nombrando por PatientID
    r = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", pat.name, "-o", str(out_dir), str(serie_ct)],
        capture_output=True, text=True
    )
    # comprobar que salio el archivo
    if (out_dir / f"{pat.name}.nii.gz").exists():
        ok += 1
    else:
        print(f"   [AVISO] no se genero {pat.name}.nii.gz")
        fallos.append(pat.name)

print(f"\n=== Terminado ===")
print(f"Convertidos OK: {ok}")
if fallos:
    print(f"Con problemas ({len(fallos)}): {fallos}")
n = len(list(out_dir.glob("*.nii.gz")))
print(f"Total .nii.gz en carpeta: {n}")
